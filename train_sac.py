import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import json
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange
from torch.profiler import profile, record_function, ProfilerActivity
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Device Configuration ---
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("--- Using device: MPS (Apple Silicon GPU) ---")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("--- Using device: CUDA ---")
else:
    device = torch.device("cpu")
    logging.info("--- Using device: CPU ---")

# --- Hyperparameters ---
WINDOW_SIZE = 50
FEATURES_PER_STEP = 7
STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + 2
ACTION_SIZE = 3  # Buy, Sell, Hold
BATCH_SIZE = 256
EPISODES = 200
GAMMA = 0.99
LEARNING_RATE = 3e-4
REPLAY_BUFFER_SIZE = 10000
TRADING_FEE = 0.0035
VALIDATION_INTERVAL = 10
PATIENCE = 50
PORTFOLIO_HISTORY_SIZE = 365
EPSILON_NUMERIC = 1e-9
TAU = 0.005  # Soft update rate for target critics
ALPHA = 0.2  # Initial temperature for entropy regularization
TARGET_ENTROPY = np.float32(-np.log(1.0 / ACTION_SIZE) * 0.98)  # Target entropy for discrete actions

class CryptoTradingEnv:
    """Environment for cryptocurrency trading using daily OHLCV data."""
    def __init__(self, df, feature_tensor, global_mean, global_std):
        self.df = df.reset_index(drop=True)
        self.feature_tensor = feature_tensor.to(device)
        self.current_step = WINDOW_SIZE
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.btc_held = 0
        self.max_steps = len(self.df)
        self.portfolio_values = deque(maxlen=PORTFOLIO_HISTORY_SIZE)
        self.global_mean = global_mean.to(device) if global_mean is not None else None
        self.global_std = global_std.to(device) if global_std is not None else None

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = WINDOW_SIZE
        self.balance = self.initial_balance
        self.btc_held = 0
        self.portfolio_values.clear()
        self.portfolio_values.append(self.initial_balance)
        return self._get_state()

    def _get_state(self):
        """Get current state (normalized features + balance, BTC held)."""
        start_idx = max(0, self.current_step - WINDOW_SIZE)
        end_idx = self.current_step
        
        features_window = self.feature_tensor[start_idx:end_idx]

        if self.global_mean is not None and self.global_std is not None:
            normalized_features = (features_window - self.global_mean) / self.global_std
        else:
            features_mean = features_window.mean(dim=0)
            features_std = features_window.std(dim=0) + EPSILON_NUMERIC
            normalized_features = (features_window - features_mean) / features_std
            
        state = torch.cat([normalized_features.flatten(), torch.tensor([self.balance / self.initial_balance, self.btc_held], device=device, dtype=torch.float32)])
        return state.cpu().numpy()

    def step(self, action):
        """Execute action (0: Hold, 1: Buy, 2: Sell) and return next state, reward, done."""
        if self.current_step >= len(self.df):
            logging.error(f"Error: current_step {self.current_step} >= df length {len(self.df)}")
            return self._get_state(), -100, True

        current_df_row = self.df.iloc[self.current_step]
        current_price = current_df_row['close']
        if pd.isna(current_price) or current_price <= 0:
            logging.warning(f"Warning: Invalid price {current_price} at step {self.current_step}")
            return self._get_state(), -0.1, False

        done = False
        reward = 0
        trade_amount_usd = 100
        btc_units = trade_amount_usd / current_price if current_price > EPSILON_NUMERIC else 0

        if action == 1:  # Buy
            if btc_units > 0:
                cost = btc_units * current_price * (1 + TRADING_FEE)
                if cost <= self.balance:
                    self.btc_held += btc_units
                    self.balance -= cost
                    reward -= cost * 0.001
                else:
                    reward = -0.1
            else:
                reward = -0.1
        elif action == 2:  # Sell
            if btc_units > 0:
                actual_btc_to_sell = min(btc_units, self.btc_held)
                if actual_btc_to_sell > EPSILON_NUMERIC:
                    revenue = actual_btc_to_sell * current_price * (1 - TRADING_FEE)
                    self.btc_held -= actual_btc_to_sell
                    self.balance += revenue
                else:
                    reward = -0.1
            else:
                reward = -0.1

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        price_for_portfolio = self.df.iloc[min(self.current_step, len(self.df)-1)]['close']
        if pd.isna(price_for_portfolio) or price_for_portfolio <= 0:
            price_for_portfolio = current_price

        portfolio_value_now = self.balance + self.btc_held * price_for_portfolio
        self.portfolio_values.append(portfolio_value_now)

        if len(self.portfolio_values) >= 2:
            prev_value = list(self.portfolio_values)[-2]
            if self.initial_balance > EPSILON_NUMERIC:
                portfolio_change_percent = (portfolio_value_now - prev_value) / self.initial_balance
                reward += np.clip(portfolio_change_percent * 10, -1, 1)

        next_state = self._get_state()
        return next_state, reward, done

class SACActor(nn.Module):
    """Actor network for SAC, outputting action probabilities."""
    def __init__(self, state_size, action_size):
        super(SACActor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        logits = self.layers(state)
        return logits

class SACCritic(nn.Module):
    """Critic network for SAC, estimating Q-values."""
    def __init__(self, state_size, action_size):
        super(SACCritic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        return self.layers(state)

class ReplayBuffer:
    """Experience replay buffer for SAC with pre-allocated tensors."""
    def __init__(self, capacity, state_size, device_):
        self.capacity = capacity
        self.device = device_

        # Pre-allocate memory on the specified device
        self.states = torch.empty((capacity, state_size), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.empty((capacity, state_size), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        
        self.size = 0
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        # state, next_state are numpy arrays; action is int; reward is float; done is bool
        self.states[self.pos].copy_(torch.from_numpy(state).float())
        self.actions[self.pos, 0] = action
        self.rewards[self.pos, 0] = reward
        self.next_states[self.pos].copy_(torch.from_numpy(next_state).float())
        self.dones[self.pos, 0] = float(done)
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        if self.size < batch_size:
            indices = np.random.choice(self.size, self.size, replace=False)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices].squeeze(1),
            self.rewards[indices].squeeze(1),
            self.next_states[indices],
            self.dones[indices].squeeze(1)
        )

    def __len__(self):
        return self.size

def soft_update(target, source, tau):
    """Soft update for target networks."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def optimize_model(actor, critic1, critic2, target_critic1, target_critic2, log_alpha, replay_buffer, actor_optimizer, critic1_optimizer, critic2_optimizer, alpha_optimizer):
    """Optimize SAC networks."""
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    
    alpha = log_alpha.exp().detach()

    # Critic loss
    with torch.no_grad():
        next_logits = actor(next_states)
        next_probs = F.softmax(next_logits, dim=-1)
        next_q1 = target_critic1(next_states)
        next_q2 = target_critic2(next_states)
        next_q = torch.min(next_q1, next_q2)
        next_value = (next_probs * (next_q - alpha * torch.log(next_probs + EPSILON_NUMERIC))).sum(dim=-1)
        target_q = rewards + GAMMA * (1 - dones) * next_value

    q1 = critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    q2 = critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    critic1_loss = F.mse_loss(q1, target_q)
    critic2_loss = F.mse_loss(q2, target_q)

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic2_optimizer.step()

    # Actor loss
    logits = actor(states)
    probs = F.softmax(logits, dim=-1)
    q1 = critic1(states).detach()
    q2 = critic2(states).detach()
    q = torch.min(q1, q2)
    actor_loss = (probs * (alpha * torch.log(probs + EPSILON_NUMERIC) - q)).sum(dim=-1).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()

    # Temperature loss
    alpha_loss = -(log_alpha * (torch.log(probs + EPSILON_NUMERIC).detach() + TARGET_ENTROPY)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # Soft update target critics
    soft_update(target_critic1, critic1, TAU)
    soft_update(target_critic2, critic2, TAU)

def load_crypto_data_from_csv(csv_filepath="data/BTCUSD_Daily_1_1_2020__5_21_2025.csv"):
    """Load and preprocess cryptocurrency data."""
    global STATE_SIZE, FEATURES_PER_STEP
    logging.info(f"--- Loading Data from CSV: {csv_filepath} ---")
    try:
        df = pd.read_csv(csv_filepath, thousands=',')
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_filepath}")
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return None, None, None, None, None

    column_map = {
        "Date": "date_str",
        "Price": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume_str"
    }
    rename_cols = {k: v for k, v in column_map.items() if k in df.columns}
    df = df[list(rename_cols.keys())].rename(columns=rename_cols)

    try:
        df['timestamp'] = pd.to_datetime(df['date_str'], format='%m/%d/%Y').astype('int64') // 10**6
    except Exception as e:
        logging.error(f"Error converting 'date_str': {e}")
        return None, None, None, None, None

    def convert_volume(vol_str):
        if isinstance(vol_str, (int, float)):
            return float(vol_str)
        vol_str = str(vol_str).upper().replace(',', '')
        try:
            if 'K' in vol_str:
                return float(vol_str.replace('K', '')) * 1000
            elif 'M' in vol_str:
                return float(vol_str.replace('M', '')) * 1000000
            elif 'B' in vol_str:
                return float(vol_str.replace('B', '')) * 1000000000
            return float(vol_str)
        except ValueError:
            return None

    df['volume'] = df['volume_str'].apply(convert_volume)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['close'] > 0]
    df = df[df['volume'] < df['volume'].quantile(0.99)]
    df.dropna(subset=['timestamp'] + required_cols, inplace=True)
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        logging.error("DataFrame empty after cleaning.")
        return None, None, None, None, None

    df_for_ta = df[required_cols].copy()
    try:
        df_with_ta = add_all_ta_features(
            df_for_ta,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=True
        )
        df_with_ta['trend_sma_slow'] = df_with_ta['close'].rolling(window=26).mean().bfill()
        df_with_ta['volatility_atr'] = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True
        ).average_true_range()
        df_with_ta['volatility_atr'] = df_with_ta['volatility_atr'].replace(0, EPSILON_NUMERIC)
    except Exception as e:
        logging.error(f"Error adding TA features: {e}")
        return None, None, None, None, None

    features_to_select = ['close', 'trend_sma_fast', 'momentum_rsi', 'trend_macd', 'volume_obv', 'trend_sma_slow', 'volatility_atr']
    missing_features = [f for f in features_to_select if f not in df_with_ta.columns]
    if missing_features:
        logging.error(f"Error: Missing features {missing_features}")
        return None, None, None, None, None

    df_with_ta['timestamp'] = df['timestamp']
    df_final = df_with_ta.dropna(subset=features_to_select).copy()

    if df_final.empty:
        logging.error("DataFrame empty after feature selection.")
        return None, None, None, None, None

    # Add input validation for data length
    MIN_REQUIRED_DATA_POINTS = WINDOW_SIZE + 60 # e.g., window + ~2 months for train/val/test
    if len(df_final) < MIN_REQUIRED_DATA_POINTS:
        logging.error(f"Error: Insufficient data: Found {len(df_final)} rows, require at least {MIN_REQUIRED_DATA_POINTS}.")
        return None, None, None, None, None

    logging.info("Feature statistics:")
    logging.info(f"\n{df_final[features_to_select].describe()}")

    train_size = int(0.8 * len(df_final))
    val_size = int(0.1 * len(df_final))
    # Ensure val_size doesn't make test_df empty or too small
    if train_size + val_size >= len(df_final) - WINDOW_SIZE: # Ensure test_df has at least WINDOW_SIZE
        val_size = max(0, len(df_final) - train_size - WINDOW_SIZE)
    
    test_size = len(df_final) - train_size - val_size
    if test_size < WINDOW_SIZE: # Check if test set is too small
        logging.error(f"Error: Test set too small ({test_size} data points) after splitting. Need at least {WINDOW_SIZE}.")
        return None, None, None, None, None

    train_df = df_final.iloc[:train_size]
    val_df = df_final.iloc[train_size:train_size + val_size]
    test_df = df_final.iloc[train_size + val_size:]

    if train_df.empty or val_df.empty or test_df.empty: # val_df can be empty if val_size is 0
        logging.error(f"Error: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)} - one or more are empty post-split.")
        return None, None, None, None, None
    
    logging.info(f"Data split: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

    FEATURES_PER_STEP = len(features_to_select)
    STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + 2 # Update global STATE_SIZE
    logging.info(f"FEATURES_PER_STEP: {FEATURES_PER_STEP}, STATE_SIZE: {STATE_SIZE}")

    full_feature_tensor = torch.FloatTensor(df_final[features_to_select].values)
    
    # Calculate mean and std from the training part of the full_feature_tensor
    train_features_for_norm = full_feature_tensor[:train_size]
    data_mean = train_features_for_norm.mean(dim=0, keepdim=True) # keepdim for broadcasting
    data_std = train_features_for_norm.std(dim=0, keepdim=True)
    data_std[data_std < EPSILON_NUMERIC] = EPSILON_NUMERIC # Avoid division by zero if a feature is constant

    # Pass sliced feature tensors and global mean/std to environments
    # Note: feature tensors sent to Env are already sliced and will be moved to device by Env
    train_env = CryptoTradingEnv(train_df, full_feature_tensor[:train_size], data_mean, data_std)
    val_env = CryptoTradingEnv(val_df, full_feature_tensor[train_size:train_size + val_size], data_mean, data_std)
    test_env = CryptoTradingEnv(test_df, full_feature_tensor[train_size + val_size:], data_mean, data_std)
    
    return train_env, val_env, test_env, data_mean, data_std # Return mean/std if needed elsewhere, though Env handles it

def plot_metrics(metrics):
    """Plot training metrics."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(metrics["episode_rewards"])
    plt.title("Episode Rewards")
    plt.subplot(1, 3, 2)
    plt.plot(metrics["portfolio_values"])
    plt.title("Training Portfolio Values")
    plt.subplot(1, 3, 3)
    plt.plot(metrics["val_portfolio_values"])
    plt.title("Validation Portfolio Values")
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

def main():
    """Train SAC for cryptocurrency trading."""
    train_env, val_env, test_env, _, _ = load_crypto_data_from_csv()
    if train_env is None or val_env is None or test_env is None:
        logging.error("Failed to load data. Exiting.")
        return

    logging.info(f"Training steps: {train_env.max_steps}")
    logging.info(f"Validation steps: {val_env.max_steps}")

    # Initialize SAC networks
    actor = SACActor(STATE_SIZE, ACTION_SIZE).to(device)
    critic1 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    critic2 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    target_critic1 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    target_critic2 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    target_critic1.eval()
    target_critic2.eval()

    # Learnable temperature
    log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True, device=device, dtype=torch.float32)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=LEARNING_RATE)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=LEARNING_RATE)
    alpha_optimizer = optim.Adam([log_alpha], lr=LEARNING_RATE)

    # Pass STATE_SIZE and device to ReplayBuffer constructor
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, STATE_SIZE, device)
    training_metrics = {
        "episode_rewards": [], "portfolio_values": [], "val_portfolio_values": [],
        "alpha_values": [], "total_steps": 0
    }
    best_val_portfolio = 10000
    patience_counter = 0

    logging.info("--- Training SAC for Crypto Trading ---")

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    # MPS profiling is not directly supported via ProfilerActivity.MPS
    # We will just use CPU profiling if on MPS for now.
    # If specific MPS profiling tools/methods are needed, they'd be different.

    with profile(activities=activities, record_shapes=True) as prof:
        for episode in range(1, EPISODES + 1):
            state = train_env.reset()
            episode_reward = 0
            done = False

            with record_function("episode_loop"):
                while not done:
                    with torch.no_grad():
                        actor.eval()
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        logits = actor(state_tensor)
                        probs = F.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample().item()
                        actor.train()

                    next_state, reward, done = train_env.step(action)
                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                    training_metrics["total_steps"] += 1

                    optimize_model(
                        actor, critic1, critic2, target_critic1, target_critic2,
                        log_alpha, replay_buffer, actor_optimizer,
                        critic1_optimizer, critic2_optimizer, alpha_optimizer
                    )

            training_metrics["episode_rewards"].append(episode_reward)
            training_metrics["portfolio_values"].append(
                train_env.portfolio_values[-1] if train_env.portfolio_values else train_env.initial_balance
            )
            training_metrics["alpha_values"].append(log_alpha.exp().item())

            # Log training progress every episode
            logging.info(f"Episode {episode}/{EPISODES}, Reward: {episode_reward:.2f}, "
                         f"Portfolio: ${training_metrics['portfolio_values'][-1]:.2f}, Alpha: {log_alpha.exp().item():.3f}")
            logging.info(f"Train Balance: ${train_env.balance:.2f}, BTC Held: {train_env.btc_held:.6f}")

            if episode % VALIDATION_INTERVAL == 0:
                val_state = val_env.reset()
                val_done = False
                val_actions = []
                val_step_count = 0
                while not val_done:
                    with torch.no_grad():
                        actor.eval()
                        val_state_tensor = torch.FloatTensor(val_state).unsqueeze(0).to(device)
                        logits = actor(val_state_tensor)
                        probs = F.softmax(logits, dim=-1)
                        action = probs.argmax().item()
                        if val_step_count < 5:
                            logging.info(f"Val Ep {episode} Step {val_step_count} Action Probs: {probs.tolist()}")
                        actor.train()
                    val_state, _, val_done = val_env.step(action)
                    val_actions.append(action)
                    val_step_count += 1
                    if val_step_count > val_env.max_steps + 5:
                        logging.warning("Warning: Validation loop exceeded max_steps.")
                        break

                val_portfolio = val_env.portfolio_values[-1] if val_env.portfolio_values else val_env.initial_balance
                training_metrics["val_portfolio_values"].append(val_portfolio)

                if val_portfolio > best_val_portfolio:
                    best_val_portfolio = val_portfolio
                    patience_counter = 0
                    torch.save({
                        'actor_state_dict': actor.state_dict(),
                        'critic1_state_dict': critic1.state_dict(),
                        'critic2_state_dict': critic2.state_dict(),
                    }, "best_sac.pth")
                    logging.info(f"Saved best model to 'best_sac.pth' with validation portfolio ${val_portfolio:.2f}")
                else:
                    patience_counter += 1

                # Log validation progress at validation interval
                logging.info(f"Validation Portfolio: ${val_portfolio:.2f}")
                logging.info(f"Val Balance: ${val_env.balance:.2f}, BTC Held: {val_env.btc_held:.6f}")

                if patience_counter >= PATIENCE:
                    logging.info("Early stopping triggered.")
                    break

            if episode % 50 == 0:
                with open("training_metrics.json", "w") as f:
                    json.dump(training_metrics, f, indent=4)

    logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Test phase
    test_state = test_env.reset()
    test_portfolio_values = []
    test_done = False
    test_actions = []
    while not test_done:
        with torch.no_grad():
            actor.eval()
            test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(device)
            logits = actor(test_state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = probs.argmax().item()
            actor.train()
        test_state, _, test_done = test_env.step(action)
        test_portfolio_values.append(test_env.portfolio_values[-1])
        test_actions.append(action)

    final_test_portfolio = test_portfolio_values[-1]
    logging.info(f"Test Portfolio Final Value: ${final_test_portfolio:.2f}")
    logging.info(f"Test Actions (first 20): {test_actions[:20]}... (Total: {len(test_actions)})")

    # Save final model weights
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
    }, "final_sac.pth")
    logging.info("Saved final model weights to 'final_sac.pth'")
    # Save comprehensive checkpoint
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
        'target_critic1_state_dict': target_critic1.state_dict(),
        'target_critic2_state_dict': target_critic2.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict': critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict': critic2_optimizer.state_dict(),
        'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
        'log_alpha': log_alpha,
        'training_metrics': training_metrics,
        'final_test_portfolio': final_test_portfolio
    }
    torch.save(checkpoint, "final_checkpoint.pth")

    # Save and plot metrics
    with open("training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=4)
    plot_metrics(training_metrics)

    logging.info("--- Training Complete ---")

if __name__ == "__main__":
    main()