import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import json
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange

# --- Device Configuration ---
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("--- Using device: MPS (Apple Silicon GPU) ---")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("--- Using device: CUDA ---")
else:
    device = torch.device("cpu")
    print("--- Using device: CPU ---")

# --- Hyperparameters ---
WINDOW_SIZE = 50
FEATURES_PER_STEP = 7  # Updated in load_crypto_data_from_csv
STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + 2  # Updated later
ACTION_SIZE = 3  # Buy, Sell, Hold
BATCH_SIZE = 256
EPISODES = 200
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
LEARNING_RATE = 0.0003
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 20
TRADING_FEE = 0.0035  # 0.2% fee
VALIDATION_INTERVAL = 10
PATIENCE = 50
PORTFOLIO_HISTORY_SIZE = 365  # 365 trading days
EPSILON_NUMERIC = 1e-9

class CryptoTradingEnv:
    """Environment for cryptocurrency trading using daily OHLCV data."""
    def __init__(self, df):
        """Initialize with a DataFrame containing OHLCV and technical indicators."""
        self.df = df.reset_index(drop=True)
        self.current_step = WINDOW_SIZE
        self.initial_balance = 10000  # $10,000 USD
        self.balance = self.initial_balance
        self.btc_held = 0
        self.max_steps = len(df) - WINDOW_SIZE - 1
        self.portfolio_values = deque(maxlen=PORTFOLIO_HISTORY_SIZE)

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

        window = self.df.iloc[start_idx:end_idx]
        feature_cols = ['close', 'trend_sma_fast', 'momentum_rsi', 'trend_macd', 'volume_obv', 'trend_sma_slow', 'volatility_atr']
        actual_feature_cols = [col for col in feature_cols if col in window.columns]
        if len(actual_feature_cols) != FEATURES_PER_STEP:
            raise ValueError(f"Expected {FEATURES_PER_STEP} features, found {len(actual_feature_cols)}: {actual_feature_cols}")

        features = window[actual_feature_cols].values.astype(np.float32)
        features_mean = features.mean(axis=0)
        features_std = features.std(axis=0) + EPSILON_NUMERIC
        features = (features - features_mean) / features_std

        state = np.concatenate([features.flatten(), [self.balance / self.initial_balance, self.btc_held]])
        return state

    def step(self, action):
        """Execute action (0: Hold, 1: Buy, 2: Sell) and return next state, reward, done."""
        if self.current_step >= len(self.df):
            print(f"Error: current_step {self.current_step} >= df length {len(self.df)}")
            return self._get_state(), -100, True

        current_df_row = self.df.iloc[self.current_step]
        current_price = current_df_row['close']
        if pd.isna(current_price) or current_price <= 0:
            print(f"Warning: Invalid price {current_price} at step {self.current_step}")
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

        # Reward: clipped portfolio value change
        if len(self.portfolio_values) >= 2:
            prev_value = list(self.portfolio_values)[-2]
            if self.initial_balance > EPSILON_NUMERIC:
                portfolio_change_percent = (portfolio_value_now - prev_value) / self.initial_balance
                reward += np.clip(portfolio_change_percent * 10, -1, 1)

        next_state = self._get_state()
        return next_state, reward, done

class DQN(nn.Module):
    """Deep Q-Network for action-value estimation."""
    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store transition as tensors."""
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def optimize_model(policy_net, target_net, replay_buffer, optimizer):
    """Optimize policy network using Double DQN."""
    if len(replay_buffer) < BATCH_SIZE:
        return

    transitions = replay_buffer.sample(BATCH_SIZE)
    if not transitions:
        return

    batch = list(zip(*transitions))
    states = torch.stack(batch[0])
    actions = torch.LongTensor(batch[1]).to(device)
    rewards = torch.FloatTensor(batch[2]).to(device)
    next_states = torch.stack(batch[3])
    dones = torch.FloatTensor(batch[4]).to(device)

    policy_net.train()
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_actions = policy_net(next_states).argmax(1).unsqueeze(1)
    next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1).detach()
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = nn.MSELoss()(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

def load_crypto_data_from_csv(csv_filepath="data/BTCUSD_Daily_1_1_2020__5_21_2025.csv"):
    """Load and preprocess cryptocurrency data."""
    global STATE_SIZE, FEATURES_PER_STEP
    print(f"--- Loading Data from CSV: {csv_filepath} ---")
    try:
        df = pd.read_csv(csv_filepath, thousands=',')
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return None, None, None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None, None

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
        print(f"Error converting 'date_str': {e}")
        return None, None, None

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
        print("DataFrame empty after cleaning.")
        return None, None, None

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
    except Exception as e:
        print(f"Error adding TA features: {e}")
        return None, None, None

    features_to_select = ['close', 'trend_sma_fast', 'momentum_rsi', 'trend_macd', 'volume_obv', 'trend_sma_slow', 'volatility_atr']
    missing_features = [f for f in features_to_select if f not in df_with_ta.columns]
    if missing_features:
        print(f"Error: Missing features {missing_features}")
        return None, None, None

    df_with_ta['timestamp'] = df['timestamp']
    df_final = df_with_ta.dropna(subset=features_to_select).copy()

    if df_final.empty:
        print("DataFrame empty after feature selection.")
        return None, None, None

    print("Feature statistics:")
    print(df_final[features_to_select].describe())

    train_size = int(0.8 * len(df_final))
    val_size = int(0.1 * len(df_final))
    if train_size + val_size >= len(df_final):
        val_size = max(0, len(df_final) - train_size - 1)

    train_df = df_final.iloc[:train_size]
    val_df = df_final.iloc[train_size:train_size + val_size]
    test_df = df_final.iloc[train_size + val_size:]

    if train_df.empty or test_df.empty:
        print(f"Error: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
        return None, None, None

    print(f"Data split: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

    FEATURES_PER_STEP = len(features_to_select)
    STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + 2
    print(f"FEATURES_PER_STEP: {FEATURES_PER_STEP}, STATE_SIZE: {STATE_SIZE}")

    train_env = CryptoTradingEnv(train_df)
    val_env = CryptoTradingEnv(val_df)
    return train_env, val_env, test_df

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
    """Train DQN for cryptocurrency trading."""
    train_env, val_env, test_df = load_crypto_data_from_csv()
    if train_env is None or val_env is None or test_df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Training steps: {train_env.max_steps}")
    print(f"Validation steps: {val_env.max_steps}")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    training_metrics = {
        "episode_rewards": [], "portfolio_values": [], "val_portfolio_values": [],
        "epsilon_values": [], "total_steps": 0
    }

    best_val_portfolio = 10000
    patience_counter = 0

    print("--- Training DQN for Crypto Trading ---")
    for episode in range(1, EPISODES + 1):
        state = train_env.reset()
        episode_reward = 0
        done = False

        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.cos(0.5 * np.pi * episode / (EPISODES * 1.5))
        epsilon = max(EPSILON_END, epsilon)

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE - 1)
            else:
                with torch.no_grad():
                    policy_net.eval()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
                    policy_net.train()

            next_state, reward, done = train_env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            training_metrics["total_steps"] += 1

            optimize_model(policy_net, target_net, replay_buffer, optimizer)

        training_metrics["episode_rewards"].append(episode_reward)
        training_metrics["portfolio_values"].append(
            train_env.portfolio_values[-1] if train_env.portfolio_values else train_env.initial_balance
        )
        training_metrics["epsilon_values"].append(epsilon)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % VALIDATION_INTERVAL == 0:
            val_state = val_env.reset()
            val_done = False
            val_actions = []
            val_step_count = 0
            while not val_done:
                with torch.no_grad():
                    policy_net.eval()
                    val_state_tensor = torch.FloatTensor(val_state).unsqueeze(0).to(device)
                    q_values = policy_net(val_state_tensor)
                    action = q_values.argmax().item()
                    if val_step_count < 5:
                        print(f"Val Ep {episode} Step {val_step_count} Q-Values: {q_values.tolist()}")
                    policy_net.train()
                val_state, _, val_done = val_env.step(action)
                val_actions.append(action)
                val_step_count += 1
                if val_step_count > val_env.max_steps + 5:
                    print("Warning: Validation loop exceeded max_steps.")
                    break

            val_portfolio = val_env.portfolio_values[-1] if val_env.portfolio_values else val_env.initial_balance
            training_metrics["val_portfolio_values"].append(val_portfolio)

            if val_portfolio > best_val_portfolio:
                best_val_portfolio = val_portfolio
                patience_counter = 0
                torch.save(policy_net.state_dict(), "best_policy_net.pth")
                print(f"Saved best model to 'best_policy_net.pth' with validation portfolio ${val_portfolio:.2f}")
            else:
                patience_counter += 1

            print(f"Episode {episode}/{EPISODES}, Reward: {episode_reward:.2f}, "
                  f"Portfolio: ${training_metrics['portfolio_values'][-1]:.2f}, Epsilon: {epsilon:.3f}")
            print(f"Validation Portfolio: ${val_portfolio:.2f}")
            print(f"Train Balance: ${train_env.balance:.2f}, BTC Held: {train_env.btc_held:.6f}")
            print(f"Val Balance: ${val_env.balance:.2f}, BTC Held: {val_env.btc_held:.6f}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        if episode % 50 == 0:
            with open("training_metrics.json", "w") as f:
                json.dump(training_metrics, f, indent=4)

    # Test phase
    test_env = CryptoTradingEnv(test_df)
    test_state = test_env.reset()
    test_portfolio_values = []
    test_done = False
    test_actions = []
    while not test_done:
        with torch.no_grad():
            policy_net.eval()
            test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(device)
            q_values = policy_net(test_state_tensor)
            action = q_values.argmax().item()
            policy_net.train()
        test_state, _, test_done = test_env.step(action)
        test_portfolio_values.append(test_env.portfolio_values[-1])
        test_actions.append(action)

    print(f"Test Portfolio Final Value: ${test_portfolio_values[-1]:.2f}")
    print(f"Test Actions (first 20): {test_actions[:20]}... (Total: {len(test_actions)})")

    # Save final model weights
    torch.save(policy_net.state_dict(), "final_policy_net.pth")
    print("Saved final model weights to 'final_policy_net.pth'")

    # Save comprehensive checkpoint
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_metrics': training_metrics
    }
    torch.save(checkpoint, "final_checkpoint.pth")
    print("Saved checkpoint to 'final_checkpoint.pth'")

    # Save and plot metrics
    with open("training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=4)
    plot_metrics(training_metrics)

    print("--- Training Complete ---")

if __name__ == "__main__":
    main()