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
import logging

# --- Logging Configuration ---
LOG_FILE_PATH = "training_run.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    filename=LOG_FILE_PATH,
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# --- Device Configuration ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("--- Using CUDA ---")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("--- Using Metal ---")
else:
    device = torch.device("cpu")
    logging.info("--- Using CPU ---")

# --- Hyperparameters ---
WINDOW_SIZE = 20
FEATURES_PER_STEP = 7
STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + WINDOW_SIZE + 3
ACTION_SIZE = 3
BATCH_SIZE = 256
EPISODES = 50
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 50000
TRADING_FEE = 0.0035
VALIDATION_INTERVAL = 1
PATIENCE = 100
PORTFOLIO_HISTORY_SIZE = 365
EPSILON_NUMERIC = 1e-9
TAU = 0.005
ALPHA = 0.2
TARGET_ENTROPY = -np.log(1.0 / ACTION_SIZE) * 1.2
EPSILON_START = 0.5
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1

def categorize_market_conditions(df, price_col='close', atr_col='volatility_atr', macd_col='trend_macd', window=30, atr_threshold=0.05, bull_return_threshold=0.2, bear_return_threshold=-0.2):
    """
    Categorizes market conditions based on price trends, volatility, and technical indicators.
    """
    df_categorized = df.copy()

    # Ensure required columns are present
    required_ta_cols = [price_col, atr_col, macd_col]
    for col in required_ta_cols:
        if col not in df_categorized.columns:
            logging.error(f"Missing required column for categorization: {col}")
            # Return with an empty 'condition' column or raise error
            df_categorized['condition'] = 'unknown'
            return df_categorized

    # Compute rolling returns using future data for labeling purposes
    # df_categorized['future_return'] = (df_categorized[price_col].shift(-window) - df_categorized[price_col]) / df_categorized[price_col]
    # To avoid issues with pandas versions and SettingWithCopyWarning, calculate returns carefully
    future_prices = df_categorized[price_col].shift(-window)
    current_prices = df_categorized[price_col]
    # Calculate return, avoid division by zero or by NaN
    df_categorized['future_return'] = np.where(
        (current_prices.notna() & (current_prices != 0) & future_prices.notna()),
        (future_prices - current_prices) / current_prices,
        np.nan
    )


    # Compute ATR relative to price
    # df_categorized['atr_ratio'] = df_categorized[atr_col] / df_categorized[price_col]
    df_categorized['atr_ratio'] = np.where(
        (df_categorized[price_col].notna() & (df_categorized[price_col] != 0) & df_categorized[atr_col].notna()),
        df_categorized[atr_col] / df_categorized[price_col],
        np.nan
    )


    conditions = []
    for i in range(len(df_categorized)):
        ret = df_categorized['future_return'].iloc[i]
        atr_r = df_categorized['atr_ratio'].iloc[i]
        macd = df_categorized[macd_col].iloc[i]

        if pd.isna(ret) or pd.isna(atr_r) or pd.isna(macd):
            conditions.append('unknown')
        elif atr_r > atr_threshold:  # High volatility
            conditions.append('high_volatility')
        elif ret > bull_return_threshold and macd > 0:  # Bull market
            conditions.append('bull')
        elif ret < bear_return_threshold and macd < 0:  # Bear market
            conditions.append('bear')
        else:  # Sideways
            conditions.append('sideways')
    df_categorized['condition'] = conditions
    
    # Log distribution
    condition_counts = df_categorized['condition'].value_counts()
    logging.info(f"Market conditions categorized (before potential merge): {condition_counts.to_dict()}")

    # Merge small 'high_volatility' subset
    # Define a minimum practical size, e.g., slightly more than WINDOW_SIZE for meaningful environment
    min_subset_size_for_hv = WINDOW_SIZE + 10 # Example threshold
    if 'high_volatility' in condition_counts and condition_counts['high_volatility'] < min_subset_size_for_hv:
        logging.info(f"High volatility subset is too small ({condition_counts['high_volatility']} rows). Merging into 'bear' market.")
        df_categorized.loc[df_categorized['condition'] == 'high_volatility', 'condition'] = 'bear' # Or 'sideways'
        condition_counts = df_categorized['condition'].value_counts() # Update counts
        logging.info(f"Market conditions after merging 'high_volatility': {condition_counts.to_dict()}")

    return df_categorized

class CryptoTradingEnv:
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
        self.avg_purchase_price = 0.0
        self.consecutive_holds = 0
        self.balance_history = deque(maxlen=WINDOW_SIZE)
        self.balance_history.append(self.initial_balance)
        logging.info(f"Env init: df_length={len(self.df)}, feature_tensor_shape={self.feature_tensor.shape}, max_steps={self.max_steps}")
        if not self.df.empty:
            logging.info(f"Price range in passed df: {self.df['close'].min():.2f} - {self.df['close'].max():.2f}")

    def reset(self):
        self.current_step = WINDOW_SIZE
        self.balance = self.initial_balance
        self.btc_held = 0
        self.portfolio_values.clear()
        self.portfolio_values.append(self.initial_balance)
        self.avg_purchase_price = 0.0
        self.consecutive_holds = 0
        self.balance_history.clear()
        self.balance_history.append(self.initial_balance)
        return self._get_state()

    def _get_state(self):
        end_idx = self.current_step
        start_idx = end_idx - WINDOW_SIZE
        features_window = self.feature_tensor[start_idx:end_idx]
        if features_window.shape[0] != WINDOW_SIZE:
            logging.critical(f"CRITICAL: features_window length {features_window.shape[0]} != WINDOW_SIZE {WINDOW_SIZE}.")
        if self.global_mean is not None and self.global_std is not None:
            normalized_features = (features_window - self.global_mean) / self.global_std
        else:
            features_mean = features_window.mean(dim=0)
            features_std = features_window.std(dim=0) + EPSILON_NUMERIC
            normalized_features = (features_window - features_mean) / features_std
        normalized_consecutive_holds = min(self.consecutive_holds, 10) / 10.0
        normalized_balance = self.balance / self.initial_balance
        balance_features = torch.tensor([normalized_balance] * WINDOW_SIZE, device=device, dtype=torch.float32)
        state_portfolio_info = torch.tensor([
            normalized_balance,
            self.btc_held,
            normalized_consecutive_holds
        ], device=device, dtype=torch.float32)
        state = torch.cat([normalized_features.flatten(), balance_features, state_portfolio_info])
        return state.cpu().numpy()

    def get_valid_actions(self):
        current_df_row = self.df.iloc[self.current_step]
        current_price = current_df_row['close']
        valid_actions = [0]  # Hold is always valid

        # Check Buy: balance must cover cost
        trade_amount_usd = min(100.0, self.balance * 0.1)
        btc_units_to_buy = trade_amount_usd / current_price if current_price > EPSILON_NUMERIC else 0
        cost = btc_units_to_buy * current_price * (1 + TRADING_FEE)
        if btc_units_to_buy > EPSILON_NUMERIC and cost <= self.balance:
            valid_actions.append(1)

        # Check Sell: must have enough btc_held
        btc_available = max(0, self.btc_held - EPSILON_NUMERIC)
        trade_amount_usd_sell = min(trade_amount_usd, btc_available * current_price * (1 - TRADING_FEE))
        btc_units_target_sell = trade_amount_usd_sell / current_price if current_price > EPSILON_NUMERIC else 0
        if btc_units_target_sell > EPSILON_NUMERIC and self.btc_held >= EPSILON_NUMERIC:
            valid_actions.append(2)

        return valid_actions


    def step(self, action):
        action_step_log = self.current_step
        logging.debug(f"Step ENTRY: current_step={action_step_log}, action={action}, balance={self.balance:.2f}, btc_held={self.btc_held:.6f}, avg_cost={self.avg_purchase_price:.2f}, consec_holds={self.consecutive_holds}")

        current_df_row = self.df.iloc[self.current_step]
        current_price = current_df_row['close']
        # Get MACD for holding incentive. Ensure 'trend_macd' is in self.df columns.
        current_macd = self.df['trend_macd'].iloc[self.current_step] if 'trend_macd' in self.df.columns and self.current_step < len(self.df) else 0


        if pd.isna(current_price) or current_price <= 0:
            logging.warning(f"Invalid price {current_price} at step {action_step_log}. Taking no action, small penalty.")
            self.current_step += 1
            done = self.current_step >= self.max_steps
            return self._get_state(), -0.1, done

        trade_reward = 0
        reward_scale = 5.0 # Original reward_scale for trade profitability
        portfolio_scale = 2.0 # Reduce portfolio_scale
        valid_trade_reward = 0.1 # Base reward for making a valid trade decision
        trade_amount_usd = min(100.0, self.balance * 0.1)
        trade_amount_usd_sell = trade_amount_usd
        action_executed = False


        if action == 0: # Hold
            self.consecutive_holds += 1
            trade_reward = 0.0 
            if current_macd > 0: # Small incentive for holding in uptrend (based on current step's MACD)
                trade_reward += 0.05
            action_executed = True
        elif action == 1: # Buy
            btc_units_to_buy = trade_amount_usd / current_price
            cost_of_buy_with_fee = btc_units_to_buy * current_price * (1 + TRADING_FEE)
            
            current_total_value_of_held_btc = self.avg_purchase_price * self.btc_held
            cost_of_new_btc_units = btc_units_to_buy * current_price
            
            self.balance -= cost_of_buy_with_fee
            self.btc_held += btc_units_to_buy
            self.avg_purchase_price = (current_total_value_of_held_btc + cost_of_new_btc_units) / self.btc_held if self.btc_held > EPSILON_NUMERIC else 0.0
            
            # Smoothed trade reward: 5-day future price
            future_price_for_reward = current_price # Default to current price if future is unavailable
            if self.current_step + 5 < len(self.df):
                future_price_candidate = self.df.iloc[self.current_step + 5]['close']
                if not (pd.isna(future_price_candidate) or future_price_candidate <= 0):
                    future_price_for_reward = future_price_candidate
            
            potential_revenue_from_trade_smoothed = btc_units_to_buy * future_price_for_reward * (1 - TRADING_FEE)
            unrealized_profit_for_this_trade = potential_revenue_from_trade_smoothed - cost_of_buy_with_fee
            
            trade_reward = valid_trade_reward
            # Normalize by initial_balance
            trade_reward_component = (unrealized_profit_for_this_trade / self.initial_balance) * reward_scale
            trade_reward += trade_reward_component
            
            action_executed = True
            self.consecutive_holds = 0
        elif action == 2: # Sell
            btc_available = max(0, self.btc_held - EPSILON_NUMERIC)
            # Adjust trade_amount_usd_sell to be based on available BTC to sell, up to the standard trade_amount_usd
            max_sellable_value_before_fee = btc_available * current_price
            trade_amount_usd_sell = min(trade_amount_usd, max_sellable_value_before_fee * (1-TRADING_FEE) ) # Sell up to $100 or 10% of balance equivalent in BTC value after fee

            btc_units_target_sell = trade_amount_usd_sell / current_price if current_price > EPSILON_NUMERIC else 0
            actual_btc_to_sell = btc_units_target_sell # Assuming this amount is available based on prior checks or logic in get_valid_actions

            revenue_from_sell_after_fee = actual_btc_to_sell * current_price * (1 - TRADING_FEE)
            cost_of_sold_btc = actual_btc_to_sell * self.avg_purchase_price 
            realized_profit_for_trade = revenue_from_sell_after_fee - cost_of_sold_btc
            
            self.balance += revenue_from_sell_after_fee
            self.btc_held -= actual_btc_to_sell
            if self.btc_held < EPSILON_NUMERIC:
                self.btc_held = 0.0
                self.avg_purchase_price = 0.0
            
            trade_reward = valid_trade_reward
            # Normalize by initial_balance
            trade_reward_component = (realized_profit_for_trade / self.initial_balance) * reward_scale
            trade_reward += trade_reward_component

            action_executed = True
            self.consecutive_holds = 0

        done = action_step_log >= self.max_steps - 1 # current_step was already advanced in some logic paths, use action_step_log
        self.current_step += 1

        if self.current_step >= len(self.df): # Use new current_step for this check
            price_for_portfolio_eval_at_next_step = current_price
        else:
            price_for_portfolio_eval_at_next_step_candidate = self.df.iloc[self.current_step]['close']
            if pd.isna(price_for_portfolio_eval_at_next_step_candidate) or price_for_portfolio_eval_at_next_step_candidate <= 0:
                price_for_portfolio_eval_at_next_step = current_price # Fallback to previous price
            else:
                price_for_portfolio_eval_at_next_step = price_for_portfolio_eval_at_next_step_candidate

        portfolio_value_at_next_step = self.balance + self.btc_held * price_for_portfolio_eval_at_next_step
        self.portfolio_values.append(portfolio_value_at_next_step)
        self.balance_history.append(self.balance)

        overall_reward = trade_reward
        if len(self.portfolio_values) >= 2:
            portfolio_change_percent = (portfolio_value_at_next_step - list(self.portfolio_values)[-2]) / self.initial_balance
            overall_reward += np.clip(portfolio_change_percent * portfolio_scale, -0.5, 0.5)

        next_state = self._get_state()
        logging.debug(f"Step EXIT: took_action_at_step={action_step_log}, new_current_step_for_next_state={self.current_step}, reward={overall_reward:.4f}, done={done}, portfolio_value={portfolio_value_at_next_step:.2f}, action_executed={action_executed}")
        return next_state, overall_reward, done

class SACActor(nn.Module):
    def __init__(self, state_size, action_size):
        super(SACActor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, state):
        logits = self.layers(state)
        logits = torch.clamp(logits, -10, 10)
        return logits

class SACCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(SACCritic, self).__init__()
        # Critic takes state and action (one-hot encoded) as input
        self.layers = nn.Sequential(
            nn.Linear(state_size + action_size, 1024), # Input size changed
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output a single Q-value
        )

    def forward(self, state, action_one_hot): # Action is now one-hot encoded
        # Ensure action_one_hot is float, matching state type
        x = torch.cat([state, action_one_hot.float()], dim=-1)
        return self.layers(x)

class ReplayBuffer:
    def __init__(self, capacity, state_size, device_):
        self.capacity = capacity
        self.device = device_
        self.states = torch.empty((capacity, state_size), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, 1), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.empty((capacity, state_size), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.size = 0
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos].copy_(torch.from_numpy(state).float())
        self.actions[self.pos, 0] = action
        self.rewards[self.pos, 0] = reward
        self.next_states[self.pos].copy_(torch.from_numpy(next_state).float())
        self.dones[self.pos, 0] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
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
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def optimize_model(actor, critic1, critic2, target_critic1, target_critic2, log_alpha, replay_buffer, actor_optimizer, critic1_optimizer, critic2_optimizer, alpha_optimizer):
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    if torch.any(torch.isnan(states)) or torch.any(torch.isnan(rewards)):
        logging.warning("NaN detected in replay buffer sample. Skipping update.")
        return
    
    current_alpha = log_alpha.exp().detach() # Alpha value for Q-target and actor loss
    actions_one_hot = F.one_hot(actions, num_classes=ACTION_SIZE).float().to(device)


    with torch.no_grad():
        next_logits = actor(next_states)
        next_probs = F.softmax(next_logits, dim=-1)
        next_probs = torch.clamp(next_probs, 1e-10, 1.0) 
        next_log_probs_dist = torch.log(next_probs + EPSILON_NUMERIC)
        
        next_q_target_vals_list1 = []
        next_q_target_vals_list2 = []
        for i in range(ACTION_SIZE):
            action_i_one_hot = F.one_hot(torch.tensor([i]*BATCH_SIZE, device=device), num_classes=ACTION_SIZE).float()
            next_q_target_vals_list1.append(target_critic1(next_states, action_i_one_hot))
            next_q_target_vals_list2.append(target_critic2(next_states, action_i_one_hot))

        next_q1_all_actions = torch.cat(next_q_target_vals_list1, dim=1) 
        next_q2_all_actions = torch.cat(next_q_target_vals_list2, dim=1) 
        next_q_all_actions = torch.min(next_q1_all_actions, next_q2_all_actions)
        
        next_value = (next_probs * (next_q_all_actions - current_alpha * next_log_probs_dist)).sum(dim=-1)
        target_q = rewards + GAMMA * (1 - dones) * next_value

    q1 = critic1(states, actions_one_hot).squeeze(1) 
    q2 = critic2(states, actions_one_hot).squeeze(1) 
    
    critic1_loss = F.mse_loss(q1, target_q)
    critic2_loss = F.mse_loss(q2, target_q)

    if torch.any(torch.isnan(critic1_loss)) or torch.any(torch.isnan(critic2_loss)):
        logging.warning("NaN detected in critic loss. Skipping update.")
        return

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic2_optimizer.step()

    # Actor update
    # Freeze critic parameters for actor update to avoid issues if grads were cleared by optimizers
    for p in critic1.parameters(): p.requires_grad = False
    for p in critic2.parameters(): p.requires_grad = False

    logits = actor(states) 
    probs = F.softmax(logits, dim=-1) 
    probs = torch.clamp(probs, 1e-10, 1.0)
    log_probs_dist = torch.log(probs + EPSILON_NUMERIC) 

    q1_pi_detached_list = []
    q2_pi_detached_list = []
    for i in range(ACTION_SIZE):
        action_i_one_hot_actor = F.one_hot(torch.tensor([i]*BATCH_SIZE, device=device), num_classes=ACTION_SIZE).float()
        # Use .eval() and torch.no_grad() for critics if they are not intended to be part of this specific grad computation path
        # However, standard SAC uses Q values from current critics, but detached from their optimizer's perspective for actor loss.
        # The .detach() on Q values below is what matters for actor stability
        q1_pi_detached_list.append(critic1(states, action_i_one_hot_actor)) # Get Q without detaching yet
        q2_pi_detached_list.append(critic2(states, action_i_one_hot_actor))

    q1_pi_all_actions = torch.cat(q1_pi_detached_list, dim=1) 
    q2_pi_all_actions = torch.cat(q2_pi_detached_list, dim=1) 
    min_q_pi_all_actions_detached = torch.min(q1_pi_all_actions, q2_pi_all_actions).detach() # Detach here for actor loss

    actor_loss = (probs * (current_alpha * log_probs_dist - min_q_pi_all_actions_detached)).sum(dim=-1).mean()

    if torch.any(torch.isnan(actor_loss)):
        logging.warning("NaN detected in actor loss. Skipping update.")
        # Unfreeze critic parameters
        for p in critic1.parameters(): p.requires_grad = True
        for p in critic2.parameters(): p.requires_grad = True
        return

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()

    # Unfreeze critic parameters
    for p in critic1.parameters(): p.requires_grad = True
    for p in critic2.parameters(): p.requires_grad = True

    # Alpha (temperature) loss
    # Calculate current policy entropy using detached probabilities from the actor's output
    detached_probs_for_alpha = probs.detach() # Key: detach probs from actor graph
    detached_log_probs_for_alpha = torch.log(detached_probs_for_alpha + EPSILON_NUMERIC)
    current_policy_entropy_per_sample = -torch.sum(detached_probs_for_alpha * detached_log_probs_for_alpha, dim=1) # H(pi(.|s))
    
    # TARGET_ENTROPY is the desired H_target (positive value)
    # Loss: log_alpha * (current_entropy - target_entropy)
    # Grad dLoss/d(log_alpha) = (current_entropy - target_entropy)
    # If current_entropy < target_entropy, grad is negative, log_alpha increases (alpha increases) -> more exploration. Correct.
    alpha_loss = (log_alpha * (current_policy_entropy_per_sample - TARGET_ENTROPY)).mean()
    
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    soft_update(target_critic1, critic1, TAU)
    soft_update(target_critic2, critic2, TAU)

    logging.debug(f"Critic1 Loss: {critic1_loss.item():.4f}, Critic2 Loss: {critic2_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Alpha: {log_alpha.exp().item():.4f}, Mean Entropy: {current_policy_entropy_per_sample.mean().item():.4f}")

def load_crypto_data_from_csv(csv_filepath="data/BTCUSD_Daily_1_1_2020__5_21_2025.csv"):
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
    volume_upper_quantile = df['volume'].quantile(0.99)
    if pd.notna(volume_upper_quantile):
        df = df[df['volume'] < volume_upper_quantile]
    
    df.dropna(subset=['timestamp'] + required_cols, inplace=True)
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        logging.error("DataFrame empty after initial cleaning.")
        return None, None, None, None, None

    df_for_ta = df[required_cols].copy()
    try:
        df_with_ta = add_all_ta_features(
            df_for_ta, open="open", high="high", low="low", close="close", volume="volume", fillna=True
        )
        # Use df_with_ta for consistency as it contains the 'close', 'high', 'low' columns from df_for_ta
        df_with_ta['trend_sma_slow'] = df_with_ta['close'].rolling(window=26).mean().bfill()
        df_with_ta['volatility_atr'] = AverageTrueRange(
            high=df_with_ta['high'], low=df_with_ta['low'], close=df_with_ta['close'], window=14, fillna=True
        ).average_true_range()
        df_with_ta['volatility_atr'] = df_with_ta['volatility_atr'].replace(0, EPSILON_NUMERIC)
    except Exception as e:
        logging.error(f"Error adding TA features: {e}")
        return None, None, None, None, None

    features_to_select = ['close', 'trend_sma_fast', 'momentum_rsi', 'trend_macd', 'volume_obv', 'trend_sma_slow', 'volatility_atr']
    actual_features_present = [f for f in features_to_select if f in df_with_ta.columns]
    if len(actual_features_present) != len(features_to_select):
        missing_features = set(features_to_select) - set(actual_features_present)
        logging.error(f"Error: Missing TA features for model input: {missing_features}. Available: {df_with_ta.columns.tolist()}")
        return None, None, None, None, None

    df_with_ta['timestamp'] = df['timestamp'] # Add timestamp back from original df
    df_final_pre_categorization = df_with_ta.dropna(subset=features_to_select + ['timestamp']).copy()

    if df_final_pre_categorization.empty:
        logging.error("DataFrame empty after TA feature addition and NaN drop.")
        return None, None, None, None, None
    
    logging.info(f"DataFrame before categorization: {len(df_final_pre_categorization)} rows, Price range: {df_final_pre_categorization['close'].min():.2f} - {df_final_pre_categorization['close'].max():.2f}")

    # Step 1: Categorize market conditions
    # The categorize_market_conditions function expects 'volatility_atr' and 'trend_macd', which are in features_to_select
    # and thus in df_final_pre_categorization.
    df_categorized = categorize_market_conditions(df_final_pre_categorization, window=30) # Using default window of 30 days for categorization

    global FEATURES_PER_STEP, STATE_SIZE
    FEATURES_PER_STEP = len(features_to_select) # This should be set based on features_to_select
    STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + WINDOW_SIZE + 3 # Re-affirm STATE_SIZE based on actual features
    logging.info(f"Updated global constants: FEATURES_PER_STEP={FEATURES_PER_STEP}, STATE_SIZE={STATE_SIZE}")

    # Step 2: Split the categorized dataset into global train, validation, and test sets
    train_size_global = int(0.8 * len(df_categorized))
    val_size_global = int(0.1 * len(df_categorized))
    
    min_data_needed_for_env = WINDOW_SIZE # Each env needs at least WINDOW_SIZE steps
    min_train_needed_global = min_data_needed_for_env + BATCH_SIZE # Roughly
    min_val_test_needed_global = min_data_needed_for_env

    if train_size_global < min_train_needed_global:
        logging.error(f"Global train size {train_size_global} is less than minimum required {min_train_needed_global}.")
        return None, None, None, None, None

    # Adjust val_size if test set would be too small
    if len(df_categorized) - train_size_global - val_size_global < min_val_test_needed_global:
        val_size_global = max(0, len(df_categorized) - train_size_global - min_val_test_needed_global)
        logging.info(f"Adjusted global val_size to {val_size_global} to ensure minimum global test set size.")

    test_size_global = len(df_categorized) - train_size_global - val_size_global
    if test_size_global < min_val_test_needed_global:
        logging.error(f"Error: Global test set too small ({test_size_global} data points). Need at least {min_val_test_needed_global}.")
        return None, None, None, None, None

    global_train_df = df_categorized.iloc[:train_size_global].copy()
    global_val_df = df_categorized.iloc[train_size_global : train_size_global + val_size_global].copy()
    global_test_df = df_categorized.iloc[train_size_global + val_size_global:].copy()
    
    logging.info(f"Global Data split: Train {len(global_train_df)}, Val {len(global_val_df)}, Test {len(global_test_df)}")

    # Step 3: Calculate normalization statistics from the global training set features
    train_features_for_norm = torch.FloatTensor(global_train_df[features_to_select].values)
    data_mean = train_features_for_norm.mean(dim=0, keepdim=True)
    data_std = train_features_for_norm.std(dim=0, keepdim=True)
    data_std[data_std < EPSILON_NUMERIC] = EPSILON_NUMERIC
    logging.info("Calculated data_mean and data_std from global training data.")

    # Step 4: Create subset environments for train, val, and test
    train_envs, val_envs, test_envs = {}, {}, {}
    market_conditions = ['bull', 'bear', 'sideways', 'high_volatility'] # High volatility might be merged by now

    def create_subset_envs(global_df, subset_name_prefix, data_mean_norm, data_std_norm, features_for_augmentation):
        envs_dict = {}
        if global_df.empty:
            logging.warning(f"Global {subset_name_prefix} DataFrame is empty. No subset environments will be created.")
            return envs_dict
            
        current_market_conditions_in_df = global_df['condition'].unique() # Get actual conditions present

        for condition in current_market_conditions_in_df:
            if condition == 'unknown': continue # Skip unknown category for env creation

            condition_df_original = global_df[global_df['condition'] == condition].copy()
            logging.info(f"Subset: {subset_name_prefix}_{condition}, Original Rows: {len(condition_df_original)}")
            
            condition_df_processed = condition_df_original.copy()

            # Augment data if subset is too small (only for training sets)
            if subset_name_prefix == "train" and len(condition_df_processed) < WINDOW_SIZE and len(condition_df_processed) > 0:
                extra_rows_needed = WINDOW_SIZE - len(condition_df_processed)
                # Ensure we don't try to sample more than available if replace=False, or handle if replace=True
                num_to_sample = min(extra_rows_needed, len(condition_df_original)) if not condition_df_original.empty else 0
                
                if num_to_sample > 0:
                    # For augmentation, sample with replacement is safer if original is very small
                    synthetic_df_source = condition_df_original.sample(n=extra_rows_needed, replace=True, random_state=42)
                    synthetic_df_augmented = synthetic_df_source.copy()
                    
                    # Perturb selected features for augmentation
                    perturb_factor = np.random.uniform(0.95, 1.05, size=(len(synthetic_df_augmented), len(features_for_augmentation)))
                    synthetic_df_augmented[features_for_augmentation] = synthetic_df_augmented[features_for_augmentation] * perturb_factor
                    
                    condition_df_processed = pd.concat([condition_df_original, synthetic_df_augmented]).reset_index(drop=True)
                    logging.info(f"Augmented '{condition}' subset in '{subset_name_prefix}' from {len(condition_df_original)} to {len(condition_df_processed)} rows.")
                else:
                    logging.warning(f"Cannot augment '{condition}' subset in '{subset_name_prefix}' as original sample size is 0 or no extra rows needed.")


            if len(condition_df_processed) >= WINDOW_SIZE:
                condition_features = torch.FloatTensor(condition_df_processed[features_to_select].values) # Use features_to_select global
                envs_dict[condition] = CryptoTradingEnv(condition_df_processed, condition_features, data_mean_norm, data_std_norm)
                logging.info(f"Created {subset_name_prefix} environment for '{condition}' market ({len(condition_df_processed)} rows).")
            else:
                envs_dict[condition] = None
                logging.warning(f"Skipping {subset_name_prefix} environment for '{condition}' market: not enough data ({len(condition_df_processed)} rows, need {WINDOW_SIZE}).")
        return envs_dict

    # Pass features_to_select for augmentation reference
    train_envs = create_subset_envs(global_train_df, "train", data_mean, data_std, features_to_select)
    val_envs = create_subset_envs(global_val_df, "validation", data_mean, data_std, features_to_select) # No augmentation for val/test
    test_envs = create_subset_envs(global_test_df, "test", data_mean, data_std, features_to_select)    # No augmentation for val/test


    # Log a summary of created environments
    for env_set_name, env_set_dict in [("Training", train_envs), ("Validation", val_envs), ("Test", test_envs)]:
        valid_envs_count = sum(1 for env in env_set_dict.values() if env is not None)
        logging.info(f"{env_set_name} environments created: {valid_envs_count} usable out of {len(market_conditions)} conditions.")
        for cond, env_instance in env_set_dict.items():
            if env_instance:
                 logging.info(f"  - {cond}: {env_instance.max_steps} steps, Price range: {env_instance.df['close'].min():.2f} - {env_instance.df['close'].max():.2f}")


    # Check if at least one training environment is usable
    if not any(train_envs.values()):
        logging.error("No usable training environments were created. Cannot proceed.")
        return None, None, None, None, None
        
    return train_envs, val_envs, test_envs, data_mean, data_std


def plot_metrics(metrics):
    plt.figure(figsize=(12, 8)) # Adjusted figure size
    plt.subplot(2, 2, 1)
    plt.plot(metrics["episode_rewards"])
    plt.title("Episode Rewards")
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics["portfolio_values"])
    plt.title("Training Portfolio Values (Per Episode Env)")
    
    plt.subplot(2, 2, 3)
    if "val_portfolio_values" in metrics and metrics["val_portfolio_values"]:
        plt.plot(metrics["val_portfolio_values"])
        plt.title("Weighted Avg Validation Portfolio")
    else:
        plt.title("No Validation Data")

    plt.subplot(2, 2, 4)
    if "val_portfolio_by_condition" in metrics and metrics["val_portfolio_by_condition"]:
        for cond, values in metrics["val_portfolio_by_condition"].items():
            if values: # Check if list is not empty
                plt.plot(values, label=cond)
        plt.legend()
        plt.title("Validation Portfolio by Condition")
    else:
        plt.title("No Condition-Specific Validation Data")
        
    # Optional: Plotting average action probabilities if collected
    # if "avg_episode_probs" in metrics and metrics["avg_episode_probs"]:
    #     probs_array = np.array(metrics["avg_episode_probs"])
    #     if probs_array.ndim == 2 and probs_array.shape[1] == ACTION_SIZE: # Expects list of lists/arrays
    #         plt.figure(figsize=(10, 5))
    #         action_labels = ['Hold', 'Buy', 'Sell']
    #         for i in range(ACTION_SIZE):
    #             plt.plot(probs_array[:, i], label=f'Prob({action_labels[i]})')
    #         plt.title("Average Action Probabilities per Episode (Approx.)")
    #         plt.xlabel("Episode")
    #         plt.ylabel("Probability")
    #         plt.legend()
    #         plt.savefig("action_probabilities.png")
    #         plt.close()


    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

def main():
    train_envs_dict, val_envs_dict, test_envs_dict, data_mean, data_std = load_crypto_data_from_csv()

    if not train_envs_dict or not any(train_envs_dict.values()):
        logging.error("Failed to load data or no usable training environments. Exiting.")
        return

    # Filter out None environments (those that were too small)
    active_train_envs = {k: v for k, v in train_envs_dict.items() if v is not None}
    if not active_train_envs:
        logging.error("All training environment subsets are too small. Exiting.")
        return
    
    logging.info(f"Starting training with {len(active_train_envs)} active training environments: {list(active_train_envs.keys())}")
    
    # Log approximate steps for each active training environment
    for name, env in active_train_envs.items():
        logging.info(f"Training steps for '{name}' env (approx): {env.max_steps - WINDOW_SIZE}")

    active_val_envs = {k: v for k, v in val_envs_dict.items() if v is not None} if val_envs_dict else {}
    if active_val_envs:
        for name, env in active_val_envs.items():
            logging.info(f"Validation steps for '{name}' env (approx): {env.max_steps - WINDOW_SIZE}")
    else:
        logging.info("No active validation environments.")
        
    active_test_envs = {k: v for k, v in test_envs_dict.items() if v is not None} if test_envs_dict else {}


    actor = SACActor(STATE_SIZE, ACTION_SIZE).to(device)
    if hasattr(actor, 'layers') and isinstance(actor.layers[-1], nn.Linear):
        nn.init.constant_(actor.layers[-1].bias, 0.0)
        logging.info("Initialized actor's final layer bias to 0.0")

    critic1 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    critic2 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    target_critic1 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    target_critic2 = SACCritic(STATE_SIZE, ACTION_SIZE).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    target_critic1.eval()
    target_critic2.eval()

    log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True, device=device, dtype=torch.float32)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=LEARNING_RATE)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=LEARNING_RATE)
    alpha_optimizer = optim.Adam([log_alpha], lr=5e-3) # Raise Alpha learning rate

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, STATE_SIZE, device)
    training_metrics = {
        "episode_rewards": [], "portfolio_values": [], "val_portfolio_values": [],
        "alpha_values": [], "total_steps": 0,
        "avg_episode_probs": [], # For logging average action probabilities
        "val_portfolio_by_condition": {} # For condition-specific validation metrics
    }
    best_avg_val_portfolio = -float('inf')
    patience_counter = 0
    current_epsilon = EPSILON_START

    logging.info("--- Training SAC for Crypto Trading (Multi-Environment) ---")

    train_env_names = list(active_train_envs.keys())
    
    # For balanced subset selection
    env_weights_raw = {name: max(1, env.max_steps - WINDOW_SIZE) for name, env in active_train_envs.items()} # Use effective steps
    total_raw_weight = sum(env_weights_raw.values())
    
    if total_raw_weight > 0: # Avoid division by zero if all envs somehow have 0 effective steps
        # Inverse weighting: smaller envs get higher probability
        # Sum of inverses: S = sum(1/steps_i)
        # Prob_i = (1/steps_i) / S
        inverse_weights = {name: 1.0 / count if count > 0 else 1.0 for name, count in env_weights_raw.items()}
        sum_inverse_weights = sum(inverse_weights.values())
        env_selection_probs = {name: w / sum_inverse_weights for name, w in inverse_weights.items()}
        logging.info(f"Environment selection probabilities (inverse weighted by steps): {env_selection_probs}")
    else: # Fallback to uniform if weights can't be calculated
        env_selection_probs = {name: 1.0 / len(train_env_names) for name in train_env_names}
        logging.warning("Fallback to uniform environment selection due to zero total weight.")


    for episode in range(1, EPISODES + 1):
        # Balanced subset selection
        if total_raw_weight > 0 and train_env_names:
             chosen_env_name = random.choices(train_env_names, weights=[env_selection_probs[k] for k in train_env_names], k=1)[0]
        elif train_env_names: # Fallback if weights are problematic but names exist
            chosen_env_name = random.choice(train_env_names)
        else: # Should not happen if active_train_envs is checked
            logging.error("No training environments available to choose from. Breaking loop.")
            break 
        
        current_train_env = active_train_envs[chosen_env_name]
        
        state = current_train_env.reset()
        episode_reward = 0
        done = False
        episode_action_counts = {0: 0, 1: 0, 2: 0}
        
        logging.info(f"Episode {episode}/{EPISODES} starting with environment: '{chosen_env_name}'")

        while not done:
            valid_actions = current_train_env.get_valid_actions()
            if not valid_actions: 
                logging.warning(f"No valid actions in '{chosen_env_name}' env at step {current_train_env.current_step}. Defaulting to Hold (0).")
                valid_actions = [0]

            probs_for_log_display = [0.0, 0.0, 0.0]
            if random.random() < current_epsilon:
                action = random.choice(valid_actions)
                with torch.no_grad(): # Log probs even for random action
                    actor.eval()
                    state_tensor_log = torch.FloatTensor(state).unsqueeze(0).to(device)
                    logits_for_log = actor(state_tensor_log)
                    mask = torch.ones(ACTION_SIZE, device=device) * -1e10
                    for valid_action in valid_actions: mask[valid_action] = 0.0
                    masked_logits = logits_for_log + mask
                    probs_for_log_softmax = F.softmax(masked_logits, dim=-1)
                    actor.train()
                probs_for_log_display = probs_for_log_softmax.cpu().numpy().tolist()[0]
            else:
                with torch.no_grad():
                    actor.eval()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    logits = actor(state_tensor)
                    # Add Gaussian noise to logits
                    logits += torch.normal(0, 0.1, size=logits.shape, device=device)
                    
                    mask = torch.ones(ACTION_SIZE, device=device) * -1e10
                    for valid_action in valid_actions: mask[valid_action] = 0.0
                    masked_logits = logits + mask
                    probs = F.softmax(masked_logits, dim=-1)
                    final_probs = torch.zeros_like(probs)
                    valid_action_tensor = torch.tensor(valid_actions, device=device, dtype=torch.long)
                    if len(valid_actions) > 0 :
                        final_probs.scatter_(1, valid_action_tensor.unsqueeze(0), probs.gather(1, valid_action_tensor.unsqueeze(0)))
                        prob_sum = final_probs.sum(dim=-1, keepdim=True)
                        final_probs = final_probs / (prob_sum + EPSILON_NUMERIC)
                    if final_probs.sum().item() < EPSILON_NUMERIC and len(valid_actions) > 0 :
                        final_probs.fill_(0.0)
                        final_probs[0, valid_action_tensor] = 1.0 / len(valid_actions)
                    elif len(valid_actions) == 0:
                        final_probs.fill_(0.0)
                        final_probs[0,0] = 1.0
                    dist = torch.distributions.Categorical(final_probs)
                    action = dist.sample().item()
                    actor.train()
                probs_for_log_display = final_probs.cpu().numpy().tolist()[0]

            episode_action_counts[action] += 1
            next_state, reward, done = current_train_env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            training_metrics["total_steps"] += 1

            if training_metrics["total_steps"] % 200 == 0: # Log less frequently to reduce noise
                probs_np = np.array(probs_for_log_display)
                logging.info(f"Train Step {training_metrics['total_steps']} (Env: {chosen_env_name}), Action: {action}, "
                             f"Probs: [{probs_np[0]:.3f}, {probs_np[1]:.3f}, {probs_np[2]:.3f}], "
                             f"Ep Action Counts: {episode_action_counts}")

            optimize_model(
                actor, critic1, critic2, target_critic1, target_critic2,
                log_alpha, replay_buffer, actor_optimizer,
                critic1_optimizer, critic2_optimizer, alpha_optimizer
            )

        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
        training_metrics["episode_rewards"].append(episode_reward)
        # Portfolio value from the specific environment instance used in this episode
        current_train_portfolio_val = current_train_env.portfolio_values[-1] if current_train_env.portfolio_values else current_train_env.initial_balance
        training_metrics["portfolio_values"].append(current_train_portfolio_val) # This tracks portfolio of the episode's env
        training_metrics["alpha_values"].append(log_alpha.exp().item())
        
        # Log average action probabilities for this episode
        # Need to collect all probs_for_log_display during the episode. For simplicity, log the last one or an approximation.
        # This was supposed to be for the episode, but probs_for_log_display is per step.
        # Let's log the final step's probabilities as a proxy or average them if collected.
        # For now, let's log the probs of the *last* action selection process of the episode for simplicity.
        # A more accurate way would be to average all `probs_for_log_display` from the episode.
        # current_episode_probs_list = [] # would collect these in the inner loop
        # if current_episode_probs_list:
        #    avg_probs_this_episode = np.mean(current_episode_probs_list, axis=0)
        #    training_metrics["avg_episode_probs"].append(avg_probs_this_episode.tolist())
        #    logging.info(f"Ep {episode} Avg Action Probs (approx from last step): Hold={probs_for_log_display[0]:.3f}, Buy={probs_for_log_display[1]:.3f}, Sell={probs_for_log_display[2]:.3f}")
        # Given the request was "average action probabilities per episode", and the provided snippet was:
        # episode_probs = np.mean([probs_for_log_display], axis=0) which is just the last one.
        # I'll stick to that for now.
        if 'probs_for_log_display' in locals() and probs_for_log_display: # Check if it was set
            training_metrics["avg_episode_probs"].append(probs_for_log_display) # Store the list of 3 probs
            logging.info(f"Ep {episode} Final Action Probs: Hold={probs_for_log_display[0]:.3f}, Buy={probs_for_log_display[1]:.3f}, Sell={probs_for_log_display[2]:.3f}")


        logging.info(f"Ep {episode} ({chosen_env_name}) Reward: {episode_reward:.2f}, Portfolio: ${current_train_portfolio_val:.2f}, Alpha: {log_alpha.exp().item():.3f}, Epsilon: {current_epsilon:.4f}")
        logging.info(f"Train Env '{chosen_env_name}' State: Balance ${current_train_env.balance:.2f}, BTC Held {current_train_env.btc_held:.6f}, Avg Cost ${current_train_env.avg_purchase_price:.2f}")
        logging.info(f"Ep Action Counts: {episode_action_counts}")

        if active_val_envs and episode % VALIDATION_INTERVAL == 0:
            total_val_portfolio_sum = 0
            num_val_envs_processed = 0 # Use this for averaging
            total_val_weights = 0 # For weighted average

            logging.info(f"--- Starting Validation for Episode {episode} ---")
            for val_env_name, val_env_instance in active_val_envs.items():
                val_state = val_env_instance.reset()
                val_done = False
                val_actions = []
                val_step_count = 0
                while not val_done:
                    valid_actions_val = val_env_instance.get_valid_actions()
                    if not valid_actions_val:
                        logging.warning(f"Val Env '{val_env_name}': No valid actions at step {val_env_instance.current_step}. Default Hold.")
                        valid_actions_val = [0]

                    with torch.no_grad():
                        actor.eval()
                        val_state_tensor = torch.FloatTensor(val_state).unsqueeze(0).to(device)
                        logits_val = actor(val_state_tensor)
                        mask_val = torch.ones(ACTION_SIZE, device=device) * -1e10
                        for valid_action in valid_actions_val: mask_val[valid_action] = 0.0
                        masked_logits_val = logits_val + mask_val
                        probs_val = F.softmax(masked_logits_val, dim=-1)
                        final_probs_val = torch.zeros_like(probs_val)
                        valid_action_tensor_val = torch.tensor(valid_actions_val, device=device, dtype=torch.long)
                        if len(valid_actions_val) > 0:
                            final_probs_val.scatter_(1, valid_action_tensor_val.unsqueeze(0), probs_val.gather(1, valid_action_tensor_val.unsqueeze(0)))
                            prob_sum_val = final_probs_val.sum(dim=-1, keepdim=True)
                            final_probs_val = final_probs_val / (prob_sum_val + EPSILON_NUMERIC)
                        if final_probs_val.sum().item() < EPSILON_NUMERIC and len(valid_actions_val) > 0 :
                            final_probs_val.fill_(0.0)
                            final_probs_val[0, valid_action_tensor_val] = 1.0 / len(valid_actions_val)
                        elif len(valid_actions_val) == 0:
                             final_probs_val.fill_(0.0)
                             final_probs_val[0,0] = 1.0
                        dist_val = torch.distributions.Categorical(final_probs_val)
                        action_val = dist_val.sample().item()
                    val_state, _, val_done = val_env_instance.step(action_val)
                    val_actions.append(action_val)
                    val_step_count += 1
                    if val_step_count > val_env_instance.max_steps + 5: # Safety break
                        logging.warning(f"Val Env '{val_env_name}' loop exceeded max_steps.")
                        break
                actor.train() # Set actor back to train mode

                current_val_portfolio_subset = val_env_instance.portfolio_values[-1] if val_env_instance.portfolio_values else val_env_instance.initial_balance
                logging.info(f"Validation Portfolio for '{val_env_name}': ${current_val_portfolio_subset:.2f}")
                logging.info(f"  Val Env '{val_env_name}' State: Balance ${val_env_instance.balance:.2f}, BTC Held {val_env_instance.btc_held:.6f}, Avg Cost ${val_env_instance.avg_purchase_price:.2f}")
                val_action_dist_subset = {i: val_actions.count(i) for i in range(ACTION_SIZE)}
                logging.info(f"  Val Env '{val_env_name}' Action Distribution: {val_action_dist_subset}")
                
                # Store condition-specific validation portfolio
                if val_env_name not in training_metrics["val_portfolio_by_condition"]:
                    training_metrics["val_portfolio_by_condition"][val_env_name] = []
                training_metrics["val_portfolio_by_condition"][val_env_name].append(current_val_portfolio_subset)
                
                total_val_portfolio_sum += current_val_portfolio_subset * len(val_env_instance.df) # Weighted sum
                total_val_weights += len(val_env_instance.df) # Sum of weights (subset sizes)
                num_val_envs_processed +=1 # Count envs that ran

            if total_val_weights > 0: # Use total_val_weights to check if any processing happened
                avg_val_portfolio = total_val_portfolio_sum / total_val_weights # Weighted average
                training_metrics["val_portfolio_values"].append(avg_val_portfolio) # This now stores weighted average
                logging.info(f"--- Weighted Average Validation Portfolio (Ep {episode}): ${avg_val_portfolio:.2f} (across {num_val_envs_processed} subsets, total weight {total_val_weights}) ---")

                if avg_val_portfolio > best_avg_val_portfolio:
                    best_avg_val_portfolio = avg_val_portfolio
                    patience_counter = 0
                    torch.save({
                        'actor_state_dict': actor.state_dict(),
                        'critic1_state_dict': critic1.state_dict(),
                        'critic2_state_dict': critic2.state_dict(),
                    }, "best_sac.pth")
                    logging.info(f"Saved best model to 'best_sac.pth' with avg validation portfolio ${best_avg_val_portfolio:.2f}")
                else:
                    patience_counter += 1
            else:
                logging.warning("No validation environments processed in this interval.")


            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break
        
        # Save metrics periodically
        if episode % 50 == 0 or episode == EPISODES:
            with open("training_metrics.json", "w") as f:
                json.dump(training_metrics, f, indent=4)

    logging.info("--- Test Phase (Multi-Environment) ---")
    overall_final_test_portfolio_value = 0
    num_test_envs_processed = 0
    all_test_results = {} # To store details per test env

    if not active_test_envs:
        logging.error("No active test environments available for final evaluation.")
    else:
        for test_env_name, test_env_instance in active_test_envs.items():
            test_state = test_env_instance.reset()
            test_done = False
            test_actions_taken_subset = []
            test_step_count = 0
            logging.info(f"Testing on environment: '{test_env_name}'")
            while not test_done:
                valid_actions_test = test_env_instance.get_valid_actions()
                if not valid_actions_test:
                    logging.warning(f"Test Env '{test_env_name}': No valid actions. Default Hold.")
                    valid_actions_test = [0]
                
                with torch.no_grad():
                    actor.eval()
                    test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(device)
                    logits_test = actor(test_state_tensor)
                    mask_test = torch.ones(ACTION_SIZE, device=device) * -1e10
                    for valid_action in valid_actions_test: mask_test[valid_action] = 0.0
                    masked_logits_test = logits_test + mask_test
                    probs_test = F.softmax(masked_logits_test, dim=-1)
                    final_probs_test = torch.zeros_like(probs_test)
                    valid_action_tensor_test = torch.tensor(valid_actions_test, device=device, dtype=torch.long)
                    if len(valid_actions_test) > 0:
                        final_probs_test.scatter_(1, valid_action_tensor_test.unsqueeze(0), probs_test.gather(1, valid_action_tensor_test.unsqueeze(0)))
                        prob_sum_test = final_probs_test.sum(dim=-1, keepdim=True)
                        final_probs_test = final_probs_test / (prob_sum_test + EPSILON_NUMERIC)
                    if final_probs_test.sum().item() < EPSILON_NUMERIC and len(valid_actions_test) > 0 :
                        final_probs_test.fill_(0.0)
                        final_probs_test[0, valid_action_tensor_test] = 1.0 / len(valid_actions_test)
                    elif len(valid_actions_test) == 0:
                        final_probs_test.fill_(0.0)
                        final_probs_test[0,0] = 1.0
                    dist_test = torch.distributions.Categorical(final_probs_test)
                    action_test = dist_test.sample().item()

                test_state, _, test_done = test_env_instance.step(action_test)
                test_actions_taken_subset.append(action_test)
                test_step_count += 1
                if test_step_count > test_env_instance.max_steps + 5: # Safety break
                    logging.warning(f"Test Env '{test_env_name}' loop exceeded max_steps.")
                    break
            # actor.train() # Not needed after testing usually, but good practice if code continues

            final_test_portfolio_subset = test_env_instance.portfolio_values[-1] if test_env_instance.portfolio_values else test_env_instance.initial_balance
            logging.info(f"Test Portfolio Final Value for '{test_env_name}': ${final_test_portfolio_subset:.2f}")
            test_action_dist_subset = {i: test_actions_taken_subset.count(i) for i in range(ACTION_SIZE)}
            logging.info(f"  Test Env '{test_env_name}' Action Distribution: {test_action_dist_subset} (Total steps: {len(test_actions_taken_subset)})")
            
            all_test_results[test_env_name] = {
                "final_portfolio": final_test_portfolio_subset,
                "action_distribution": test_action_dist_subset,
                "steps": len(test_actions_taken_subset)
            }
            overall_final_test_portfolio_value += final_test_portfolio_subset
            num_test_envs_processed += 1

    if num_test_envs_processed > 0:
        avg_final_test_portfolio = overall_final_test_portfolio_value / num_test_envs_processed
        logging.info(f"--- Average Test Portfolio Final Value: ${avg_final_test_portfolio:.2f} (across {num_test_envs_processed} subsets) ---")
    else:
        avg_final_test_portfolio = 0 # Or initial balance if preferred.
        logging.info("--- No test environments were processed. Average Test Portfolio is $0 or initial. ---")
    
    # Save final model weights
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
        'data_mean': data_mean, # Save the normalization stats used
        'data_std': data_std
    }, "final_sac.pth")
    logging.info("Saved final model weights (and normalization stats) to 'final_sac.pth'")

    # Save checkpoint
    checkpoint = {
        'episode': episode if 'episode' in locals() else EPISODES, # ensure episode is defined
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
        'final_test_portfolio_avg': avg_final_test_portfolio, # Store the average
        'all_test_results_details': all_test_results, # Store detailed results
        'data_mean': data_mean,
        'data_std': data_std,
        'features_per_step': FEATURES_PER_STEP,
        'window_size': WINDOW_SIZE,
        'state_size': STATE_SIZE
    }
    torch.save(checkpoint, "final_checkpoint.pth")
    logging.info("Saved final checkpoint to 'final_checkpoint.pth'")

    with open("training_metrics.json", "w") as f: # Final save of metrics
        json.dump(training_metrics, f, indent=4)
    plot_metrics(training_metrics)

    logging.info("--- Training Complete ---")

if __name__ == "__main__":
    main()