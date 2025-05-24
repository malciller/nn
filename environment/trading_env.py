import numpy as np
import torch
from collections import deque
from config.config import (
    WINDOW_SIZE, TRADING_FEE, PORTFOLIO_HISTORY_SIZE, 
    EPSILON_NUMERIC, DEVICE
)
from utils.logging_setup import get_logger

logger = get_logger(__name__)

class CryptoTradingEnv:
    """Cryptocurrency trading environment for reinforcement learning."""
    
    def __init__(self, df, feature_tensor, global_mean, global_std):
        self.df = df.reset_index(drop=True)
        self.feature_tensor = feature_tensor.to(DEVICE)
        self.current_step = WINDOW_SIZE
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.btc_held = 0
        self.max_steps = len(self.df)
        self.portfolio_values = deque(maxlen=PORTFOLIO_HISTORY_SIZE)
        self.global_mean = global_mean.to(DEVICE) if global_mean is not None else None
        self.global_std = global_std.to(DEVICE) if global_std is not None else None
        self.avg_purchase_price = 0.0
        self.consecutive_holds = 0
        self.balance_history = deque(maxlen=WINDOW_SIZE)
        self.balance_history.append(self.initial_balance)
        
        logger.info(f"Env init: df_length={len(self.df)}, feature_tensor_shape={self.feature_tensor.shape}, max_steps={self.max_steps}")
        if not self.df.empty:
            logger.info(f"Price range in passed df: {self.df['close'].min():.2f} - {self.df['close'].max():.2f}")

    def reset(self):
        """Reset environment to initial state."""
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
        """Get current normalized state representation."""
        end_idx = self.current_step
        start_idx = end_idx - WINDOW_SIZE
        features_window = self.feature_tensor[start_idx:end_idx]
        
        if features_window.shape[0] != WINDOW_SIZE:
            logger.critical(f"CRITICAL: features_window length {features_window.shape[0]} != WINDOW_SIZE {WINDOW_SIZE}.")
        
        # Normalize features
        if self.global_mean is not None and self.global_std is not None:
            normalized_features = (features_window - self.global_mean) / self.global_std
        else:
            features_mean = features_window.mean(dim=0)
            features_std = features_window.std(dim=0) + EPSILON_NUMERIC
            normalized_features = (features_window - features_mean) / features_std
        
        # Portfolio state
        normalized_consecutive_holds = min(self.consecutive_holds, 10) / 10.0
        normalized_balance = self.balance / self.initial_balance
        balance_features = torch.tensor([normalized_balance] * WINDOW_SIZE, device=DEVICE, dtype=torch.float32)
        
        state_portfolio_info = torch.tensor([
            normalized_balance,
            self.btc_held,
            normalized_consecutive_holds
        ], device=DEVICE, dtype=torch.float32)
        
        state = torch.cat([normalized_features.flatten(), balance_features, state_portfolio_info])
        return state.cpu().numpy()

    def get_valid_actions(self):
        """Get list of valid actions based on current state."""
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
        btc_units_to_sell = min(trade_amount_usd / current_price if current_price > EPSILON_NUMERIC else 0, self.btc_held)
        if btc_units_to_sell > EPSILON_NUMERIC:
            valid_actions.append(2)

        return valid_actions

    def step(self, action):
        """Execute action and return next state, reward, done."""
        if self.current_step >= len(self.df):
            logger.error(f"Error: current_step {self.current_step} >= df length {len(self.df)}")
            return self._get_state(), -100, True

        current_df_row = self.df.iloc[self.current_step]
        current_price = current_df_row['close']
        
        if np.isnan(current_price) or current_price <= 0:
            logger.warning(f"Warning: Invalid price {current_price} at step {self.current_step}")
            return self._get_state(), -0.1, False

        done = False
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            reward += self._execute_buy(current_price)
        elif action == 2:  # Sell
            reward += self._execute_sell(current_price)
        else:  # Hold
            self.consecutive_holds += 1

        # Move to next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Calculate portfolio value and reward
        reward += self._calculate_portfolio_reward()
        
        # Update balance history
        self.balance_history.append(self.balance)

        next_state = self._get_state()
        return next_state, reward, done

    def _execute_buy(self, current_price):
        """Execute buy action."""
        trade_amount_usd = min(100.0, self.balance * 0.1)
        btc_units_to_buy = trade_amount_usd / current_price if current_price > EPSILON_NUMERIC else 0
        cost = btc_units_to_buy * current_price * (1 + TRADING_FEE)
        
        if btc_units_to_buy > EPSILON_NUMERIC and cost <= self.balance:
            # Update average purchase price
            total_btc_after = self.btc_held + btc_units_to_buy
            if total_btc_after > EPSILON_NUMERIC:
                self.avg_purchase_price = (
                    (self.avg_purchase_price * self.btc_held + current_price * btc_units_to_buy) / total_btc_after
                )
            
            self.btc_held += btc_units_to_buy
            self.balance -= cost
            self.consecutive_holds = 0
            
            # Small negative reward for transaction cost
            return -cost * 0.001
        else:
            # Invalid buy action
            return -0.1

    def _execute_sell(self, current_price):
        """Execute sell action."""
        trade_amount_usd = min(100.0, self.balance * 0.1)
        btc_units_to_sell = min(trade_amount_usd / current_price if current_price > EPSILON_NUMERIC else 0, self.btc_held)
        
        if btc_units_to_sell > EPSILON_NUMERIC:
            revenue = btc_units_to_sell * current_price * (1 - TRADING_FEE)
            self.btc_held -= btc_units_to_sell
            self.balance += revenue
            self.consecutive_holds = 0
            
            # Reward based on profit/loss vs average purchase price
            if self.avg_purchase_price > EPSILON_NUMERIC:
                profit_per_unit = current_price - self.avg_purchase_price
                total_profit = profit_per_unit * btc_units_to_sell
                return total_profit * 0.01  # Scale reward
            return revenue * 0.001  # Small positive reward
        else:
            # Invalid sell action
            return -0.1

    def _calculate_portfolio_reward(self):
        """Calculate reward based on portfolio value change."""
        current_price = self.df.iloc[min(self.current_step, len(self.df)-1)]['close']
        if np.isnan(current_price) or current_price <= 0:
            current_price = self.df.iloc[self.current_step-1]['close']
        
        portfolio_value_now = self.balance + self.btc_held * current_price
        self.portfolio_values.append(portfolio_value_now)

        reward = 0
        if len(self.portfolio_values) >= 2:
            prev_value = list(self.portfolio_values)[-2]
            if self.initial_balance > EPSILON_NUMERIC:
                portfolio_change_percent = (portfolio_value_now - prev_value) / self.initial_balance
                reward = np.clip(portfolio_change_percent * 10, -1, 1)

        # Penalty for holding too long without action
        if self.consecutive_holds > 20:
            reward -= 0.01

        return reward

    def get_portfolio_value(self):
        """Get current portfolio value."""
        if self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]['close']
            return self.balance + self.btc_held * current_price
        return self.balance

    def get_info(self):
        """Get environment info for debugging."""
        return {
            'current_step': self.current_step,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'portfolio_value': self.get_portfolio_value(),
            'consecutive_holds': self.consecutive_holds
        } 