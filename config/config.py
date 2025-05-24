import numpy as np
import torch

# Device Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Model Hyperparameters
WINDOW_SIZE = 20
FEATURES_PER_STEP = 7
STATE_SIZE = WINDOW_SIZE * FEATURES_PER_STEP + WINDOW_SIZE + 3
ACTION_SIZE = 3

# Training Hyperparameters
BATCH_SIZE = 256
EPISODES = 50
GAMMA = 0.99
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 50000
VALIDATION_INTERVAL = 1
PATIENCE = 100

# SAC Specific
TAU = 0.005
ALPHA = 0.2
TARGET_ENTROPY = -np.log(1.0 / ACTION_SIZE) * 1.2

# Environment Parameters
TRADING_FEE = 0.0035
PORTFOLIO_HISTORY_SIZE = 365
EPSILON_NUMERIC = 1e-9
EPSILON_START = 0.5
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1

# Market Categorization Parameters
MARKET_WINDOW = 30
ATR_THRESHOLD = 0.05
BULL_RETURN_THRESHOLD = 0.2
BEAR_RETURN_THRESHOLD = -0.2

# Data Paths
DATA_PATH = "data/BTCUSD_Daily_1_1_2020__5_21_2025.csv"
CATEGORIZED_DATA_PATH = "data/categorized_BTCUSD.csv"

# Logging
LOG_FILE_PATH = "training_run.log" 