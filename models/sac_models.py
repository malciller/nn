import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import STATE_SIZE, ACTION_SIZE, DEVICE

class SACActor(nn.Module):
    """SAC Actor network for action probability distribution."""
    
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
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
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        """Forward pass returning action probabilities."""
        x = self.layers(state)
        return F.softmax(x, dim=-1)

class SACCritic(nn.Module):
    """SAC Critic network for Q-value estimation."""
    
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        super(SACCritic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size + action_size, 512),
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
            nn.Linear(128, 1)
        )

    def forward(self, state, action_one_hot):
        """Forward pass with state and action concatenation."""
        # Ensure action_one_hot is float, matching state type
        action_one_hot = action_one_hot.float()
        x = torch.cat([state, action_one_hot], dim=1)
        return self.layers(x) 