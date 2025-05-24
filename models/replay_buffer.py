import torch
import random
from collections import deque
from config.config import DEVICE, STATE_SIZE

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning agents."""
    
    def __init__(self, capacity, state_size=STATE_SIZE, device=DEVICE):
        self.capacity = capacity
        self.state_size = state_size
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            return None
            
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        
        states = torch.stack(batch[0])
        actions = torch.stack(batch[1])
        rewards = torch.stack(batch[2])
        next_states = torch.stack(batch[3])
        dones = torch.stack(batch[4])
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size 