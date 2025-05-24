import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from models.sac_models import SACActor, SACCritic
from models.replay_buffer import ReplayBuffer
from config.config import (
    STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA, TAU, 
    TARGET_ENTROPY, REPLAY_BUFFER_SIZE, BATCH_SIZE, DEVICE,
    EPSILON_START, EPSILON_DECAY, MIN_EPSILON
)
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def soft_update(target, source, tau):
    """Soft update target network parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

class SACAgent:
    """Soft Actor-Critic agent for trading."""
    
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        
        # Networks
        self.actor = SACActor(state_size, action_size).to(DEVICE)
        self.critic1 = SACCritic(state_size, action_size).to(DEVICE)
        self.critic2 = SACCritic(state_size, action_size).to(DEVICE)
        self.target_critic1 = SACCritic(state_size, action_size).to(DEVICE)
        self.target_critic2 = SACCritic(state_size, action_size).to(DEVICE)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LEARNING_RATE)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LEARNING_RATE)
        
        # Temperature parameter
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=DEVICE, dtype=torch.float32)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, state_size, DEVICE)
        
        # Metrics
        self.actor_losses = []
        self.critic_losses = []

    def act(self, state, valid_actions=None, training=True):
        """Select action using actor network or epsilon-greedy."""
        if training and random.random() < self.epsilon:
            # Epsilon-greedy exploration
            if valid_actions:
                return random.choice(valid_actions)
            else:
                return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        # Set actor to eval mode for inference to handle batch norm with single sample
        was_training = self.actor.training
        self.actor.eval()
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        # Restore original training mode
        if was_training:
            self.actor.train()
            
        # Mask invalid actions if provided
        if valid_actions:
            masked_probs = torch.zeros_like(action_probs)
            for action in valid_actions:
                masked_probs[0, action] = action_probs[0, action]
            
            if masked_probs.sum() > 0:
                action_probs = masked_probs / masked_probs.sum()
            else:
                # Fallback: uniform distribution over valid actions
                action_probs = torch.zeros_like(action_probs)
                for action in valid_actions:
                    action_probs[0, action] = 1.0 / len(valid_actions)
        
        if training:
            # Sample from distribution
            action = torch.multinomial(action_probs, 1).item()
        else:
            # Greedy action selection
            action = action_probs.argmax().item()
            
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        """Train the agent using SAC algorithm."""
        if not self.replay_buffer.is_ready(BATCH_SIZE):
            return None, None
        
        # Sample batch
        batch = self.replay_buffer.sample(BATCH_SIZE)
        if batch is None:
            return None, None
            
        states, actions, rewards, next_states, dones = batch
        
        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor and temperature
        actor_loss = self._update_actor(states)
        
        # Soft update target networks
        soft_update(self.target_critic1, self.critic1, TAU)
        soft_update(self.target_critic2, self.critic2, TAU)
        
        # Update epsilon
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        
        # Store losses
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
        return actor_loss, critic_loss

    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks."""
        with torch.no_grad():
            # Get next actions from actor
            next_action_probs = self.actor(next_states)
            next_actions_one_hot = F.gumbel_softmax(torch.log(next_action_probs + 1e-8), hard=True)
            
            # Get target Q-values
            target_q1 = self.target_critic1(next_states, next_actions_one_hot)
            target_q2 = self.target_critic2(next_states, next_actions_one_hot)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            alpha = self.log_alpha.exp()
            entropy = -torch.sum(next_action_probs * torch.log(next_action_probs + 1e-8), dim=-1, keepdim=True)
            target_q = target_q + alpha * entropy
            
            # Compute target
            target = rewards + (1 - dones.float()) * GAMMA * target_q
        
        # Convert actions to one-hot
        actions_one_hot = F.one_hot(actions.squeeze(), self.action_size).float()
        
        # Current Q-values
        current_q1 = self.critic1(states, actions_one_hot)
        current_q2 = self.critic2(states, actions_one_hot)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return (critic1_loss + critic2_loss).item() / 2

    def _update_actor(self, states):
        """Update actor network and temperature parameter."""
        # Get action probabilities
        action_probs = self.actor(states)
        
        # Sample actions using Gumbel-Softmax
        actions_one_hot = F.gumbel_softmax(torch.log(action_probs + 1e-8), hard=True)
        
        # Get Q-values
        q1 = self.critic1(states, actions_one_hot)
        q2 = self.critic2(states, actions_one_hot)
        q = torch.min(q1, q2)
        
        # Compute entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1, keepdim=True)
        
        # Actor loss
        alpha = self.log_alpha.exp()
        actor_loss = -torch.mean(q + alpha * entropy)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # Update temperature parameter
        alpha_loss = -torch.mean(self.log_alpha * (entropy + TARGET_ENTROPY).detach())
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return actor_loss.item()

    def save_models(self, filepath_prefix):
        """Save model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, f"{filepath_prefix}.pth")

    def load_models(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        logger.info(f"Models loaded from {filepath}")

    def get_metrics(self):
        """Get training metrics."""
        return {
            'actor_losses': self.actor_losses.copy(),
            'critic_losses': self.critic_losses.copy(),
            'epsilon': self.epsilon,
            'alpha': self.log_alpha.exp().item()
        } 