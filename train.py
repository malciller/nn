#!/usr/bin/env python3
"""
Modular SAC Training Script for Cryptocurrency Trading

This script demonstrates the modular architecture for training a SAC agent
on cryptocurrency trading data with different market conditions.
"""

import time
import random
import numpy as np
from config.config import EPISODES, VALIDATION_INTERVAL, PATIENCE, DEVICE
from utils.logging_setup import setup_logging, get_logger
from utils.plotting import plot_metrics, save_metrics
from data.data_loader import load_crypto_data_from_csv, create_subset_envs
from agents.sac_agent import SACAgent

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_episode(agent, env, episode_num):
    """Train agent for one episode."""
    state = env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Select action
        action = agent.act(state, valid_actions, training=True)
        
        # Execute action
        next_state, reward, done = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Train agent
        actor_loss, critic_loss = agent.train()
        
        total_reward += reward
        step_count += 1
        state = next_state
        
        if done:
            break
    
    portfolio_value = env.get_portfolio_value()
    return total_reward, portfolio_value, step_count

def evaluate_agent(agent, env):
    """Evaluate agent performance without training."""
    state = env.reset()
    total_reward = 0
    
    while True:
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions, training=False)
        next_state, reward, done = env.step(action)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return total_reward, env.get_portfolio_value()

def main():
    """Main training function."""
    # Setup logging
    logger = setup_logging()
    logger.info(f"Starting SAC training on device: {DEVICE}")
    
    # Set seeds for reproducibility
    set_seeds()
    
    # Load and prepare data
    logger.info("Loading cryptocurrency data...")
    df, feature_tensor, global_mean, global_std = load_crypto_data_from_csv()
    
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    
    # Create training environments for different market conditions
    logger.info("Creating market condition environments...")
    train_envs = create_subset_envs(df, "train", global_mean, global_std, feature_tensor)
    
    if not train_envs:
        logger.error("No training environments created. Exiting.")
        return
    
    logger.info(f"Created {len(train_envs)} training environments: {list(train_envs.keys())}")
    
    # Initialize agent
    agent = SACAgent()
    logger.info("SAC agent initialized")
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'final_portfolio_values': [],
        'actor_losses': [],
        'critic_losses': [],
        'validation_rewards': [],
        'validation_portfolio_values': []
    }
    
    best_portfolio_value = 0
    episodes_without_improvement = 0
    
    # Training loop
    logger.info(f"Starting training for {EPISODES} episodes...")
    start_time = time.time()
    
    for episode in range(EPISODES):
        # Select random environment for training
        env_name = random.choice(list(train_envs.keys()))
        env = train_envs[env_name]
        
        # Train episode
        episode_reward, portfolio_value, steps = train_episode(agent, env, episode)
        
        # Log episode results
        metrics['episode_rewards'].append(episode_reward)
        metrics['final_portfolio_values'].append(portfolio_value)
        
        # Get agent metrics
        agent_metrics = agent.get_metrics()
        if agent_metrics['actor_losses']:
            metrics['actor_losses'].extend(agent_metrics['actor_losses'])
        if agent_metrics['critic_losses']:
            metrics['critic_losses'].extend(agent_metrics['critic_losses'])
        
        logger.info(f"Episode {episode+1}/{EPISODES} ({env_name}): "
                   f"Reward={episode_reward:.2f}, Portfolio=${portfolio_value:.2f}, "
                   f"Steps={steps}, Epsilon={agent_metrics['epsilon']:.3f}")
        
        # Validation
        if (episode + 1) % VALIDATION_INTERVAL == 0:
            val_rewards = []
            val_portfolios = []
            
            # Evaluate on all environments
            for val_env_name, val_env in train_envs.items():
                val_reward, val_portfolio = evaluate_agent(agent, val_env)
                val_rewards.append(val_reward)
                val_portfolios.append(val_portfolio)
            
            avg_val_reward = np.mean(val_rewards)
            avg_val_portfolio = np.mean(val_portfolios)
            
            metrics['validation_rewards'].append(avg_val_reward)
            metrics['validation_portfolio_values'].append(avg_val_portfolio)
            
            logger.info(f"Validation - Avg Reward: {avg_val_reward:.2f}, "
                       f"Avg Portfolio: ${avg_val_portfolio:.2f}")
            
            # Check for improvement
            if avg_val_portfolio > best_portfolio_value:
                best_portfolio_value = avg_val_portfolio
                episodes_without_improvement = 0
                agent.save_models("best_sac")
                logger.info(f"New best portfolio value: ${best_portfolio_value:.2f}")
            else:
                episodes_without_improvement += VALIDATION_INTERVAL
                
            # Early stopping
            if episodes_without_improvement >= PATIENCE:
                logger.info(f"Early stopping: No improvement for {PATIENCE} episodes")
                break
        
        # Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.save_models("final_checkpoint")
    
    # Final save
    agent.save_models("final_sac")
    
    # Training summary
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best portfolio value: ${best_portfolio_value:.2f}")
    logger.info(f"Final portfolio value: ${metrics['final_portfolio_values'][-1]:.2f}")
    
    # Save and plot metrics
    save_metrics(metrics)
    plot_metrics(metrics)
    logger.info("Metrics saved and plotted")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    for env_name, env in train_envs.items():
        final_reward, final_portfolio = evaluate_agent(agent, env)
        logger.info(f"Final {env_name}: Reward={final_reward:.2f}, Portfolio=${final_portfolio:.2f}")

if __name__ == "__main__":
    main() 