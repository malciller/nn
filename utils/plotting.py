import matplotlib.pyplot as plt
import json
import os

def plot_metrics(metrics, save_path="training_metrics.png"):
    """Plot training metrics including rewards, portfolio values, and losses."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SAC Training Metrics', fontsize=16)

    # Episode Rewards
    if 'episode_rewards' in metrics:
        axs[0, 0].plot(metrics['episode_rewards'], color='blue', alpha=0.7)
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True, alpha=0.3)

    # Portfolio Values
    if 'final_portfolio_values' in metrics:
        axs[0, 1].plot(metrics['final_portfolio_values'], color='green', alpha=0.7)
        axs[0, 1].set_title('Final Portfolio Values')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Portfolio Value ($)')
        axs[0, 1].grid(True, alpha=0.3)

    # Actor Loss
    if 'actor_losses' in metrics:
        axs[1, 0].plot(metrics['actor_losses'], color='red', alpha=0.7)
        axs[1, 0].set_title('Actor Loss')
        axs[1, 0].set_xlabel('Training Step')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True, alpha=0.3)

    # Critic Loss
    if 'critic_losses' in metrics:
        axs[1, 1].plot(metrics['critic_losses'], color='orange', alpha=0.7)
        axs[1, 1].set_title('Critic Loss')
        axs[1, 1].set_xlabel('Training Step')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_metrics(metrics, save_path="training_metrics.json"):
    """Save metrics to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(load_path="training_metrics.json"):
    """Load metrics from JSON file."""
    if os.path.exists(load_path):
        with open(load_path, 'r') as f:
            return json.load(f)
    return {} 