import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any

# Set global Seaborn & Matplotlib visual styles
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})


class TrainingVisualizer:
    """
    Seaborn & Matplotlib training visualizer for recommender system models.
    Supports post-training dashboards, loss/reward trend curves, action distribution,
    and real-time live plotting during model training.
    """

    def __init__(self, theme: str = "darkgrid", palette: str = "deep"):
        self.theme = theme
        self.palette = palette
        sns.set_theme(style=theme, palette=palette)

    @staticmethod
    def plot_summary_dashboard(
        rewards_history: List[float],
        epsilon_history: List[float],
        action_counts: np.ndarray,
        loss_history: Optional[List[float]] = None,
        save_path: str = "model_data/training_dashboard.png",
        show: bool = False,
        title: str = "Recommender Model Training Dashboard"
    ) -> plt.Figure:
        """
        Generates a multi-panel grid combining Matplotlib and Seaborn to visualize:
        1. Episode Rewards & Moving Average
        2. Training Loss Trend (MSE)
        3. Epsilon Decay (Exploration vs Exploitation)
        4. Recommended Action Distribution (Item Frequencies)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        has_loss = loss_history is not None and len(loss_history) > 0
        num_plots = 4 if has_loss else 3
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10)) if num_plots == 4 else plt.subplots(1, 3, figsize=(18, 5.5))
        axes_flat = axes.flatten() if num_plots == 4 else axes

        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
        episodes_range = np.arange(1, len(rewards_history) + 1)

        # --- Panel 1: Reward Convergence & Moving Avg ---
        ax1 = axes_flat[0]
        sns.lineplot(x=episodes_range, y=rewards_history, ax=ax1, color="#4C72B0", alpha=0.35, label="Raw Episode Reward")
        
        window = min(20, max(1, len(rewards_history) // 10))
        if window > 1 and len(rewards_history) >= window:
            rolling_avg = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
            rolling_episodes = np.arange(window, len(rewards_history) + 1)
            sns.lineplot(x=rolling_episodes, y=rolling_avg, ax=ax1, color="#1D3557", linewidth=2.5, label=f"{window}-Episode Moving Avg")
            
        ax1.set_title("Reward Convergence", fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Cumulative Reward")
        ax1.legend(loc="upper left")

        # --- Panel 2: Training Loss Trend (if available) or Epsilon Decay ---
        if has_loss:
            ax2 = axes_flat[1]
            loss_range = np.arange(1, len(loss_history) + 1)
            sns.lineplot(x=loss_range, y=loss_history, ax=ax2, color="#E63946", alpha=0.4, label="Batch Loss (MSE)")
            
            if len(loss_history) >= window and window > 1:
                loss_smooth = np.convolve(loss_history, np.ones(window) / window, mode="valid")
                sns.lineplot(x=np.arange(window, len(loss_history) + 1), y=loss_smooth, ax=ax2, color="#9B2226", linewidth=2.5, label=f"Smoothed Loss ({window}-Ep)")
                
            ax2.set_title("Training Loss Trend (MSE)", fontweight="bold")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Loss")
            ax2.legend(loc="upper right")

        # --- Panel 3 (or 2): Epsilon Decay ---
        ax3 = axes_flat[2 if has_loss else 1]
        sns.lineplot(x=episodes_range, y=epsilon_history, ax=ax3, color="#2A9D8F", linewidth=2.5, label="Epsilon (Exploration Rate)")
        ax3.set_title("Epsilon Decay Curve", fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Epsilon")
        ax3.legend(loc="upper right")

        # --- Panel 4 (or 3): Action Recommendation Frequency ---
        ax4 = axes_flat[3 if has_loss else 2]
        item_ids = [f"Item {i+1}" for i in range(len(action_counts))]
        
        palette = sns.color_palette("viridis", n_colors=len(action_counts))
        sns.barplot(x=item_ids, y=action_counts, palette=palette, ax=ax4, hue=item_ids, legend=False)
        
        ax4.set_title("Recommended Item Frequency", fontweight="bold")
        ax4.set_xlabel("Product ID")
        ax4.set_ylabel("Selection Count")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] Dashboard plot saved to: {save_path}")

        if show:
            plt.show()
            
        return fig

    @staticmethod
    def plot_reward_distribution(
        rewards_history: List[float],
        save_path: str = "model_data/reward_distribution.png",
        show: bool = False
    ) -> plt.Figure:
        """
        Plots a Seaborn KDE density plot showing early vs late training reward distributions.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        halfway = len(rewards_history) // 2
        early_rewards = rewards_history[:halfway]
        late_rewards = rewards_history[halfway:]

        sns.kdeplot(early_rewards, ax=ax, fill=True, color="#E63946", label="First Half of Training", alpha=0.5)
        sns.kdeplot(late_rewards, ax=ax, fill=True, color="#2A9D8F", label="Second Half of Training", alpha=0.5)

        ax.set_title("Reward Distribution Shift (Early vs. Late Training)", fontweight="bold", fontsize=15)
        ax.set_xlabel("Episode Reward")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] Reward distribution plot saved to: {save_path}")

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_keras_history(
        history_dict: Dict[str, List[float]],
        metrics: List[str] = ["loss", "val_loss"],
        save_path: str = "model_data/keras_training_history.png",
        show: bool = False
    ) -> plt.Figure:
        """
        Plots loss and metric trends from Keras model fit history.
        """
        epochs = range(1, len(history_dict.get(metrics[0], [])) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric in metrics:
            if metric in history_dict:
                sns.lineplot(x=epochs, y=history_dict[metric], ax=ax, label=metric.capitalize(), linewidth=2)

        ax.set_title("Model Training & Validation Performance", fontweight="bold", fontsize=15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] Keras training history plot saved to: {save_path}")

        if show:
            plt.show()

        return fig


class LiveTrainingPlotter:
    """
    Interactive real-time Matplotlib & Seaborn plotter that dynamically updates figure window
    during model training loops.
    """

    def __init__(self, total_episodes: int, update_interval: int = 10):
        self.total_episodes = total_episodes
        self.update_interval = update_interval
        
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.suptitle("⚡ Live Model Training Monitor", fontsize=16, fontweight="bold")
        
        self.episodes = []
        self.rewards = []
        self.epsilons = []

    def update(self, episode: int, reward: float, epsilon: float, action_counts: np.ndarray):
        """Update training metrics live in matplotlib window."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)

        if episode % self.update_interval != 0 and episode != self.total_episodes:
            return

        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Plot 1: Live Reward
        self.axes[0].plot(self.episodes, self.rewards, color="#4C72B0", alpha=0.4, label="Episode Reward")
        if len(self.rewards) >= 10:
            window = min(20, len(self.rewards))
            rolling_avg = np.convolve(self.rewards, np.ones(window) / window, mode="valid")
            self.axes[0].plot(range(window, len(self.rewards) + 1), rolling_avg, color="#1D3557", linewidth=2, label=f"{window}-Ep Moving Avg")
        self.axes[0].set_title("Live Reward Convergence")
        self.axes[0].set_xlabel("Episode")
        self.axes[0].set_ylabel("Reward")
        self.axes[0].legend(loc="upper left")

        # Plot 2: Live Epsilon
        self.axes[1].plot(self.episodes, self.epsilons, color="#2A9D8F", linewidth=2)
        self.axes[1].set_title("Live Epsilon Decay")
        self.axes[1].set_xlabel("Episode")
        self.axes[1].set_ylabel("Epsilon")

        # Plot 3: Live Action Distribution
        items = [f"I{i+1}" for i in range(len(action_counts))]
        self.axes[2].bar(items, action_counts, color="#E76F51", alpha=0.85)
        self.axes[2].set_title("Action Distribution")
        self.axes[2].set_xlabel("Item")
        self.axes[2].set_ylabel("Count")

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close interactive plotting session."""
        plt.ioff()
        plt.close(self.fig)
