import json
import os
import random
import warnings
from collections import deque
import pandas as pd

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.visualizer import TrainingVisualizer, LiveTrainingPlotter

warnings.filterwarnings("ignore")

# MLflow logging (fallback to local if DagsHub auth fails)
import mlflow

DAGSHUB_REPO_OWNER = "dagshub_name"
DAGSHUB_REPO_NAME = "dagshub_name"
DAGSHUB_TOKEN = "dagshub_token"

try:
    import dagshub
    import dagshub.auth
    dagshub.auth.add_app_token(DAGSHUB_TOKEN)
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
except Exception as e:
    print(f"DagsHub init skipped (using local MLflow): {e}")

mlflow.set_experiment("hybrid-recommender")


class ECommerceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset_path="Ecommerce_Dataset.csv"):
        super(ECommerceEnv, self).__init__()
        self.history_length = 20
        self.vocab_size = 11  # 10 products + padding token

        self.observation_space = spaces.Box(
            low=0, high=self.vocab_size, shape=(self.history_length,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(10)
        self.history = deque(maxlen=self.history_length)
        self.state = None
        self.steps = 0
        self.max_steps = 20

        # Load real dataset if available
        self.real_dataset = None
        self.user_sessions = []
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                if 'user_id' in df.columns and 'product_id' in df.columns:
                    self.real_dataset = df
                    for uid, group in df.groupby('user_id'):
                        seq = group['product_id'].tolist()
                        if len(seq) > 0:
                            self.user_sessions.append((uid, seq, group['event_type'].tolist()))
                    print(f"[INFO] ECommerceEnv loaded real dataset '{dataset_path}' ({len(self.user_sessions)} user sessions).")
            except Exception as e:
                print(f"[WARNING] Could not load dataset '{dataset_path}': {e}")

    def _generate_user_behavior(self):
        purchases = np.random.randint(0, 11) / 10
        clicks = np.random.randint(0, 21) / 20
        return np.array([purchases, clicks], dtype=np.float32)

    def _generate_user_demographics(self):
        age = np.random.rand(1)
        gender = np.zeros(2)
        gender[np.random.randint(0, 2)] = 1
        location = np.zeros(5)
        location[np.random.randint(0, 5)] = 1
        return np.concatenate([age, gender, location])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.history.clear()
        self.current_behavior = self._generate_user_behavior()
        self.current_demographics = self._generate_user_demographics()

        # Seed initial interaction history from real user session if available
        if self.user_sessions and np.random.rand() < 0.7:
            session = self.user_sessions[np.random.randint(0, len(self.user_sessions))]
            raw_seq = session[1][:self.history_length]
            for item in raw_seq:
                idx = (abs(hash(str(item))) % 10) + 1
                self.history.append(idx)

        self._update_state()
        return self.state, {}

    def _update_state(self):
        state = list(self.history)
        while len(state) < self.history_length:
            state.insert(0, 0)
        self.state = np.array(state, dtype=np.int32)

    def step(self, action):
        self.steps += 1
        self.history.append(action)

        location_vector = self.current_demographics[3:]
        location_index = np.argmax(location_vector)

        click_prob = 0.15
        if action == location_index:
            click_prob = 0.45 + (0.4 * self.current_behavior[0])

        clicked = np.random.rand() < click_prob

        if clicked:
            reward = self._calculate_reward("click")
            if np.random.rand() < 0.3:
                reward += self._calculate_reward("purchase")
            self.current_behavior[0] = min(1.0, self.current_behavior[0] + 0.1)
            self.current_behavior[1] = min(1.0, self.current_behavior[1] + 0.05)
        else:
            reward = self._calculate_reward("no_action")

        self._update_state()
        done = self.steps >= self.max_steps
        return self.state, reward, done, False, {}

    def _calculate_reward(self, event):
        reward_map = {"purchase": 2.5, "click": 1.0, "view": 0.2, "no_action": -0.05}
        return reward_map.get(event, 0.0)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_hybrid_model(history_length, action_size, vocab_size=12):
    """
    Builds a Dueling Transformer-DQN Hybrid Model.
    Separates state value V(s) and advantage A(s,a) streams for faster learning.
    """
    inputs = Input(shape=(history_length,), dtype=tf.int32)
    embedding = Embedding(vocab_size, 32)(inputs)
    x = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)(embedding)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)

    # State Value Stream V(s)
    val_fc = Dense(32, activation="relu")(x)
    value = Dense(1)(val_fc)

    # Advantage Stream A(s, a)
    adv_fc = Dense(32, activation="relu")(x)
    advantage = Dense(action_size)(adv_fc)

    # Dueling Combination: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    def dueling_combine(args):
        val, adv = args
        return val + (adv - tf.reduce_mean(adv, axis=-1, keepdims=True))

    outputs = tf.keras.layers.Lambda(dueling_combine, name="q_values")([value, advantage])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.Huber(delta=1.0),
        optimizer=Adam(learning_rate=0.0003, clipnorm=1.0)
    )
    return model


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer sampling high-error transitions more frequently."""

    def __init__(self, max_size=10000, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, transition, td_error=1.0):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], []
        probs = np.array(self.priorities, dtype=np.float32)
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_err) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class OfflineRLAgent:
    def __init__(self, state_size, action_size, model):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        # Target network for stable Double DQN target computation
        self.target_model = build_hybrid_model(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        
        self.memory = PrioritizedReplayBuffer(max_size=10000, alpha=0.6)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.gamma = 0.99
        self.tau = 0.005  # Smooth target update rate
        self.warmup_steps = 150

    def remember(self, state, action, reward, next_state, done, behavior_prob):
        # Initial default high priority so new experiences get trained at least once
        max_p = max(self.memory.priorities) if len(self.memory) > 0 else 1.0
        self.memory.add((state, action, reward, next_state, done, behavior_prob), td_error=max_p)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            behavior_prob = 1.0 / self.action_size
            return action, behavior_prob

        act_values = self.model.predict_on_batch(state)
        action = np.argmax(act_values[0])
        exp_q = np.exp(act_values[0] - np.max(act_values[0]))
        probs = exp_q / np.sum(exp_q)
        behavior_prob = probs[action]
        return action, behavior_prob

    def update_target_network(self):
        """Soft update target network weights: theta_target = tau*theta_main + (1-tau)*theta_target"""
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(main_weights)):
            target_weights[i] = self.tau * main_weights[i] + (1.0 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def replay(self, batch_size):
        if len(self.memory) < max(batch_size, self.warmup_steps):
            return None

        minibatch, indices = self.memory.sample(batch_size)
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Current Q-values from main model
        targets = self.model.predict_on_batch(states)
        
        # Double DQN target computation:
        # Action selection from main model, Action evaluation from target model
        next_actions_main = np.argmax(self.model.predict_on_batch(next_states), axis=1)
        next_q_target = self.target_model.predict_on_batch(next_states)

        td_errors = []
        for i in range(len(minibatch)):
            old_q = targets[i][actions[i]]
            if dones[i]:
                target_q = rewards[i]
            else:
                best_action_next = next_actions_main[i]
                target_q = rewards[i] + self.gamma * next_q_target[i][best_action_next]
            
            targets[i][actions[i]] = target_q
            td_errors.append(abs(target_q - old_q))

        # Update PER priorities based on TD-errors
        self.memory.update_priorities(indices, td_errors)

        loss = self.model.train_on_batch(states, targets)
        self.update_target_network()
        return float(loss)


def generate_training_plots(rewards_history, epsilon_history, action_counts, loss_history=None, save_path="model_data/training_dashboard.png", show=True):
    """Generates and saves visual charts of training metrics using Seaborn & Matplotlib."""
    visualizer = TrainingVisualizer()
    visualizer.plot_summary_dashboard(
        rewards_history=rewards_history,
        epsilon_history=epsilon_history,
        action_counts=action_counts,
        loss_history=loss_history,
        save_path=save_path,
        show=False,
        title="Transformer-DQN Hybrid Recommender Training Dashboard"
    )
    visualizer.plot_reward_distribution(
        rewards_history=rewards_history,
        save_path="model_data/reward_distribution.png",
        show=False
    )
    if show:
        print("\n[INFO] Displaying Matplotlib & Seaborn visualization window...")
        plt.show()


import argparse

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid RL Recommender with Matplotlib & Seaborn Visualizations")
    parser.add_argument("--episodes", type=int, default=600, help="Total training episodes")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs (1 Epoch = 1 full pass over user sessions dataset)")
    parser.add_argument("--batch-size", type=int, default=32, help="RL replay batch size")
    parser.add_argument("--live", action="store_true", help="Enable real-time interactive plotting during training")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Disable displaying Matplotlib/Seaborn plots on screen after training")
    parser.add_argument("--show", action="store_true", dest="show", default=True, help="Display Matplotlib/Seaborn plots on screen after training (default: True)")
    args = parser.parse_args()

    env = ECommerceEnv()
    state_size = int(env.history_length)
    action_size = int(env.action_space.n)

    num_sessions = len(env.user_sessions) if env.user_sessions else 50
    if args.epochs is not None and args.epochs > 0:
        total_epochs = args.epochs
        episodes = total_epochs * num_sessions
        episodes_per_epoch = num_sessions
        print(f"[INFO] Training for {total_epochs} Epochs ({episodes_per_epoch} user sessions per epoch = {episodes} total episodes)")
    else:
        episodes = args.episodes
        total_epochs = max(1, episodes // num_sessions)
        episodes_per_epoch = num_sessions

    print(f"Building Transformer-DQN model: Input={state_size}, Output={action_size}")
    model = build_hybrid_model(state_size, action_size)
    agent = OfflineRLAgent(state_size, action_size, model)

    batch_size = args.batch_size

    rewards_history = []
    epsilon_history = []
    loss_history = []
    action_counts = np.zeros(action_size, dtype=int)

    live_plotter = LiveTrainingPlotter(total_episodes=episodes, update_interval=10) if args.live else None

    with mlflow.start_run(run_name="Transformer_DQN_Run"):
        mlflow.log_params({"episodes": episodes, "batch_size": batch_size, "epochs": total_epochs})
        print(f"\n=== Starting RL Model Training ({total_epochs} Epochs / {episodes} Episodes) ===\n")

        for episode in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            episode_reward = 0
            ep_losses = []

            for step in range(200):
                action, behavior_prob = agent.act(state)
                action_counts[action] += 1
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                agent.remember(state, action, reward, next_state, done, behavior_prob)
                episode_reward += reward
                state = next_state
                loss = agent.replay(batch_size)
                if loss is not None:
                    ep_losses.append(loss)

                if done:
                    break

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            rewards_history.append(episode_reward)
            epsilon_history.append(agent.epsilon)
            avg_loss = float(np.mean(ep_losses)) if len(ep_losses) > 0 else 0.0
            loss_history.append(avg_loss)

            if live_plotter:
                live_plotter.update(episode + 1, episode_reward, agent.epsilon, action_counts)

            mlflow.log_metric("reward", episode_reward, step=episode)
            mlflow.log_metric("epsilon", agent.epsilon, step=episode)
            if avg_loss > 0:
                mlflow.log_metric("loss", avg_loss, step=episode)

            curr_ep = (episode + 1) // episodes_per_epoch + 1
            if (episode + 1) % episodes_per_epoch == 0 or (episode + 1) == episodes:
                print(
                    f"Epoch {min(total_epochs, (episode+1)//episodes_per_epoch)}/{total_epochs} (Episode {episode+1}/{episodes}) | Avg Reward: {np.mean(rewards_history[-episodes_per_epoch:]):.2f} | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}"
                )

        if live_plotter:
            live_plotter.close()

        # Statistical Metrics Summary
        initial_avg_reward = np.mean(rewards_history[:min(50, len(rewards_history))])
        final_avg_reward = np.mean(rewards_history[-min(50, len(rewards_history)):])
        max_reward = float(np.max(rewards_history))
        avg_reward = float(np.mean(rewards_history))
        reward_lift = float(final_avg_reward - initial_avg_reward)

        print("\n" + "=" * 55)
        print("           TRAINING PERFORMANCE METRICS SUMMARY          ")
        print("=" * 55)
        print(f" Total Episodes Trained   : {episodes}")
        print(f" Initial 50-Ep Avg Reward : {initial_avg_reward:.2f}")
        print(f" Final 50-Ep Avg Reward   : {final_avg_reward:.2f}")
        print(f" Reward Improvement (Lift): {reward_lift:+.2f}")
        print(f" Overall Max Reward       : {max_reward:.2f}")
        print(f" Overall Mean Reward      : {avg_reward:.2f}")
        print(f" Final Epsilon            : {agent.epsilon:.4f}")
        print("=" * 55 + "\n")

        mlflow.log_metrics({
            "initial_50ep_avg_reward": initial_avg_reward,
            "final_50ep_avg_reward": final_avg_reward,
            "reward_lift": reward_lift,
            "max_reward": max_reward,
            "mean_reward": avg_reward,
            "final_epsilon": agent.epsilon
        })

        # Save JSON history for interactive web dashboard
        history_json_path = "model_data/training_history.json"
        history_data = {
            "episodes": episodes,
            "rewards": [float(r) for r in rewards_history],
            "epsilon": [float(e) for e in epsilon_history],
            "losses": [float(l) for l in loss_history],
            "action_counts": [int(c) for c in action_counts],
            "summary": {
                "initial_50ep_avg_reward": float(initial_avg_reward),
                "final_50ep_avg_reward": float(final_avg_reward),
                "reward_lift": float(reward_lift),
                "max_reward": max_reward,
                "mean_reward": avg_reward,
                "final_epsilon": float(agent.epsilon)
            }
        }
        try:
            with open(history_json_path, "w") as f:
                json.dump(history_data, f, indent=2)
            print(f"Exported interactive training metrics to '{history_json_path}'")
        except Exception as e:
            print(f"Failed to export JSON metrics: {e}")

        # Generate & Log Graphical Dashboard Plot using Matplotlib and Seaborn
        dashboard_path = "model_data/training_dashboard.png"
        visualizer = TrainingVisualizer()
        visualizer.plot_summary_dashboard(
            rewards_history=rewards_history,
            epsilon_history=epsilon_history,
            action_counts=action_counts,
            loss_history=loss_history,
            save_path=dashboard_path,
            show=False,
            title="Transformer-DQN Hybrid Recommender Training Dashboard"
        )
        visualizer.plot_reward_distribution(
            rewards_history=rewards_history,
            save_path="model_data/reward_distribution.png",
            show=False
        )

        # Calculate Comprehensive Evaluation Metrics (Accuracy, Precision, Recall, F1, NDCG, MRR)
        try:
            from evaluate import evaluate_model, plot_evaluation_dashboard
            print("\nCalculating Comprehensive Evaluation Metrics (Accuracy, Precision@K, Recall@K, F1@K, NDCG@K, MRR)...")
            eval_metrics = evaluate_model()
            plot_evaluation_dashboard(eval_metrics, save_path="model_data/evaluation_metrics.png", show=False)
            mlflow.log_metrics({
                "accuracy_top10": eval_metrics["accuracy_top10"],
                "mrr": eval_metrics["mrr"],
                "precision_at_10": eval_metrics["precision"]["K=10"],
                "recall_at_10": eval_metrics["recall"]["K=10"],
                "f1_at_10": eval_metrics["f1"]["K=10"],
                "ndcg_at_10": eval_metrics["ndcg"]["K=10"]
            })
            mlflow.log_artifact("model_data/evaluation_metrics.png")
            print("Logged evaluation metrics & dashboard plot to MLflow.")
        except Exception as e:
            print(f"Could not compute evaluation metrics: {e}")

        try:
            mlflow.log_artifact(dashboard_path)
            mlflow.log_artifact("model_data/reward_distribution.png")
            print("Logged graphical Matplotlib/Seaborn dashboard images to MLflow artifacts.")
        except Exception as e:
            print(f"Could not log artifact to MLflow: {e}")

        if args.show:
            print("\n[INFO] Automatically popping up Matplotlib & Seaborn visualization windows...")
            plt.show()

    print("\nTraining complete. Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    try:
        tflite_model = converter.convert()
        os.makedirs("model_data", exist_ok=True)
        out_path = os.path.join("model_data", "hybrid_dqn_model.tflite")
        tmp_path = os.path.join("model_data", "hybrid_dqn_model_tmp.tflite")
        with open(tmp_path, "wb") as f:
            f.write(tflite_model)
        try:
            os.replace(tmp_path, out_path)
            print(f"Successfully saved TFLite model artifact to {out_path}")
        except PermissionError:
            print(
                f"Successfully saved trained model to {tmp_path}.\n"
                f"Note: '{out_path}' is currently locked by a running process (e.g. active API server). "
                f"Stop the server if you wish to overwrite '{out_path}'."
            )
    except Exception as e:
        print(f"Failed to export TFLite model: {e}")


if __name__ == "__main__":
    main()

