'''
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
'''

"""
Transformer-DQN Hybrid E-Commerce Recommender
================================================

FIXES INCLUDED
--------------
1. Deterministic product vocabulary instead of hash-based product IDs.
2. Consistent 0-based product/action indexing.
3. Transformer MultiHeadAttention key_dim fixed.
4. Lower learning rate for stable Q-learning.
5. Hard target-network updates instead of aggressive soft updates.
6. Double DQN target calculation retained.
7. Correct PER priority initialization.
8. Gradient clipping retained.
9. Reward clipping added for numerical stability.
10. Slower epsilon decay.
11. Lower gamma for this short-horizon environment.
12. Q-value / TD-error diagnostics added.
13. NaN / Inf protection added.
14. More robust training loop.
15. Existing MLflow, dashboard and TFLite functionality retained.

NOTE
----
The environment still contains a simulated interaction mechanism because
your current dataset does not necessarily provide an explicit counterfactual
reward for every possible recommendation action.

For a true offline recommender evaluation, the environment should eventually
be replaced with logged-interaction/offline-RL evaluation.
"""

from tensorflow.python.platform import self_check
import json
import os
import random
import warnings
from collections import deque
import random
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


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# MLflow / DagsHub
# ============================================================

import mlflow

DAGSHUB_REPO_OWNER = "dagshub_name"
DAGSHUB_REPO_NAME = "dagshub_name"
DAGSHUB_TOKEN = "dagshub_token"

try:
    import dagshub
    import dagshub.auth

    dagshub.auth.add_app_token(DAGSHUB_TOKEN)

    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True,
    )

except Exception as e:
    print(f"DagsHub init skipped (using local MLflow): {e}")

mlflow.set_experiment("hybrid-recommender")


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_LENGTH = 20

EMBED_DIM = 32
NUM_HEADS = 4
FF_DIM = 64

LEARNING_RATE = 1e-4

GAMMA = 0.95

BATCH_SIZE_DEFAULT = 32

REPLAY_BUFFER_SIZE = 10000
PER_ALPHA = 0.6

WARMUP_STEPS = 500

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0
EPSILON_MIN = 0.10
EPSILON_DECAY = 0.998

REWARD_CLIP_MIN = -1.0
REWARD_CLIP_MAX = 1.0

MAX_STEPS_PER_EPISODE = 20


# ============================================================
# ENVIRONMENT
# ============================================================

class ECommerceEnv(gym.Env):
    """
    E-commerce recommendation environment.

    Important:
    ----------
    Product IDs are converted into a deterministic vocabulary.

    Old implementation:
        (abs(hash(product_id)) % 10) + 1

    This caused many different products to collide into the same
    representation.

    New implementation:
        product_id -> deterministic integer index
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset_path="Ecommerce_Dataset.csv"):

        super().__init__()

        self.history_length = HISTORY_LENGTH
        self.max_steps = MAX_STEPS_PER_EPISODE

        self.real_dataset = None
        self.user_sessions = []

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        if os.path.exists(dataset_path):

            try:

                df = pd.read_csv(dataset_path)

                required_columns = {"user_id", "product_id"}

                if required_columns.issubset(df.columns):

                    self.real_dataset = df.copy()

                    # ------------------------------------------------
                    # FIX #1:
                    # Build deterministic product vocabulary.
                    # ------------------------------------------------

                    unique_products = sorted(
                        self.real_dataset["product_id"]
                        .astype(str)
                        .unique()
                    )

                    self.product_to_idx = {
                        product_id: idx
                        for idx, product_id in enumerate(unique_products)
                    }

                    self.idx_to_product = {
                        idx: product_id
                        for product_id, idx
                        in self.product_to_idx.items()
                    }

                    self.num_products = len(unique_products)

                    # ------------------------------------------------
                    # Each action corresponds directly to a product.
                    # ------------------------------------------------

                    self.action_space = spaces.Discrete(
                        self.num_products
                    )

                    self.vocab_size = self.num_products + 1

                    # Padding token = 0
                    self.padding_idx = 0

                    # ------------------------------------------------
                    # Build user sessions
                    # ------------------------------------------------

                    if "event_type" in self.real_dataset.columns:

                        for uid, group in self.real_dataset.groupby(
                            "user_id"
                        ):

                            products = [
                                self.product_to_idx[str(product)]
                                for product in group["product_id"]
                            ]

                            events = group["event_type"].tolist()

                            if len(products) > 0:

                                self.user_sessions.append(
                                    (
                                        uid,
                                        products,
                                        events,
                                    )
                                )

                    else:

                        for uid, group in self.real_dataset.groupby(
                            "user_id"
                        ):

                            products = [
                                self.product_to_idx[str(product)]
                                for product in group["product_id"]
                            ]

                            if len(products) > 0:

                                self.user_sessions.append(
                                    (
                                        uid,
                                        products,
                                        ["unknown"] * len(products),
                                    )
                                )

                    print(
                        f"[INFO] ECommerceEnv loaded real dataset "
                        f"'{dataset_path}' "
                        f"({len(self.user_sessions)} user sessions)."
                    )

                    print(
                        f"[INFO] Unique products: {self.num_products}"
                    )

                else:

                    raise ValueError(
                        "Dataset must contain user_id and product_id columns."
                    )

            except Exception as e:

                print(
                    f"[WARNING] Could not load dataset "
                    f"'{dataset_path}': {e}"
                )

        else:

            print(
                f"[WARNING] Dataset '{dataset_path}' was not found."
            )

            # --------------------------------------------------------
            # Fallback.
            # --------------------------------------------------------

            self.num_products = 10

            self.product_to_idx = {
                str(i): i
                for i in range(self.num_products)
            }

            self.idx_to_product = {
                i: str(i)
                for i in range(self.num_products)
            }

            self.vocab_size = self.num_products + 1

            self.padding_idx = 0

            self.action_space = spaces.Discrete(
                self.num_products
            )

        # --------------------------------------------------------
        # Observation space
        # --------------------------------------------------------

        self.observation_space = spaces.Box(
            low=0,
            high=self.vocab_size - 1,
            shape=(self.history_length,),
            dtype=np.int32,
        )

        self.history = deque(
            maxlen=self.history_length
        )

        self.state = None
        self.steps = 0

        self.current_behavior = None
        self.current_demographics = None

        self.current_session_events = {}

    # ========================================================
    # USER BEHAVIOR
    # ========================================================

    def _generate_user_behavior(self):

        purchases = np.random.randint(
            0,
            11
        ) / 10.0

        clicks = np.random.randint(
            0,
            21
        ) / 20.0

        return np.array(
            [purchases, clicks],
            dtype=np.float32,
        )

    # ========================================================
    # DEMOGRAPHICS
    # ========================================================

    def _generate_user_demographics(self):

        age = np.random.rand(1)

        gender = np.zeros(2)

        gender[
            np.random.randint(0, 2)
        ] = 1

        location = np.zeros(5)

        location[
            np.random.randint(0, 5)
        ] = 1

        return np.concatenate(
            [
                age,
                gender,
                location,
            ]
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.steps = 0

        self.history.clear()

        self.current_behavior = (
            self._generate_user_behavior()
        )

        self.current_demographics = (
            self._generate_user_demographics()
        )

        self.current_session_events = {}

        # ----------------------------------------------------
        # Seed history from actual user session.
        # ----------------------------------------------------

        if self.user_sessions:

            session = random.choice(
                self.user_sessions
            )

            raw_products = session[1]
            raw_events = session[2]

            # Use a random subsequence rather than always
            # starting from the first product.
            if len(raw_products) > self.history_length:

                start = np.random.randint(
                    0,
                    len(raw_products)
                    - self.history_length
                    + 1
                )

                selected_products = raw_products[
                    start:
                    start + self.history_length
                ]

                selected_events = raw_events[
                    start:
                    start + self.history_length
                ]

            else:

                selected_products = raw_products
                selected_events = raw_events

            for product, event in zip(
                selected_products,
                selected_events,
            ):

                self.history.append(
                    int(product)
                )

                self.current_session_events[
                    int(product)
                ] = str(event).lower()

        # ----------------------------------------------------
        # If no real history is available,
        # create a small random history.
        # ----------------------------------------------------

        if len(self.history) == 0:

            for _ in range(
                min(
                    5,
                    self.history_length
                )
            ):

                self.history.append(
                    np.random.randint(
                        0,
                        self.num_products
                    )
                )

        self._update_state()

        return self.state.copy(), {}

    # ========================================================
    # STATE UPDATE
    # ========================================================

    def _update_state(self):

        state = list(self.history)

        while len(state) < self.history_length:

            state.insert(
                0,
                self.padding_idx
            )

        self.state = np.array(
            state,
            dtype=np.int32,
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        action = int(action)

        self.steps += 1

        # ----------------------------------------------------
        # Action must be a valid product index.
        # ----------------------------------------------------

        action = max(
            0,
            min(
                action,
                self.num_products - 1
            )
        )

        self.history.append(action)

        # ----------------------------------------------------
        # Determine simulated click probability.
        #
        # This section remains simulated because the dataset
        # does not tell us what would have happened if a
        # different product had been recommended.
        # ----------------------------------------------------

        click_prob = 0.05

        # -----------------------------------------------
        # Existing historical event
        # -----------------------------------------------

        historical_event = (
            self.current_session_events.get(
                action
            )
        )

        if historical_event is not None:

            if historical_event in {
                "purchase",
                "buy",
                "purchased",
            }:

                click_prob = 0.80

            elif historical_event in {
                "click",
                "clicked",
            }:

                click_prob = 0.60

            elif historical_event in {
                "view",
                "viewed",
            }:

                click_prob = 0.30

        # -----------------------------------------------
        # Behavioral preference signal
        # -----------------------------------------------

        if historical_event is None:

            behavior_bonus = (
                0.10
                * float(
                    self.current_behavior[0]
                )
            )

            click_prob += behavior_bonus

        click_prob = float(
            np.clip(
                click_prob,
                0.01,
                0.90,
            )
        )

        clicked = (
            np.random.rand()
            < click_prob
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        if clicked:

            reward = self._calculate_reward(
                "click"
            )

            # Purchase probability
            purchase_probability = (
                0.20
                + 0.30
                * float(
                    self.current_behavior[0]
                )
            )

            if (
                historical_event
                in {
                    "purchase",
                    "buy",
                    "purchased",
                }
            ):

                purchase_probability = 0.80

            if (
                np.random.rand()
                < purchase_probability
            ):

                reward += self._calculate_reward(
                    "purchase"
                )

        else:

            reward = self._calculate_reward(
                "no_action"
            )

        # ----------------------------------------------------
        # Reward clipping.
        #
        # Prevents extremely large targets from destabilizing
        # Q-learning.
        # ----------------------------------------------------

        reward = float(
            np.clip(
                reward,
                REWARD_CLIP_MIN,
                REWARD_CLIP_MAX,
            )
        )

        self._update_state()

        terminated = (
            self.steps >= self.max_steps
        )

        truncated = False

        return (
            self.state.copy(),
            reward,
            terminated,
            truncated,
            {},
        )

    # ========================================================
    # REWARD
    # ========================================================

    def _calculate_reward(self, event):

        # ----------------------------------------------------
        # FIX:
        # Reward scale is deliberately smaller and bounded.
        #
        # Purchase > click > no interaction.
        # ----------------------------------------------------

        reward_map = {
            "purchase": 1.0,
            "click": 0.30,
            "view": 0.05,
            "no_action": -0.01,
        }

        return reward_map.get(
            event,
            0.0,
        )


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        # ----------------------------------------------------
        # FIX #2:
        #
        # OLD:
        # key_dim = embed_dim
        #
        # With 4 heads and embed_dim=32:
        # 4 x 32 attention dimensions
        #
        # NEW:
        # key_dim = embed_dim // num_heads
        #
        # 4 x 8 = 32 total attention dimension.
        # ----------------------------------------------------

        self.att = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )

        self.ffn = tf.keras.Sequential(
            [
                Dense(
                    ff_dim,
                    activation="relu"
                ),

                Dense(
                    embed_dim
                ),
            ]
        )

        self.layernorm1 = LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = Dropout(rate)

        self.dropout2 = Dropout(rate)

    def call(
        self,
        inputs,
        training=False,
    ):

        attn_output = self.att(
            inputs,
            inputs,
            training=training,
        )

        attn_output = self.dropout1(
            attn_output,
            training=training,
        )

        out1 = self.layernorm1(
            inputs + attn_output
        )

        ffn_output = self.ffn(
            out1,
            training=training,
        )

        ffn_output = self.dropout2(
            ffn_output,
            training=training,
        )

        return self.layernorm2(
            out1 + ffn_output
        )


# ============================================================
# MODEL
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="Custom"
)
class DuelingCombine(tf.keras.layers.Layer):
    """
    Combines the Value and Advantage streams
    of a Dueling DQN.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        value, advantage = inputs

        # Remove the average advantage so that
        # V and A are identifiable.
        advantage_mean = tf.reduce_mean(
            advantage,
            axis=-1,
            keepdims=True
        )

        return value + (
            advantage - advantage_mean
        )

    def get_config(self):
        return super().get_config()

def build_hybrid_model(history_length, action_size, vocab_size=11):
    """
    Stable Dueling Transformer-DQN.

    Changes:
    1. Lower model capacity slightly.
    2. Uses Huber loss instead of MSE.
    3. Uses a much smaller learning rate.
    4. Uses gradient clipping.
    """

    inputs = Input(
        shape=(history_length,),
        dtype=tf.int32,
        name="state"
    )

    # Product IDs -> dense vectors
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=16,
        name="product_embedding"
    )(inputs)

    # Transformer
    x = TransformerBlock(
        embed_dim=16,
        num_heads=2,
        ff_dim=32,
        rate=0.1
    )(embedding)

    x = GlobalAveragePooling1D()(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    # -------------------------
    # VALUE STREAM
    # -------------------------

    value = Dense(
        16,
        activation="relu"
    )(x)

    value = Dense(
        1,
        name="state_value"
    )(value)

    # -------------------------
    # ADVANTAGE STREAM
    # -------------------------

    advantage = Dense(
        16,
        activation="relu"
    )(x)

    advantage = Dense(
        action_size,
        name="advantage"
    )(advantage)

    # Dueling DQN:
    #
    # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    #
    # =========================================================
# DUELING DQN OUTPUT
# =========================================================
#
# Instead of using Lambda(dueling_combine), we use a
# registered Keras Layer.
#
# This is important because the target network is created
# by cloning the online network.
# =========================================================

    outputs = DuelingCombine(
        name="q_values"
    )([value, advantage])

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    # IMPORTANT:
    # 1e-5 instead of 3e-4.
    # clipnorm prevents a huge gradient from destroying
    # the Q-network in one update.
    optimizer = Adam(
        learning_rate=1e-5,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0)
    )

    return model
    
    # .........................................................

    def dueling_combine(args):

        val, adv = args

        return (
            val
            + (
                adv
                - tf.reduce_mean(
                    adv,
                    axis=-1,
                    keepdims=True,
                )
            )
        )

    outputs = tf.keras.layers.Lambda(
        dueling_combine,
        name="q_values",
    )(
        [value, advantage]
    )

    model = Model(
        inputs=inputs,
        outputs=outputs,
    )

    # --------------------------------------------------------
    # FIX #3:
    #
    # Learning rate reduced from 3e-4 to 1e-4.
    #
    # This is important because the previous Q-values were
    # diverging aggressively.
    # --------------------------------------------------------

    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(
            delta=1.0
        ),
    )

    return model


# ============================================================
# PRIORITIZED EXPERIENCE REPLAY
# ============================================================

class ReplayBuffer:
    """
    Simple uniform experience replay buffer.

    We are intentionally NOT using Prioritized Experience Replay
    during stabilization.

    Why:
    PER can amplify very large TD-errors. Since your original
    problem was exploding loss, we first want a stable baseline.
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Return only the sampled transitions.

        IMPORTANT:
        This returns ONE object, not:
            samples, indices
        """

        sample_size = min(
            batch_size,
            len(self.buffer)
        )

        return random.sample(
            self.buffer,
            sample_size
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN AGENT
# ============================================================

class ReplayBuffer:
    """
    Standard uniform replay buffer.

    We intentionally use uniform replay first.

    Why?

    Your current PER implementation can amplify unstable
    TD errors. We want to first make the DQN mathematically
    stable before introducing prioritized sampling.
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):

        batch_size = min(
            batch_size,
            len(self.buffer)
        )

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):
        return len(self.buffer)


class OfflineRLAgent:

    def __init__(
        self,
        state_size,
        action_size,
        model
    ):

        self.state_size = state_size
        self.action_size = action_size

        # Main / online network
        self.model = model

        # =========================================================
        # TARGET NETWORK
        # =========================================================
        # IMPORTANT:
        # Clone the ONLINE model instead of creating a second model
        # with build_hybrid_model().
        #
        # This guarantees that both networks have EXACTLY the same:
        # - vocabulary size
        # - embedding dimensions
        # - Transformer dimensions
        # - number of attention heads
        # - Dense layers
        # - output/action dimensions
        #
        # Previously the online model had an embedding of:
        #     (3544, 16)
        #
        # while the newly-created target model had:
        #     (11, 16)
        #
        # Therefore set_weights() failed.
        # =========================================================

        self.target_model = tf.keras.models.clone_model(
            self.model
        )

        # Build the cloned model before assigning weights.
        dummy_input = np.zeros(
            (1, state_size),
            dtype=np.int32
        )

        self.target_model(dummy_input, training=False)

        # Make target network initially identical
        # to the online network.
        self.target_model.set_weights(
            self.model.get_weights()
        )

        # Replay buffer
        self.memory = ReplayBuffer(
            max_size=10000
        )

        # -------------------------
        # Exploration
        # -------------------------

        self.epsilon = 1.0
        self.epsilon_min = 0.05

        # Slower decay than your previous 0.996.
        #
        # Your previous epsilon reached 0.05 after roughly
        # 1000 episodes, which is too early.
        self.epsilon_decay = 0.999

        # -------------------------
        # RL parameters
        # -------------------------

        self.gamma = 0.95

        # Wait until replay buffer contains enough samples.
        self.warmup_steps = 500

        # IMPORTANT:
        #
        # We DON'T soft-update the target every step anymore.
        #
        # Instead we perform a HARD target update every
        # target_update_frequency gradient updates.
        self.target_update_frequency = 500

        self.train_steps = 0

    # ------------------------------------------------
    # Store experience
    # ------------------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        behavior_prob=None
    ):

        self.memory.add(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # ------------------------------------------------
    # Choose action
    # ------------------------------------------------

    def act(self, state):

        # Exploration
        if np.random.rand() < self.epsilon:

            action = random.randrange(
                self.action_size
            )

            behavior_prob = (
                1.0 / self.action_size
            )

            return action, behavior_prob

        # Exploitation
        q_values = self.model(
            tf.convert_to_tensor(
                state,
                dtype=tf.int32
            ),
            training=False
        ).numpy()

        q_values = q_values[0]

        action = int(
            np.argmax(q_values)
        )

        # Softmax only for logging behavior probability.
        # It does NOT affect the selected action.
        shifted = q_values - np.max(q_values)

        exp_q = np.exp(
            np.clip(
                shifted,
                -20,
                20
            )
        )

        probs = exp_q / (
            np.sum(exp_q) + 1e-8
        )

        behavior_prob = float(
            probs[action]
        )

        return action, behavior_prob

    # ------------------------------------------------
    # Target network update
    # ------------------------------------------------

    def update_target_network(self):

        """
        Hard target update.

        target <- online
        """

        self.target_model.set_weights(
            self.model.get_weights()
        )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    def replay(self, batch_size):

        # Don't train until enough experiences exist.
        if len(self.memory) < max(
            batch_size,
            self.warmup_steps
        ):
            return None

        batch = self.memory.sample(
            batch_size
        )

        states = np.asarray(
            [x[0][0] for x in batch],
            dtype=np.int32
        )

        actions = np.asarray(
            [x[1] for x in batch],
            dtype=np.int32
        )

        # ------------------------------------------------
        # IMPORTANT:
        #
        # Clip ONLY the reward used by the DQN.
        #
        # The original environment reward remains unchanged.
        # This prevents a large reward from producing a huge
        # Bellman target.
        # ------------------------------------------------

        rewards = np.asarray(
            [
                np.clip(x[2], -1.0, 1.0)
                for x in batch
            ],
            dtype=np.float32
        )

        next_states = np.asarray(
            [x[3][0] for x in batch],
            dtype=np.int32
        )

        dones = np.asarray(
            [x[4] for x in batch],
            dtype=np.float32
        )

        # ------------------------------------------------
        # Current Q values
        # ------------------------------------------------

        current_q = self.model(
            states,
            training=False
        ).numpy()

        # ------------------------------------------------
        # Double DQN
        # ------------------------------------------------

        # Online network chooses the action.
        next_q_online = self.model(
            next_states,
            training=False
        ).numpy()

        next_actions = np.argmax(
            next_q_online,
            axis=1
        )

        # Target network evaluates that action.
        next_q_target = self.target_model(
            next_states,
            training=False
        ).numpy()

        next_q = next_q_target[
            np.arange(len(batch)),
            next_actions
        ]

        # ------------------------------------------------
        # Bellman target
        # ------------------------------------------------

        targets_q = (
            rewards +
            self.gamma *
            next_q *
            (1.0 - dones)
        )

        # Prevent numerical runaway.
        targets_q = np.clip(
            targets_q,
            -10.0,
            10.0
        )

        # Update ONLY the action that was actually taken.
        target_matrix = current_q.copy()

        target_matrix[
            np.arange(len(batch)),
            actions
        ] = targets_q

        # ------------------------------------------------
        # Train online network
        # ------------------------------------------------

        loss = self.model.train_on_batch(
            states,
            target_matrix
        )

        self.train_steps += 1

        # ------------------------------------------------
        # Hard target update
        # ------------------------------------------------

        if (
            self.train_steps %
            self.target_update_frequency
            == 0
        ):

            self.update_target_network()

            print(
                f"[INFO] Target network updated "
                f"at training step {self.train_steps}"
            )

        return float(loss)

    # ========================================================
    # REPLAY
    # ========================================================

    def replay(
        self,
        batch_size,
    ):

        if (
            len(self.memory)
            < max(
                batch_size,
                self.warmup_steps,
            )
        ):

            return None

        minibatch, indices = (
            self.memory.sample(
                batch_size
            )
        )

        if not minibatch:

            return None

        # ----------------------------------------------------
        # Prepare batch
        # ----------------------------------------------------

        states = np.asarray(
            [
                x[0][0]
                for x in minibatch
            ],
            dtype=np.int32,
        )

        actions = np.asarray(
            [
                x[1]
                for x in minibatch
            ],
            dtype=np.int32,
        )

        rewards = np.asarray(
            [
                x[2]
                for x in minibatch
            ],
            dtype=np.float32,
        )

        next_states = np.asarray(
            [
                x[3][0]
                for x in minibatch
            ],
            dtype=np.int32,
        )

        dones = np.asarray(
            [
                x[4]
                for x in minibatch
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Current Q-values
        # ----------------------------------------------------

        current_q = (
            self.model.predict(
                states,
                verbose=0,
            )
        )

        # ----------------------------------------------------
        # Double DQN
        #
        # Online network:
        #     selects action
        #
        # Target network:
        #     evaluates action
        # ----------------------------------------------------

        next_q_online = (
            self.model.predict(
                next_states,
                verbose=0,
            )
        )

        next_actions = np.argmax(
            next_q_online,
            axis=1,
        )

        next_q_target = (
            self.target_model.predict(
                next_states,
                verbose=0,
            )
        )

        # ----------------------------------------------------
        # Build targets
        # ----------------------------------------------------

        targets = current_q.copy()

        td_errors = []

        for i in range(
            len(minibatch)
        ):

            old_q = float(
                current_q[i][
                    actions[i]
                ]
            )

            if dones[i]:

                target_q = float(
                    rewards[i]
                )

            else:

                target_q = float(
                    rewards[i]
                    + self.gamma
                    * next_q_target[i][
                        next_actions[i]
                    ]
                )

            # ------------------------------------------------
            # Numerical protection.
            #
            # Q-values should not become infinite.
            # ------------------------------------------------

            if not np.isfinite(
                target_q
            ):

                target_q = float(
                    rewards[i]
                )

            target_q = float(
                np.clip(
                    target_q,
                    -10.0,
                    10.0,
                )
            )

            targets[i][
                actions[i]
            ] = target_q

            td_error = (
                target_q
                - old_q
            )

            td_errors.append(
                abs(td_error)
            )

        # ----------------------------------------------------
        # Update PER priorities AFTER target calculation.
        # ----------------------------------------------------

        self.memory.update_priorities(
            indices,
            td_errors,
        )

        # ----------------------------------------------------
        # Train online network
        # ----------------------------------------------------

        loss = self.model.train_on_batch(
            states,
            targets,
        )

        if isinstance(
            loss,
            (list, tuple),
        ):

            loss = loss[0]

        loss = float(
            np.nan_to_num(
                loss,
                nan=0.0,
                posinf=1e6,
                neginf=0.0,
            )
        )

        self.train_steps += 1

        # ----------------------------------------------------
        # HARD TARGET UPDATE
        #
        # Instead of updating target after every replay.
        # ----------------------------------------------------

        if (
            self.train_steps
            % self.target_update_frequency
            == 0
        ):

            self.update_target_network()

        return {
            "loss": loss,
            "mean_q": float(
                np.mean(current_q)
            ),
            "max_abs_q": float(
                np.max(
                    np.abs(
                        current_q
                    )
                )
            ),
            "mean_td_error": float(
                np.mean(
                    td_errors
                )
            ),
            "max_td_error": float(
                np.max(
                    td_errors
                )
            ),
        }


# ============================================================
# TRAINING PLOTS
# ============================================================

def generate_training_plots(
    rewards_history,
    epsilon_history,
    action_counts,
    loss_history=None,
    save_path="model_data/training_dashboard.png",
    show=True,
):

    visualizer = TrainingVisualizer()

    visualizer.plot_summary_dashboard(
        rewards_history=rewards_history,
        epsilon_history=epsilon_history,
        action_counts=action_counts,
        loss_history=loss_history,
        save_path=save_path,
        show=False,
        title=(
            "Transformer-DQN Hybrid "
            "Recommender Training Dashboard"
        ),
    )

    visualizer.plot_reward_distribution(
        rewards_history=rewards_history,
        save_path=(
            "model_data/"
            "reward_distribution.png"
        ),
        show=False,
    )

    if show:

        plt.show()


# ============================================================
# MAIN
# ============================================================

import argparse


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train stabilized Transformer-DQN "
            "Hybrid Recommender"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
        help="Total training episodes",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Total training epochs. "
            "One epoch = one pass over user sessions."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="RL replay batch size",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Enable real-time interactive "
            "plotting during training"
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show",
        help=(
            "Disable displaying plots "
            "after training"
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        dest="show",
        default=True,
        help="Display plots after training",
    )

    args = parser.parse_args()

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = ECommerceEnv()

    state_size = int(
        env.history_length
    )

    action_size = int(
        env.action_space.n
    )

    vocab_size = int(
        env.vocab_size
    )

    num_sessions = (
        len(env.user_sessions)
        if env.user_sessions
        else 50
    )

    # ========================================================
    # TRAINING LENGTH
    # ========================================================

    if (
        args.epochs is not None
        and args.epochs > 0
    ):

        total_epochs = args.epochs

        episodes = (
            total_epochs
            * num_sessions
        )

        episodes_per_epoch = (
            num_sessions
        )

        print(
            f"[INFO] Training for "
            f"{total_epochs} Epochs "
            f"({episodes_per_epoch} sessions/epoch "
            f"= {episodes} total episodes)"
        )

    else:

        episodes = args.episodes

        total_epochs = max(
            1,
            episodes
            // num_sessions,
        )

        episodes_per_epoch = (
            num_sessions
        )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        f"[INFO] Product vocabulary: "
        f"{vocab_size}"
    )

    print(
        f"[INFO] Action space: "
        f"{action_size}"
    )

    model = build_hybrid_model(
        state_size,
        action_size,
        vocab_size
    )

    agent = OfflineRLAgent(
        state_size,
        action_size,
        model,
    )

    batch_size = args.batch_size

    # ========================================================
    # METRICS
    # ========================================================

    rewards_history = []

    epsilon_history = []

    loss_history = []

    q_history = []

    max_q_history = []

    td_history = []

    max_td_history = []

    action_counts = np.zeros(
        action_size,
        dtype=int,
    )

    live_plotter = (
        LiveTrainingPlotter(
            total_episodes=episodes,
            update_interval=10,
        )
        if args.live
        else None
    )

    # ========================================================
    # MLflow RUN
    # ========================================================

    with mlflow.start_run(
        run_name="Transformer_DQN_Stabilized_Run"
    ):

        # ----------------------------------------------------
        # Log hyperparameters.
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "episodes": episodes,
                "batch_size": batch_size,
                "epochs": total_epochs,
                "learning_rate": LEARNING_RATE,
                "gamma": GAMMA,
                "epsilon_min": EPSILON_MIN,
                "epsilon_decay": EPSILON_DECAY,
                "target_update_frequency":
                    TARGET_UPDATE_FREQUENCY,
                "warmup_steps": WARMUP_STEPS,
                "per_alpha": PER_ALPHA,
                "history_length":
                    HISTORY_LENGTH,
                "num_products":
                    env.num_products,
                "transformer_embed_dim":
                    EMBED_DIM,
                "transformer_heads":
                    NUM_HEADS,
            }
        )

        print(
            "\n"
            + "=" * 65
        )

        print(
            "STARTING STABILIZED "
            "TRANSFORMER-DQN TRAINING"
        )

        print(
            "=" * 65
            + "\n"
        )

        # ====================================================
        # EPISODES
        # ====================================================

        for episode in range(
            episodes
        ):

            state, _ = env.reset()

            state = np.reshape(
                state,
                [1, state_size],
            )

            episode_reward = 0.0

            ep_losses = []

            ep_q_values = []

            ep_max_q = []

            ep_td_errors = []

            ep_max_td = []

            # ------------------------------------------------
            # Run one episode
            # ------------------------------------------------

            for step in range(
                MAX_STEPS_PER_EPISODE
            ):

                action, behavior_prob = (
                    agent.act(state)
                )

                action_counts[
                    action
                ] += 1

                next_state, reward, done, _, _ = (
                    env.step(action)
                )

                next_state = np.reshape(
                    next_state,
                    [1, state_size],
                )

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    behavior_prob,
                )

                episode_reward += reward

                state = next_state

                # ------------------------------------------------
                # Train only after replay warmup.
                # ------------------------------------------------

                metrics = agent.replay(
                    batch_size
                )

                if metrics is not None:

                    ep_losses.append(
                        metrics["loss"]
                    )

                    ep_q_values.append(
                        metrics["mean_q"]
                    )

                    ep_max_q.append(
                        metrics["max_abs_q"]
                    )

                    ep_td_errors.append(
                        metrics[
                            "mean_td_error"
                        ]
                    )

                    ep_max_td.append(
                        metrics[
                            "max_td_error"
                        ]
                    )

                if done:

                    break

            # ====================================================
            # EPSILON DECAY
            # ====================================================

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon = max(
                    agent.epsilon_min,
                    agent.epsilon * agent.epsilon_decay
                )

            # ====================================================
            # EPISODE METRICS
            # ====================================================

            rewards_history.append(
                float(
                    episode_reward
                )
            )

            epsilon_history.append(
                float(
                    agent.epsilon
                )
            )

            avg_loss = (
                float(
                    np.mean(
                        ep_losses
                    )
                )
                if ep_losses
                else 0.0
            )

            avg_q = (
                float(
                    np.mean(
                        ep_q_values
                    )
                )
                if ep_q_values
                else 0.0
            )

            max_q = (
                float(
                    np.max(
                        ep_max_q
                    )
                )
                if ep_max_q
                else 0.0
            )

            avg_td = (
                float(
                    np.mean(
                        ep_td_errors
                    )
                )
                if ep_td_errors
                else 0.0
            )

            max_td = (
                float(
                    np.max(
                        ep_max_td
                    )
                )
                if ep_max_td
                else 0.0
            )

            loss_history.append(
                avg_loss
            )

            q_history.append(
                avg_q
            )

            max_q_history.append(
                max_q
            )

            td_history.append(
                avg_td
            )

            max_td_history.append(
                max_td
            )

            # ====================================================
            # MLFLOW
            # ====================================================

            mlflow.log_metric(
                "reward",
                episode_reward,
                step=episode,
            )

            mlflow.log_metric(
                "epsilon",
                agent.epsilon,
                step=episode,
            )

            if avg_loss > 0:

                mlflow.log_metric(
                    "loss",
                    avg_loss,
                    step=episode,
                )

                mlflow.log_metric(
                    "mean_q",
                    avg_q,
                    step=episode,
                )

                mlflow.log_metric(
                    "max_abs_q",
                    max_q,
                    step=episode,
                )

                mlflow.log_metric(
                    "mean_td_error",
                    avg_td,
                    step=episode,
                )

                mlflow.log_metric(
                    "max_td_error",
                    max_td,
                    step=episode,
                )

            # ====================================================
            # LIVE PLOT
            # ====================================================

            if live_plotter:

                live_plotter.update(
                    episode + 1,
                    episode_reward,
                    agent.epsilon,
                    action_counts,
                )

            # ====================================================
            # EPOCH SUMMARY
            # ====================================================

            if (
                (episode + 1)
                % episodes_per_epoch
                == 0
                or
                (episode + 1)
                == episodes
            ):

                epoch_number = min(
                    total_epochs,
                    (
                        (episode + 1)
                        // episodes_per_epoch
                    ),
                )

                recent_rewards = (
                    rewards_history[
                        -episodes_per_epoch:
                    ]
                )

                recent_losses = (
                    loss_history[
                        -episodes_per_epoch:
                    ]
                )

                recent_q = (
                    q_history[
                        -episodes_per_epoch:
                    ]
                )

                print(
                    f"\n"
                    f"Epoch "
                    f"{epoch_number}/"
                    f"{total_epochs}"
                    f" | Episode "
                    f"{episode + 1}/"
                    f"{episodes}"
                    f"\n"
                    f"  Avg Reward   : "
                    f"{np.mean(recent_rewards):.4f}"
                    f"\n"
                    f"  Avg Loss     : "
                    f"{np.mean(recent_losses):.6f}"
                    f"\n"
                    f"  Mean Q       : "
                    f"{np.mean(recent_q):.4f}"
                    f"\n"
                    f"  Epsilon      : "
                    f"{agent.epsilon:.4f}"
                    f"\n"
                    f"  Train Steps  : "
                    f"{agent.train_steps}"
                )

                # ------------------------------------------------
                # Warning system.
                # ------------------------------------------------

                if (
                    np.mean(recent_losses)
                    > 1000
                ):

                    print(
                        "[WARNING] Loss is becoming "
                        "very large."
                    )

                if (
                    np.max(
                        recent_q
                    )
                    > 100
                ):

                    print(
                        "[WARNING] Q-values are "
                        "growing unusually large."
                    )

        # ====================================================
        # CLOSE LIVE PLOT
        # ====================================================

        if live_plotter:

            live_plotter.close()

        # ====================================================
        # TRAINING SUMMARY
        # ====================================================

        initial_count = min(
            50,
            len(rewards_history),
        )

        final_count = min(
            50,
            len(rewards_history),
        )

        initial_avg_reward = float(
            np.mean(
                rewards_history[
                    :initial_count
                ]
            )
        )

        final_avg_reward = float(
            np.mean(
                rewards_history[
                    -final_count:
                ]
            )
        )

        max_reward = float(
            np.max(
                rewards_history
            )
        )

        avg_reward = float(
            np.mean(
                rewards_history
            )
        )

        reward_lift = float(
            final_avg_reward
            - initial_avg_reward
        )

        print(
            "\n"
            + "=" * 65
        )

        print(
            "TRAINING PERFORMANCE "
            "METRICS SUMMARY"
        )

        print(
            "=" * 65
        )

        print(
            f"Total Episodes Trained : "
            f"{episodes}"
        )

        print(
            f"Initial 50-Ep Reward   : "
            f"{initial_avg_reward:.4f}"
        )

        print(
            f"Final 50-Ep Reward     : "
            f"{final_avg_reward:.4f}"
        )

        print(
            f"Reward Improvement     : "
            f"{reward_lift:+.4f}"
        )

        print(
            f"Overall Max Reward     : "
            f"{max_reward:.4f}"
        )

        print(
            f"Overall Mean Reward    : "
            f"{avg_reward:.4f}"
        )

        print(
            f"Final Epsilon          : "
            f"{agent.epsilon:.4f}"
        )

        print(
            f"Final Mean Q           : "
            f"{q_history[-1]:.4f}"
        )

        print(
            f"Final Max |Q|          : "
            f"{max_q_history[-1]:.4f}"
        )

        print(
            "=" * 65
            + "\n"
        )

        # ====================================================
        # LOG SUMMARY
        # ====================================================

        mlflow.log_metrics(
            {
                "initial_50ep_avg_reward":
                    initial_avg_reward,

                "final_50ep_avg_reward":
                    final_avg_reward,

                "reward_lift":
                    reward_lift,

                "max_reward":
                    max_reward,

                "mean_reward":
                    avg_reward,

                "final_epsilon":
                    agent.epsilon,

                "final_mean_q":
                    q_history[-1],

                "final_max_abs_q":
                    max_q_history[-1],

                "final_mean_td_error":
                    td_history[-1],

                "final_max_td_error":
                    max_td_history[-1],
            }
        )

        # ====================================================
        # SAVE TRAINING HISTORY
        # ====================================================

        os.makedirs(
            "model_data",
            exist_ok=True,
        )

        history_json_path = (
            "model_data/"
            "training_history.json"
        )

        history_data = {

            "episodes": episodes,

            "rewards": [
                float(r)
                for r in rewards_history
            ],

            "epsilon": [
                float(e)
                for e in epsilon_history
            ],

            "losses": [
                float(l)
                for l in loss_history
            ],

            "mean_q": [
                float(q)
                for q in q_history
            ],

            "max_abs_q": [
                float(q)
                for q in max_q_history
            ],

            "mean_td_error": [
                float(x)
                for x in td_history
            ],

            "max_td_error": [
                float(x)
                for x in max_td_history
            ],

            "action_counts": [
                int(c)
                for c in action_counts
            ],

            "summary": {

                "initial_50ep_avg_reward":
                    initial_avg_reward,

                "final_50ep_avg_reward":
                    final_avg_reward,

                "reward_lift":
                    reward_lift,

                "max_reward":
                    max_reward,

                "mean_reward":
                    avg_reward,

                "final_epsilon":
                    float(
                        agent.epsilon
                    ),

                "final_mean_q":
                    float(
                        q_history[-1]
                    ),

                "final_max_abs_q":
                    float(
                        max_q_history[-1]
                    ),
            },

            "hyperparameters": {

                "learning_rate":
                    LEARNING_RATE,

                "gamma":
                    GAMMA,

                "epsilon_min":
                    EPSILON_MIN,

                "epsilon_decay":
                    EPSILON_DECAY,

                "target_update_frequency":
                    TARGET_UPDATE_FREQUENCY,

                "warmup_steps":
                    WARMUP_STEPS,

                "per_alpha":
                    PER_ALPHA,

                "history_length":
                    HISTORY_LENGTH,
            },
        }

        try:

            with open(
                history_json_path,
                "w",
            ) as f:

                json.dump(
                    history_data,
                    f,
                    indent=2,
                )

            print(
                f"Exported training metrics to "
                f"'{history_json_path}'"
            )

        except Exception as e:

            print(
                f"Failed to export JSON metrics: {e}"
            )

        # ====================================================
        # DASHBOARDS
        # ====================================================

        dashboard_path = (
            "model_data/"
            "training_dashboard.png"
        )

        visualizer = TrainingVisualizer()

        visualizer.plot_summary_dashboard(
            rewards_history=rewards_history,
            epsilon_history=epsilon_history,
            action_counts=action_counts,
            loss_history=loss_history,
            save_path=dashboard_path,
            show=False,
            title=(
                "Stabilized Transformer-DQN "
                "Hybrid Recommender"
            ),
        )

        visualizer.plot_reward_distribution(
            rewards_history=rewards_history,
            save_path=(
                "model_data/"
                "reward_distribution.png"
            ),
            show=False,
        )

        # ====================================================
        # ADDITIONAL Q-VALUE DIAGNOSTIC
        # ====================================================

        q_plot_path = (
            "model_data/"
            "q_value_stability.png"
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            q_history,
            label="Mean Q",
        )

        plt.plot(
            max_q_history,
            label="Max |Q|",
        )

        plt.axhline(
            10,
            linestyle="--",
            label="Q=10 safety reference",
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Q-value"
        )

        plt.title(
            "Q-Value Stability During Training"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            q_plot_path,
            dpi=150,
        )

        plt.close()

        # ====================================================
        # TD ERROR DIAGNOSTIC
        # ====================================================

        td_plot_path = (
            "model_data/"
            "td_error_stability.png"
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            td_history,
            label="Mean TD Error",
        )

        plt.plot(
            max_td_history,
            label="Max TD Error",
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "TD Error"
        )

        plt.title(
            "TD Error Stability During Training"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            td_plot_path,
            dpi=150,
        )

        plt.close()

        # ====================================================
        # MLflow ARTIFACTS
        # ====================================================

        try:

            mlflow.log_artifact(
                dashboard_path
            )

            mlflow.log_artifact(
                "model_data/"
                "reward_distribution.png"
            )

            mlflow.log_artifact(
                q_plot_path
            )

            mlflow.log_artifact(
                td_plot_path
            )

        except Exception as e:

            print(
                f"Could not log artifacts: {e}"
            )

        # ====================================================
        # EVALUATION
        # ====================================================

        try:

            from evaluate import (
                evaluate_model,
                plot_evaluation_dashboard,
            )

            print(
                "\nCalculating comprehensive "
                "evaluation metrics..."
            )

            eval_metrics = (
                evaluate_model()
            )

            plot_evaluation_dashboard(
                eval_metrics,
                save_path=(
                    "model_data/"
                    "evaluation_metrics.png"
                ),
                show=False,
            )

            mlflow.log_metrics(
                {
                    "accuracy_top10":
                        eval_metrics[
                            "accuracy_top10"
                        ],

                    "mrr":
                        eval_metrics[
                            "mrr"
                        ],

                    "precision_at_10":
                        eval_metrics[
                            "precision"
                        ]["K=10"],

                    "recall_at_10":
                        eval_metrics[
                            "recall"
                        ]["K=10"],

                    "f1_at_10":
                        eval_metrics[
                            "f1"
                        ]["K=10"],

                    "ndcg_at_10":
                        eval_metrics[
                            "ndcg"
                        ]["K=10"],
                }
            )

            mlflow.log_artifact(
                "model_data/"
                "evaluation_metrics.png"
            )

            print(
                "Evaluation metrics completed."
            )

        except Exception as e:

            print(
                f"Could not compute evaluation metrics: {e}"
            )

        # ====================================================
        # OPTIONAL DISPLAY
        # ====================================================

        if args.show:

            print(
                "\n[INFO] Displaying "
                "training visualizations..."
            )

            plt.show()

    # ========================================================
    # TFLITE EXPORT
    # ========================================================

    print(
        "\nTraining complete. "
        "Converting model to TFLite..."
    )

    try:

        converter = (
            tf.lite.TFLiteConverter
            .from_keras_model(
                agent.model
            )
        )

        converter.target_spec.supported_ops = [

            tf.lite.OpsSet.TFLITE_BUILTINS,

            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        tflite_model = (
            converter.convert()
        )

        os.makedirs(
            "model_data",
            exist_ok=True,
        )

        out_path = os.path.join(
            "model_data",
            "hybrid_dqn_model.tflite",
        )

        tmp_path = os.path.join(
            "model_data",
            "hybrid_dqn_model_tmp.tflite",
        )

        with open(
            tmp_path,
            "wb",
        ) as f:

            f.write(
                tflite_model
            )

        try:

            os.replace(
                tmp_path,
                out_path,
            )

            print(
                f"Successfully saved TFLite "
                f"model to {out_path}"
            )

        except PermissionError:

            print(
                f"Successfully saved trained "
                f"model to {tmp_path}.\n"
                f"'{out_path}' is currently locked."
            )

    except Exception as e:

        print(
            f"Failed to export TFLite model: {e}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()