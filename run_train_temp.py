

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Embedding,
    GlobalAveragePooling1D, Input
)
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')
import dagshub
import mlflow
import dagshub.auth

DAGSHUB_REPO_OWNER = "dagshub_name"
DAGSHUB_REPO_NAME = "dagshub_name"
DAGSHUB_TOKEN = "dagshub_token"

my_token = DAGSHUB_TOKEN

try:
    dagshub.auth.add_app_token(my_token)
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME , mlflow=True)
except Exception as e:
    print(f"DagsHub init skipped: {type(e).__name__}")

mlflow.set_experiment("hybrid-recommender")

class ECommerceEnv(gym.Env):
    
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ECommerceEnv, self).__init__()

        # Configuration
        self.history_length = 20
        self.vocab_size = 11  # 10 products + padding token

        # Observation: Sequence of product IDs
        self.observation_space = spaces.Box(
            low=0,
            high=self.vocab_size,
            shape=(self.history_length,),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(10)

        # Internal state (hidden from agent)
        self.history = deque(maxlen=self.history_length)
        self.state = None
        self.steps = 0
        self.max_steps = 20

    def _generate_user_behavior(self):
        """Simulate user engagement metrics"""
        purchases = np.random.randint(0, 11) / 10
        clicks = np.random.randint(0, 21) / 20
        return np.array([purchases, clicks], dtype=np.float32)

    def _generate_user_demographics(self):
        """Simulate user profile features"""
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

        # Generate hidden user profile
        self.current_behavior = self._generate_user_behavior()
        self.current_demographics = self._generate_user_demographics()

        self._update_state()
        return self.state, {}

    def _update_state(self):
        """Convert history deque to padded array for Transformer"""
        state = list(self.history)
        while len(state) < self.history_length:
            state.insert(0, 0)  # Left-pad with zeros
        self.state = np.array(state, dtype=np.int32)

    def step(self, action):
        self.steps += 1
        self.history.append(action)

        # Calculate reward from implicit feedback
        location_vector = self.current_demographics[3:]
        location_index = np.argmax(location_vector)

        click_prob = 0.1
        if action == location_index:
            click_prob = 0.4 + (0.5 * self.current_behavior[0])

        clicked = np.random.rand() < click_prob

        # Map implicit feedback to rewards
        if clicked:
            reward = self._calculate_reward("click")
            self.current_behavior[0] = min(1.0, self.current_behavior[0] + 0.1)
            self.current_behavior[1] = min(1.0, self.current_behavior[1] + 0.05)
        else:
            reward = self._calculate_reward("no_action")

        self._update_state()
        done = self.steps >= self.max_steps

        return self.state, reward, done, False, {}

    def _calculate_reward(self, event):
        
        reward_map = {
            "purchase": 2.0,
            "click": 1.0,
            "view": 0.2,
            "no_action": 0.0
        }
        return reward_map.get(event, 0.0)

class TransformerBlock(tf.keras.layers.Layer):
    """Multi-head self-attention block for sequence processing"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
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
    
    inputs = Input(shape=(history_length,), dtype=tf.int32)

    # Transformer encoder
    embedding = Embedding(vocab_size, 32)(inputs)
    x = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=64)(embedding)
    x = GlobalAveragePooling1D()(x)

    # DQN head
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(action_size, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

class OfflineRLAgent:
    

    def __init__(self, state_size, action_size, model):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model

        # Replay buffer with behavior policy probabilities
        self.memory = deque(maxlen=2000)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Discount factor
        self.gamma = 0.99

        # Training mode flag
        self.training = True

    def behavior_policy(self, state, candidates):
        
        probs = np.ones(len(candidates)) / len(candidates)
        action = np.random.choice(len(candidates), p=probs)
        return action, probs[action]

    def remember(self, state, action, reward, next_state, done, behavior_prob):
        
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'behavior_prob': behavior_prob
        })

    def act(self, state, candidates=None):
        
        if candidates is None:
            candidates = list(range(self.action_size))

        # Exploration
        if self.training and np.random.rand() <= self.epsilon:
            action = random.choice(candidates)
            behavior_prob = 1.0 / len(candidates)
            return action, behavior_prob

        # Exploitation: Candidate re-ranking
        q_values = self.model.predict(state, verbose=0)[0]
        candidate_qs = [(idx, q_values[idx]) for idx in candidates]
        action = max(candidate_qs, key=lambda x: x[1])[0]

        # Compute behavior probability for logging
        _, behavior_prob = self.behavior_policy(state, candidates)

        return action, behavior_prob

    def replay(self, batch_size):
        
        if len(self.memory) < batch_size:
            return

        # Sample minibatch
        minibatch = random.sample(self.memory, batch_size)

        # Vectorized data preparation
        states = np.array([t['state'] for t in minibatch])
        states = np.squeeze(states)

        actions = np.array([t['action'] for t in minibatch])
        rewards = np.array([t['reward'] for t in minibatch])
        next_states = np.array([t['next_state'] for t in minibatch])
        next_states = np.squeeze(next_states)
        dones = np.array([t['done'] for t in minibatch])
        behavior_probs = np.array([t['behavior_prob'] for t in minibatch])

        # Predict Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Compute IPS weights with clipping
        ips_weights = 1.0 / np.maximum(behavior_probs, 0.01)
        ips_weights = np.minimum(ips_weights, 10.0)  # Safety clipping

        # Bellman update with IPS correction
        for i in range(batch_size):
            target_val = rewards[i]
            if not dones[i]:
                target_val = rewards[i] + self.gamma * np.amax(next_q_values[i])

            # Apply IPS weight to TD error
            targets[i][actions[i]] = target_val

        # Weighted training
        sample_weights = ips_weights
        self.model.fit(
            states,
            targets,
            sample_weight=sample_weights,
            epochs=1,
            verbose=0,
            batch_size=batch_size
        )


    def log_interaction(self, state, action, reward, behavior_prob, q_value):
        
        log_entry = {
            "state_hash": hash(state.tobytes()),
            "action": int(action),
            "reward": float(reward),
            "behavior_prob": float(behavior_prob),
            "q_value": float(q_value),
            "epsilon": float(self.epsilon)
        }
        return log_entry

env = ECommerceEnv()

# !!! FIX: Explicitly cast to Python int to prevent Keras ValueError !!!
state_size = int(env.history_length)
action_size = int(env.action_space.n)

# Build model
print(f"Building model: Input={state_size}, Output={action_size}")
model = build_hybrid_model(state_size, action_size)
agent = OfflineRLAgent(state_size, action_size, model)

with mlflow.start_run(run_name="Transformer_DQN_Final"):

    params = {"episodes": 700, "batch_size": 32}
    mlflow.log_params(params)
    print("\n=== Starting Training with MLflow ===\n")

    for episode in range(params["episodes"]):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0

        for step in range(200):
            action, behavior_prob = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done, behavior_prob)

            episode_reward += reward
            state = next_state

            agent.replay(params["batch_size"])

            if done: break
            # Decay exploration
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay


        # Log metrics
        mlflow.log_metric("reward", episode_reward, step=episode)
        mlflow.log_metric("epsilon", agent.epsilon, step=episode)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

mlflow.tensorflow.log_model(agent.model, "model")


converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS 
]
    
try:
    tflite_model = converter.convert()
    with open('hybrid_dqn_model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    mlflow.log_artifact('hybrid_dqn_model.tflite')
    print("saved and uploaded to DagsHub.")
except Exception as e:
    print(f"Failed: {e}")

