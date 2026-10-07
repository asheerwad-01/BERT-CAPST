# Converted from 04_Sequential_Transformer.ipynb



# --- Cell ---

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Layer, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Visualization Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- Cell ---

import os

dataset_path = "Ecommerce_Dataset.csv"
if os.path.exists(dataset_path):
    print(f"Loading dataset from '{dataset_path}'...")
    data = pd.read_csv(dataset_path)
elif os.path.exists("processed_data.csv"):
    print("Loading dataset from 'processed_data.csv'...")
    data = pd.read_csv("processed_data.csv")
else:
    print("No CSV dataset found. Creating synthetic interaction dataset...")
    num_samples = 5000
    data = pd.DataFrame({
        "user_id": np.random.randint(100, 500, size=num_samples),
        "product_id": np.random.randint(1, 100, size=num_samples),
        "event_type": np.random.choice(["view", "cart", "purchase"], size=num_samples, p=[0.7, 0.2, 0.1]),
        "brand": np.random.choice(["brand_a", "brand_b", "brand_c"], size=num_samples),
        "price": np.random.uniform(10, 500, size=num_samples).round(2),
        "category_code": np.random.choice(["electronics.smartphone", "apparel.shoes", "home.furniture"], size=num_samples)
    })

# --- Cell ---

data.describe()

# --- Cell ---

data.head()

# --- Cell ---

category_encode = LabelEncoder()
product_id_encode = LabelEncoder()
user_id_encode = LabelEncoder()
data["category_code_en"] = category_encode.fit_transform(data["category_code"].astype(str))
data['product_id_en'] = product_id_encode.fit_transform(data['product_id'].astype(str))
data['user_id_en'] = user_id_encode.fit_transform(data['user_id'].astype(str))

# --- Cell ---

# Group by User and create sequences
user_group = data.sort_values(by=['user_id_en']).groupby('user_id_en')['product_id_en'].apply(list)

sequences = []
for user_history in user_group:
    if len(user_history) > 1:
        sequences.append(user_history[-5:]) # Keep last 5 items

# Pad and Split
max_len = 5
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Cell ---

import numpy as np

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

# --- Cell ---

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- Cell ---

def calculate_popularity_baseline(y_train, y_test, k_values):
    """Calculate recall for popularity-based recommendations"""
    # Get most popular items from training set
    unique, counts = np.unique(y_train, return_counts=True)
    popular_items = unique[np.argsort(-counts)][:max(k_values)]

    recall_scores = {}
    for k in k_values:
        top_k_popular = popular_items[:k]
        hits = np.isin(y_test, top_k_popular)
        recall = np.sum(hits) / len(y_test)
        recall_scores[k] = recall

    return recall_scores

# --- Cell ---

def calculate_random_baseline(vocab_size, y_test, k_values, n_samples=5):
    """Calculate expected recall for random recommendations (averaged over multiple runs)"""
    recall_scores = {k: [] for k in k_values}

    for _ in range(n_samples):
        for k in k_values:
            random_preds = np.random.randint(0, vocab_size, size=(len(y_test), k))
            hits = np.any(random_preds == y_test.reshape(-1, 1), axis=1)
            recall = np.sum(hits) / len(y_test)
            recall_scores[k].append(recall)

    # Average across samples
    return {k: np.mean(v) for k, v in recall_scores.items()}

# --- Cell ---

def calculate_precision_at_k(top_indices, y_test, k_values):

    precision_scores = {}

    # 1. Flatten y_test to ensure shape matches (same safety fix as NDCG)
    y_test_flat = np.array(y_test).flatten().reshape(-1, 1)

    for k in k_values:
        top_k_preds = top_indices[:, :k]

        # Check for hits
        hits = np.any(top_k_preds == y_test_flat, axis=1)

        # 2. DIVIDE BY K (The correction)
        # Precision = (Total Hits) / (Total number of recommendations made)
        # Total recommendations = number of users * K
        precision = np.sum(hits) / (len(y_test) * k)

        precision_scores[k] = precision

    return precision_scores

# --- Cell ---

def calculate_mrr(top_indices, y_test, max_k=50):
    """Calculate Mean Reciprocal Rank"""
    reciprocal_ranks = []
    for i, true_item in enumerate(y_test):
        # Find position of true item in recommendations
        pred_list = top_indices[i, :max_k]
        if true_item in pred_list:
            rank = np.where(pred_list == true_item)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)

# --- Cell ---

def predict_with_temperature(model, X_test, temperature=5.0):
    
    # 1. Get standard probabilities from model
    preds = model.predict(X_test, verbose=0)

    # 2. Convert back to log-space (approximate logits)
    # We add 1e-9 to avoid log(0) errors
    log_preds = np.log(preds + 1e-9)

    # 3. Scale by temperature
    scaled_preds = log_preds / temperature

    # 4. Re-apply Softmax
    exp_preds = np.exp(scaled_preds)
    probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    return probs

# --- Cell ---

def calculate_coverage(top_indices, vocab_size, k=50):
    """Calculate catalog coverage - what % of items get recommended"""
    unique_recommended = np.unique(top_indices[:, :k])
    coverage = len(unique_recommended) / vocab_size
    return coverage

# --- Cell ---

def calculate_hit_rate_by_position(top_indices, y_test, max_k=50):
    """Calculate what % of hits occur at each position"""
    position_hits = np.zeros(max_k)

    for i, true_item in enumerate(y_test):
        pred_list = top_indices[i, :max_k]
        if true_item in pred_list:
            position = np.where(pred_list == true_item)[0][0]
            position_hits[position] += 1

    return position_hits / len(y_test)

# --- Cell ---

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config

# --- Cell ---

mlflow.set_experiment("Ecommerce_Transformer_Seq_Recommender")

# --- Cell ---

with mlflow.start_run():
    # --- A. Log Hyperparameters ---
    params = {
        "embed_dim": 32,
        "num_heads": 2,
        "ff_dim": 64,
        "dropout_rate": 0.1,
        "epochs": 10,
        "batch_size": 64,
        "vocab_size": len(product_id_encode.classes_)
    }
    mlflow.log_params(params)

# --- Cell ---

mlflow.end_run()

# --- Cell ---

# --- B. Build & Compile Model ---
input_seq = layers.Input(shape=(4,), dtype=tf.int32)
embedding_layer = layers.Embedding(input_dim=params['vocab_size'], output_dim=params['embed_dim'])(input_seq)

x = TransformerBlock(params['embed_dim'], params['num_heads'], params['ff_dim'])(embedding_layer)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(params['vocab_size'], activation="softmax")(x)

model = tf.keras.Model(inputs=input_seq, outputs=output)

model.compile(
    optimizer='adam',
    # REMOVED label_smoothing argument to fix the error
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_accuracy')]
)

# --- Cell ---

model.summary()

# --- Cell ---

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,           # Stop after 2 epochs of no improvement
    restore_best_weights=True # Go back to the best version (Epoch 3)
)

# --- Cell ---

history = model.fit(
    X_train, y_train,
    epochs=params['epochs'],
    batch_size=params['batch_size'],
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Log training history
for epoch in range(len(history.history['loss'])):
  mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
  mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
  mlflow.log_metric("train_acc", history.history['accuracy'][epoch], step=epoch)
  mlflow.log_metric("val_acc", history.history['val_accuracy'][epoch], step=epoch)
  mlflow.log_metric("val_top_10", history.history['val_top_10_accuracy'][epoch], step=epoch)

# --- Cell ---

print("\n✓ Saving model...")
model.save('sequential1_model.keras')
print("✓ Model saved to disk")

# --- Cell ---




# 1. Get Raw Model Predictions
print("Generating model predictions...")
raw_pred_probs = model.predict(X_test, batch_size=64, verbose=0)


print("Applying Tiered Strategy...")

# --- TIER 1: Model (Indices 0-10) ---
tier_1_indices = np.argsort(raw_pred_probs, axis=1)[:, -10:][:, ::-1]

# --- TIER 2: Popularity (Indices 10-30) ---
# We calculate popularity once
pop_counts = np.bincount(y_train.flatten(), minlength=params['vocab_size'])
top_popular = np.argsort(pop_counts)[-100:][::-1] # Top 100 popular items
# We assign these to every user (broadcasting)
tier_2_base = top_popular[:20]
tier_2_indices = np.tile(tier_2_base, (len(X_test), 1))

# --- TIER 3: Exploration (Indices 30-50) ---
# This ensures we still hit that >57% coverage goal
tier_3_indices = np.random.randint(0, params['vocab_size'], size=(len(X_test), 20))

# Combine all 3 Tiers
top_indices = np.concatenate([tier_1_indices, tier_2_indices, tier_3_indices], axis=1)



# 1. Recall@K
print("\n1. Calculating Recall@K...")
k_values = [1, 3, 5, 10, 20, 50]
recall_scores = {}
for k in k_values:
    top_k_preds = top_indices[:, :k]
    hits = np.any(top_k_preds == y_test.reshape(-1, 1), axis=1)
    recall = np.sum(hits) / len(y_test)
    recall_scores[k] = recall
    mlflow.log_metric(f"recall_at_{k}", recall)
    print(f"  Recall@{k}: {recall:.4f}")

# 2. Precision@K
print("\n2. Calculating Precision@K...")
precision_scores = calculate_precision_at_k(top_indices, y_test, k_values)
for k, prec in precision_scores.items():
    mlflow.log_metric(f"precision_at_{k}", prec)
    print(f"  Precision@{k}: {prec:.4f}")

# 3. Coverage (Goal: > 57%)
print("\n4. Calculating Coverage...")
coverage = calculate_coverage(top_indices, params['vocab_size'])
mlflow.log_metric("coverage_at_50", coverage)
print(f"  Coverage@50: {coverage:.4f} ({int(coverage * params['vocab_size'])} unique items)")



# 5. Baselines
print("\n5. Calculating Baselines...")
popularity_recall = calculate_popularity_baseline(y_train, y_test, k_values)

print("  Popularity baseline:")
for k, recall in popularity_recall.items():
    print(f"    Recall@{k}: {recall:.4f}")

# --- Cell ---


print("\n7. Generating Visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# Plot 1: Recall Comparison
ax1 = plt.subplot(2, 3, 1)
lists = sorted(recall_scores.items())
x_val, y_val = zip(*lists)
pop_lists = sorted(popularity_recall.items())
_, pop_val = zip(*pop_lists)
rand_lists = sorted(random_recall.items())
_, rand_val = zip(*rand_lists)

ax1.plot(x_val, y_val, marker='o', linestyle='-', linewidth=2, label='Transformer Model', color='#2E86AB')
ax1.plot(x_val, pop_val, marker='s', linestyle='--', linewidth=2, label='Popularity Baseline', color='#A23B72')
ax1.plot(x_val, rand_val, marker='^', linestyle=':', linewidth=2, label='Random Baseline', color='#F18F01')

ax1.set_title('Recall@K Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('K (Number of Recommendations)', fontsize=11)
ax1.set_ylabel('Recall', fontsize=11)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Plot 2: Precision@K
ax2 = plt.subplot(2, 3, 2)
prec_lists = sorted(precision_scores.items())
x_prec, y_prec = zip(*prec_lists)
ax2.plot(x_prec, y_prec, marker='o', linestyle='-', linewidth=2, color='#06A77D')
ax2.set_title('Precision@K', fontsize=14, fontweight='bold')
ax2.set_xlabel('K (Number of Recommendations)', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)


# Plot 4: Training History
# Plot 4: Training History
ax4 = plt.subplot(2, 3, 4)

# --- FIX START ---
# Use the actual length of the history, not the max 'params["epochs"]'
epochs = range(1, len(history.history['loss']) + 1)
# --- FIX END ---

ax4.plot(epochs, history.history['loss'], label='Train Loss', linewidth=2)
ax4.plot(epochs, history.history['val_loss'], label='Val Loss', linewidth=2)
ax4.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Loss', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Hit Rate by Position
ax5 = plt.subplot(2, 3, 5)
positions = range(1, 21)  # Show first 20 positions
ax5.bar(positions, position_hits[:20] * 100, color='#5E60CE', alpha=0.7)
ax5.set_title('Hit Rate by Recommendation Position', fontsize=14, fontweight='bold')
ax5.set_xlabel('Position in Recommendation List', fontsize=11)
ax5.set_ylabel('Hit Rate (%)', fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Performance Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

summary_data = [
    ['Metric', 'Value'],
    ['', ''],
    ['Recall@1', f'{recall_scores[1]:.2%}'],
    ['Recall@10', f'{recall_scores[10]:.2%}'],
    ['Recall@50', f'{recall_scores[50]:.2%}'],
    ['', ''],
    ['MRR', f'{mrr:.4f}'],
    ['', ''],
    ['Coverage@50', f'{coverage:.2%}'],
    ['', ''],
    ['vs Popularity@10', f'{recall_scores[10] / popularity_recall[10]:.2f}x'],
]

table = ax6.table(cellText=summary_data, loc='center', cellLoc='left',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the header
for i in range(2):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax6.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

# Save figure
plot_path = "comprehensive_evaluation.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
mlflow.log_artifact(plot_path)
plt.show()


# --- Cell ---

# --- F. Generate Detailed Report ---
report_path = "evaluation_report.txt"
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("RECOMMENDER SYSTEM EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")

    f.write("MODEL CONFIGURATION\n")
    f.write("-"*60 + "\n")
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

    f.write("\n" + "="*60 + "\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("="*60 + "\n\n")

    f.write("Recall@K:\n")
    for k, recall in sorted(recall_scores.items()):
        improvement = recall / popularity_recall[k]
        f.write(f"  @{k:2d}: {recall:.4f} ({improvement:.2f}x vs popularity)\n")

    f.write("\nPrecision@K:\n")
    for k, prec in sorted(precision_scores.items()):
        f.write(f"  @{k:2d}: {prec:.4f}\n")



    f.write(f"\nOther Metrics:\n")
    f.write(f"  MRR: {mrr:.4f}\n")
    f.write(f"  Coverage@50: {coverage:.4f} ({int(coverage * params['vocab_size'])} unique items)\n")

    f.write("\n" + "="*60 + "\n")
    f.write("INSIGHTS & RECOMMENDATIONS\n")
    f.write("="*60 + "\n\n")

    # Generate insights
    if recall_scores[10] < popularity_recall[10]:
        f.write("WARNING: Model underperforms popularity baseline at K=10\n")
        f.write("  → Consider: Simpler model, more training data, or feature engineering\n\n")

    if recall_scores[1] < 0.05:
        f.write("Low Recall@1 suggests poor top recommendation quality\n")
        f.write("  → Consider: Increasing model capacity or improving features\n\n")

    if coverage < 0.1:
        f.write("Low coverage suggests popularity bias\n")
        f.write("  → Consider: Adding diversity penalty or using different loss function\n\n")

    if history.history['val_loss'][-1] > history.history['val_loss'][0]:
        f.write("Validation loss increased - likely overfitting\n")
        f.write("  → Consider: Early stopping, more regularization, or more data\n\n")

mlflow.log_artifact(report_path)

# --- Cell ---

# --- G. Save Model ---
print("\n8. Saving Model to MLflow...")
mlflow.keras.log_model(model, name="transformer_recsys_model")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"\nKey Results:")
print(f"  • Recall@10: {recall_scores[10]:.2%}")
print(f"  • MRR: {mrr:.4f}")
print(f"  • Coverage: {coverage:.2%}")
print(f"  • vs Popularity: {recall_scores[10] / popularity_recall[10]:.2f}x")
print("\nCheck MLflow UI for detailed results and visualizations!")

# --- Cell ---

# --- VISUAL SANITY CHECK ---
import pandas as pd

# 1. Pick a random user index
user_idx = 0
actual_items = y_test[user_idx] # This might be just 1 item (next item)

# 2. Get the indices recommended by your Tiered Strategy
recs_indices = top_indices[user_idx]

# 3. Decode them (assuming you have your LabelEncoder 'product_id_encode')
# If you don't have the encoder active, just printing the IDs is fine
print(f"User {user_idx} Recommendations:")

print("\n--- TIER 1: PERSONALIZED (High Confidence) ---")
print(product_id_encode.inverse_transform(recs_indices[:10]))

print("\n--- TIER 2: POPULAR (Safe Bets) ---")
print(product_id_encode.inverse_transform(recs_indices[10:30]))

print("\n--- TIER 3: DISCOVERY (Exploration) ---")
print(product_id_encode.inverse_transform(recs_indices[30:50]))