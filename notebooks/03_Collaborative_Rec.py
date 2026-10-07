# Converted from 03_Collaborative_Rec.ipynb



# --- Cell ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Cornac Imports
import cornac
from cornac.models import VAECF
from cornac.eval_methods import RatioSplit
from cornac.data import Dataset
from cornac.metrics import Recall, NDCG
import torch

warnings.simplefilter("ignore")

# GPU Check for Cornac
print(f"PyTorch Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# --- Cell ---

# Load the Master Dataset
data = pd.read_csv('Master_Ecommerce_Dataset.csv')

print(f"Data Shape: {data.shape}")
display(data.head())
data = pd.concat([data1, data2], ignore_index=True)

# --- Cell ---



# Map weights before encoding
event_mapping = {
    "view": 1.0,
    "purchase": 3.0
}
data["event_weight"] = data["event_type"].map(event_mapping).fillna(0.0)
data["event_weight"] = data["event_weight"] / data["event_weight"].max()

# Now do label encoding

# --- Cell ---

data['event_type'].head().unique()

# --- Cell ---

event_encode = LabelEncoder()
data['event_type'] = event_encode.fit_transform(data['event_type'])
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
data['user'] = user_encoder.fit_transform(data['user_id'])
data['item'] = item_encoder.fit_transform(data['product_id'])

# --- Cell ---

interactions = list(zip(data['user'], data['item'], data['event_weight']))



# Split train/test
eval_method = RatioSplit(
    data=interactions,
    test_size=0.2,
    rating_threshold=0.5,  # this part is for implicit feedback
    exclude_unknowns=True,
    verbose=True
)


# --- Cell ---

# VAE-CF model
model = VAECF(
    k=64,  # Latent factors
    autoencoder_structure=[200],
    n_epochs=10,
    batch_size=512,
    learning_rate=1e-4,
    use_gpu=True,  # set to false if cpu is used 
    verbose=True
)



# Train
history= model.fit(eval_method.train_set)
model.save("collab_recommender.keras")

# --- Cell ---

def recommend_for_users(model, eval_method, user_ids, top_n=10, fallback_items=None):
    train_set = eval_method.train_set
    uid_map = train_set.uid_map
    iid_map = train_set.iid_map
    iid_map_inv = {v: k for k, v in iid_map.items()}

    results = {}

    for user_id_str in user_ids:
        print(f"\n User: {user_id_str}")

        if user_id_str not in uid_map:
            print("Cold-start user (not in training data).")

            if fallback_items:
                fallback_recs = fallback_items[:top_n]
                print(f"Fallback recommendations: {fallback_recs}")
                results[user_id_str] = [(item_id, None) for item_id in fallback_recs]
            else:
                print("No fallback available.")
                results[user_id_str] = []
            continue

        user_id = uid_map[user_id_str]
        n_items = train_set.num_items

        scores = [model.score(user_id, item_id) for item_id in range(n_items)]

        user_interactions = train_set.matrix[user_id]
        seen_items = set(user_interactions.indices)

        unseen_scores = [(item_id, score) for item_id, score in enumerate(scores) if item_id not in seen_items]
        top_items = sorted(unseen_scores, key=lambda x: x[1], reverse=True)[:top_n]
        rec_items = [(iid_map_inv[item_id], round(score, 4)) for item_id, score in top_items]

        for item_id, score in rec_items:
            print(f"🛍️ Item ID: {item_id} | Score: {score}")

        results[user_id_str] = rec_items

    return results







# --- Cell ---

user_list = ['555475417','1005115']

# --- Cell ---

recommendations = recommend_for_users(
    model=model, 
    eval_method=eval_method, 
    user_ids=['555475417'], 
    top_n=5,
   
)