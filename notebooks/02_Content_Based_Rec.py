# Converted from 02_Content_Based_Rec.ipynb

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# GPU Check
from tensorflow.python.client import device_lib
print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Set Seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- Cell ---

import warnings
warnings.simplefilter("ignore")

# --- Cell ---

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print ("No of GPUS present:",len(tf.config.list_physical_devices('GPU')))

# --- Cell ---

# Load the Clean Data
data = pd.read_csv("Master_Ecommerce_Dataset.csv")

# Fill missing values for text generation
data['brand'] = data['brand'].fillna('unknown')
data['category_code'] = data['category_code'].fillna('unknown')

# Normalize Price for Embedding
scaler = MinMaxScaler()
data['price_norm'] = scaler.fit_transform(data[['price']])


data['text'] = (
    data['brand'] + " " + 
    data['category_code'] + " price " + 
    data['price_norm'].astype(str)
)


# --- Cell ---




# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts, batch_size=64):
    all_embeddings = []
    
    # Process in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[i:i+batch_size]]
        
        inputs = tokenizer(
            batch, 
            padding="max_length", 
            truncation=True, 
            max_length=32, 
            return_tensors="tf"
        )
        
        outputs = bert_model(**inputs)
        # Extract CLS token (contextual representation)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.numpy())
        
        if i % (batch_size * 10) == 0:
            print(f"Processed {i}/{len(texts)}", end='\r')
            
    return np.concatenate(all_embeddings, axis=0)

# Generate Embeddings
print("Generating Product Embeddings...")
unique_products = data[['product_id', 'text', 'brand', 'category_code', 'price']].drop_duplicates('product_id')
product_embeddings = get_bert_embeddings(unique_products['text'].tolist())

# Map Product ID to Embedding
product_id_to_embedding = {
    pid: emb for pid, emb in zip(unique_products['product_id'], product_embeddings)
}

# --- Cell ---

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# --- Cell ---

bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# --- Cell ---



def get_bert_embeddings(texts, tokenizer, model, batch_size=64, save_path="embeddings_chunk.npy"):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [str(x) for x in batch] 

        inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=32, return_tensors="tf")
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        np_embeddings = cls_embeddings.numpy()

        all_embeddings.append(np_embeddings)

        # Save chunk periodically
        if i % (batch_size * 10) == 0:
            print(f"Processed {i}/{len(texts)}")
            np.save(save_path, np.concatenate(all_embeddings, axis=0))

    return tf.convert_to_tensor(np.concatenate(all_embeddings, axis=0))

# --- Cell ---

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts, batch_size=64):
    all_embeddings = []
    
    # Process in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[i:i+batch_size]]
        
        inputs = tokenizer(
            batch, 
            padding="max_length", 
            truncation=True, 
            max_length=32, 
            return_tensors="tf"
        )
        
        outputs = bert_model(**inputs)
        # Extract CLS token (contextual representation)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.numpy())
        
        if i % (batch_size * 10) == 0:
            print(f"Processed {i}/{len(texts)}", end='\r')
            
    return np.concatenate(all_embeddings, axis=0)


unique_products = data[['product_id', 'text', 'brand', 'category_code', 'price']].drop_duplicates('product_id')
product_embeddings = get_bert_embeddings(unique_products['text'].tolist())

# Map Product ID to Embedding
product_id_to_embedding = {
    pid: emb for pid, emb in zip(unique_products['product_id'], product_embeddings)
}

# --- Cell ---

from sklearn.metrics.pairwise import cosine_similarity

def recommend_for_user(user_id, top_k):
    if user_id not in user_id_to_profile:
        print("No interaction data for this user.")
        return []

    user_vec = user_id_to_profile[user_id].reshape(1, -1)
    all_item_vecs = embeddings.numpy()

    # Compute cosine similarity
    scores = cosine_similarity(user_vec, all_item_vecs).flatten()

    # Top-K item indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Return product_ids
    return data.iloc[top_indices][['product_id', 'brand', 'category_code', 'price', 'text']]

# --- Cell ---

recommend_for_user(513103710,top_k=5)

# --- Cell ---

def get_brand_embeddings(texts, tokenizer, model, batch_size=64, save_path="brand_embeddings_chunk.npy"):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [str(x) for x in batch] 

        inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=32, return_tensors="tf")
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        np_embeddings = cls_embeddings.numpy()

        all_embeddings.append(np_embeddings)

        # Save chunk periodically
        if i % (batch_size * 10) == 0:
            print(f"Processed {i}/{len(texts)}")
            np.save(save_path, np.concatenate(all_embeddings, axis=0))

    return tf.convert_to_tensor(np.concatenate(all_embeddings, axis=0))

# --- Cell ---

brand_embeddings = get_brand_embeddings(list(data1['brand_text']), tokenizer, bert_model)

# --- Cell ---

def recommend_similar_brand_price_bert(
    product_id=None,
    brand=None,
    top_k= None,
    same_category=True,
    min_price=None,
    max_price=None
):
    if product_id is not None:
        if product_id not in data1['product_id'].values:
            print("Product not found.")
            return []

        idx = data1.index.get_loc(data1.index[data1['product_id'] == product_id][0])
    
    elif brand is not None:
        brand_matches = data1[data1['brand'].str.lower() == brand.lower()]
        if brand_matches.empty:
            print("Brand not found.")
            return []

        rep_row = brand_matches.iloc[0]
        idx = data1.index.get_loc(rep_row.name)
        product_id = rep_row['product_id']
    
    else:
        print("You must provide either a product_id or a brand.")
        return []

    # Get product info
    product = data1.iloc[idx]
    product_vec = brand_embeddings[idx].numpy().reshape(1, -1)

    # Filter the dataset
    filtered_data = data1.copy()
    if same_category:
        filtered_data = filtered_data[filtered_data['category_code'] == product['category_code']]

    # price range filter
    if min_price is not None:
        filtered_data = filtered_data[filtered_data['price'] >= min_price]
    if max_price is not None:
        filtered_data = filtered_data[filtered_data['price'] <= max_price]

 
    filtered_data = filtered_data[filtered_data['product_id'] != product_id]

    # Get filtered indices
    filtered_indices = [data1.index.get_loc(i) for i in filtered_data.index]
    if not filtered_indices:
        print("No similar products found after filtering.")
        return []

    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity
    filtered_embeddings = tf.gather(brand_embeddings, filtered_indices)
    similarities = cosine_similarity(product_vec, filtered_embeddings.numpy()).flatten()

    # Get top K
    top_indices = similarities.argsort()[::-1][:top_k]
    top_data_indices = [filtered_data.index[i] for i in top_indices]

    return data1.loc[top_data_indices][['product_id', 'brand', 'category_code', 'price']]




# --- Cell ---

recommend_similar_brand_price_bert( brand = "wincars", top_k=10, max_price= 13) 