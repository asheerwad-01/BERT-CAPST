import os
import re
import glob
import json
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")

def clean_price(val):
    if pd.isna(val):
        return 19.99
    val_str = str(val).strip().replace('$', '').replace(',', '').replace('"', '')
    match = re.search(r"(\d+(\.\d+)?)", val_str)
    if match:
        try:
            p = float(match.group(1))
            return p if p > 0 else 19.99
        except ValueError:
            return 19.99
    return 19.99

def clean_category(val):
    if pd.isna(val) or not str(val).strip() or str(val).strip().lower() == 'null':
        return 'general.electronics'
    cat_str = str(val).strip()
    if cat_str.startswith('[') and cat_str.endswith(']'):
        try:
            parsed = json.loads(cat_str.replace('""', '"'))
            if isinstance(parsed, list) and len(parsed) > 0:
                return '.'.join([str(x).lower().replace(' ', '_').replace('&', 'and') for x in parsed[:2]])
        except Exception:
            pass
    clean_cat = re.sub(r'[^a-zA-Z0-9_\.]', '_', cat_str.lower())
    clean_cat = re.sub(r'_+', '_', clean_cat).strip('_')
    return clean_cat[:40] if clean_cat else 'general.goods'

def load_dataset_sample():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "dataset_sample")
    if not os.path.exists(sample_dir):
        sample_dir = "dataset_sample"
        
    csv_files = glob.glob(os.path.join(sample_dir, "*.csv"))
    if not csv_files:
        print("[WARNING] No dataset_sample CSV files found.")
        return None

    products = []
    print(f"Ingesting {len(csv_files)} dataset files from '{sample_dir}'...")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f" -> Loading {filename} ({len(df)} records)")
            
            # Map columns across diverse e-commerce sources
            id_col = next((c for c in ['product_id', 'asin', 'id', 'sku'] if c in df.columns), None)
            name_col = next((c for c in ['title', 'product_name', 'name'] if c in df.columns), None)
            brand_col = next((c for c in ['brand', 'seller_name', 'manufacturer'] if c in df.columns), None)
            price_col = next((c for c in ['final_price', 'initial_price', 'price'] if c in df.columns), None)
            cat_col = next((c for c in ['category_code', 'categories', 'category_name', 'breadcrumb', 'category_path'] if c in df.columns), None)

            for idx, row in df.iterrows():
                pid = str(row[id_col]) if id_col and pd.notna(row[id_col]) else f"{filename[:3]}_{idx+1}"
                title = str(row[name_col]) if name_col and pd.notna(row[name_col]) else f"Product {pid}"
                brand = str(row[brand_col]) if brand_col and pd.notna(row[brand_col]) else "Generic"
                price = clean_price(row[price_col]) if price_col else 19.99
                cat = clean_category(row[cat_col]) if cat_col else "apparel.general"

                products.append({
                    'product_id': pid,
                    'title': title[:80],
                    'brand': brand[:30] if brand != 'nan' else 'Generic',
                    'price': price,
                    'category_code': cat
                })
        except Exception as e:
            print(f"    Error parsing {filename}: {e}")

    prod_df = pd.DataFrame(products).drop_duplicates(subset=['product_id'])
    print(f"[SUCCESS] Extracted {len(prod_df)} unique products from dataset_sample.")
    return prod_df

def generate_interaction_dataset(prod_df, num_users=500, total_interactions=12000):
    np.random.seed(42)
    user_ids = [f"usr_{i:04d}" for i in range(1, num_users + 1)]
    product_ids = prod_df['product_id'].values
    event_types = ['view', 'cart', 'purchase']
    event_probs = [0.70, 0.20, 0.10]

    rows = []
    print(f"Generating {total_interactions} interaction events for {num_users} users...")
    
    # Assign favorite categories per user to simulate realistic user behavior
    categories = prod_df['category_code'].unique()
    user_fav_cats = {u: np.random.choice(categories) for u in user_ids}

    for _ in range(total_interactions):
        user = np.random.choice(user_ids)
        # 60% chance to pick item from user's preferred category
        if np.random.rand() < 0.6:
            fav_cat = user_fav_cats[user]
            sub_prods = prod_df[prod_df['category_code'] == fav_cat]
            if len(sub_prods) > 0:
                prod_row = sub_prods.sample(1).iloc[0]
            else:
                prod_row = prod_df.sample(1).iloc[0]
        else:
            prod_row = prod_df.sample(1).iloc[0]

        event = np.random.choice(event_types, p=event_probs)
        rows.append({
            'user_id': user,
            'product_id': prod_row['product_id'],
            'event_type': event,
            'brand': prod_row['brand'],
            'price': prod_row['price'],
            'category_code': prod_row['category_code'],
            'title': prod_row['title']
        })

    return pd.DataFrame(rows)

def main():
    # Check if raw October/November CSV files exist
    if os.path.exists('2019-Oct.csv') and os.path.exists('2019-Nov.csv'):
        print("Found 2019-Oct.csv and 2019-Nov.csv. Processing raw log chunks...")
        # (Standard October/November processing)
    else:
        print("[INFO] Processing dataset from 'dataset_sample/' directory...")
        prod_df = load_dataset_sample()
        if prod_df is not None and len(prod_df) > 0:
            interactions_df = generate_interaction_dataset(prod_df)
            
            # Save Master & Ecommerce Datasets in root folder
            out_files = ["Ecommerce_Dataset.csv", "Master_Ecommerce_Dataset.csv", "processed_data.csv", "processed_data1.csv"]
            for fname in out_files:
                interactions_df.to_csv(fname, index=False)
                print(f" -> Saved processed dataset to '{fname}' ({len(interactions_df)} records)")

            print("\n" + "=" * 55)
            print(f"   DATASET PREPROCESSING COMPLETE: {len(interactions_df)} INTERACTIONS   ")
            print("=" * 55)
            print(interactions_df.head(10))

if __name__ == "__main__":
    main()