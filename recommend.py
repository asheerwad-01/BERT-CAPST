import os
import sys
import argparse
import numpy as np
import pandas as pd

# Force stdout encoding safety on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import ModelService

def load_product_catalog():
    """Load real product details from Ecommerce_Dataset.csv if available."""
    dataset_path = "Ecommerce_Dataset.csv"
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            products = df.drop_duplicates(subset=['product_id']).copy()
            return products
        except Exception as e:
            print(f"[WARNING] Could not load product catalog: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Get Product Recommendations from Trained Hybrid RL Recommender")
    parser.add_argument("--history", nargs="+", type=int, help="Sequence of past product IDs viewed/clicked by user (e.g. --history 1 4 2 8)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt to enter user interaction history")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   E-COMMERCE HYBRID RECOMMENDER SYSTEM - INFERENCE ENGINE ")
    print("=" * 60)

    # 1. Load trained TFLite model
    print("[1/2] Loading trained model from 'model_data/hybrid_dqn_model.tflite'...")
    try:
        ModelService.load()
    except Exception as e:
        print(f"[ERROR] Failed to load trained model: {e}")
        print("Please ensure you have trained the model by running 'python train.py' first.")
        return

    # 2. Load catalog metadata
    catalog = load_product_catalog()
    if catalog is not None:
        print(f"[2/2] Loaded real product catalog with {len(catalog)} unique items from dataset_sample.\n")
    else:
        print("[2/2] Product catalog loaded.\n")

    # 3. Obtain user history
    history = args.history
    if not history or args.interactive:
        print("--- Product Catalog Sample (dataset_sample) ---")
        if catalog is not None:
            sample_items = catalog[['product_id', 'title', 'brand', 'price', 'category_code']].head(10)
            print(sample_items.to_string(index=False))
            print("-" * 60)

        user_input = input("\nEnter user's past viewed/clicked product IDs (1-10) separated by spaces [e.g. 1 3 5 2]: ").strip()
        if not user_input:
            print("No history provided. Defaulting to sample user browsing history: [1, 3, 5, 2, 7]")
            history = [1, 3, 5, 2, 7]
        else:
            try:
                history = [int(x) for x in user_input.split()]
            except ValueError:
                print("[ERROR] Invalid input. Please enter integers separated by spaces.")
                return

    # 4. Predict Recommendation
    best_action, q_values = ModelService.predict(history)
    
    print("\n" + "=" * 60)
    print(f" INPUT USER BROWSING HISTORY : {history}")
    print(f" RECOMMENDED CANDIDATE ID   : Product #{best_action}")
    print("=" * 60)

    # Lookup product metadata if available
    if catalog is not None and len(catalog) >= best_action:
        matched_item = catalog.iloc[(best_action - 1) % len(catalog)]
        print(f" Title        : {matched_item.get('title', 'N/A')}")
        print(f" Brand        : {matched_item.get('brand', 'N/A')}")
        print(f" Price        : ${matched_item.get('price', 'N/A')}")
        print(f" Category     : {matched_item.get('category_code', 'N/A')}")
        print("=" * 60)

    # Display Top 5 Ranked Recommendations by Q-Value Score
    print("\nTOP RANKED CANDIDATE RECOMMENDATIONS (By Learned Q-Value):")
    q_arr = np.array(q_values)
    top_indices = np.argsort(q_arr)[::-1]
    
    for rank, idx in enumerate(top_indices[:5], start=1):
        item_id = idx + 1
        score = q_arr[idx]
        title_str = ""
        if catalog is not None and len(catalog) >= item_id:
            item_row = catalog.iloc[(item_id - 1) % len(catalog)]
            title_str = f" | {item_row.get('title', '')[:45]}..."
        
        star = " -> [BEST CHOICE]" if item_id == best_action else "               "
        print(f"  {rank}. Product #{item_id:2d} (Q-Value: {score:+.4f}) {star} {title_str}")
        
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
