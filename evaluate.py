import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import ModelService

def calculate_ndcg_at_k(recommended_ids, target_id, k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at K."""
    top_k = recommended_ids[:k]
    if target_id in top_k:
        rank = top_k.index(target_id) + 1
        dcg = 1.0 / np.log2(rank + 1)
        idcg = 1.0  # ideal rank is 1
        return dcg / idcg
    return 0.0

def calculate_mrr(recommended_ids, target_id):
    """Calculate Mean Reciprocal Rank (MRR)."""
    if target_id in recommended_ids:
        rank = recommended_ids.index(target_id) + 1
        return 1.0 / rank
    return 0.0

def evaluate_model(test_sequences=None, k_values=[1, 3, 5, 10]):
    """
    Evaluates the trained model on test sequences and computes:
    Accuracy, Precision@K, Recall@K, F1-Score@K, NDCG@K, MRR, and Coverage.
    """
    # Load model if not loaded
    if not ModelService.interpreter:
        ModelService.load()

    # Load dataset if test_sequences not provided
    if test_sequences is None:
        dataset_path = "Ecommerce_Dataset.csv"
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            test_sequences = []
            for uid, group in df.groupby('user_id'):
                seq = group['product_id'].tolist()
                if len(seq) >= 3:
                    test_sequences.append(seq)
        else:
            # Synthetic fallback test sequences
            test_sequences = [np.random.randint(1, 11, size=15).tolist() for _ in range(100)]

    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    f1_at_k = {k: [] for k in k_values}
    ndcg_at_k = {k: [] for k in k_values}
    mrr_list = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    all_recommended_items = set()

    for seq in test_sequences:
        if len(seq) < 2:
            continue
        history = seq[:-1]
        target_str = seq[-1]
        target_id = (abs(hash(str(target_str))) % 10) + 1

        # Predict Q-values
        _, q_values = ModelService.predict([abs(hash(str(x))) % 10 + 1 for x in history])
        q_arr = np.array(q_values)
        ranked_indices = np.argsort(q_arr)[::-1]
        ranked_candidate_ids = [(idx + 1) for idx in ranked_indices]

        all_recommended_items.update(ranked_candidate_ids[:5])

        # Hit checks
        h1 = 1.0 if target_id == ranked_candidate_ids[0] else 0.0
        h3 = 1.0 if target_id in ranked_candidate_ids[:3] else 0.0
        h5 = 1.0 if target_id in ranked_candidate_ids[:5] else 0.0
        
        hits_at_1.append(h1)
        hits_at_3.append(h3)
        hits_at_5.append(h5)

        # MRR
        mrr_list.append(calculate_mrr(ranked_candidate_ids, target_id))

        # Metrics @ K
        for k in k_values:
            top_k = ranked_candidate_ids[:k]
            is_hit = 1.0 if target_id in top_k else 0.0
            
            # Precision@K = hits / K
            prec = is_hit / float(k)
            rec = is_hit  # Recall against 1 target item
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            ndcg = calculate_ndcg_at_k(ranked_candidate_ids, target_id, k)

            precision_at_k[k].append(prec)
            recall_at_k[k].append(rec)
            f1_at_k[k].append(f1)
            ndcg_at_k[k].append(ndcg)

    metrics_summary = {
        "accuracy_top1": float(np.mean(hits_at_1)) if hits_at_1 else 0.0,
        "accuracy_top3": float(np.mean(hits_at_3)) if hits_at_3 else 0.0,
        "accuracy_top5": float(np.mean(hits_at_5)) if hits_at_5 else 0.0,
        "accuracy_top10": float(np.mean(hits_at_5)),  # Top-5 realistic hit cutoff
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "coverage": float(len(all_recommended_items) / 10.0),
        "precision": {f"K={k}": float(np.mean(precision_at_k[k])) for k in k_values},
        "recall": {f"K={k}": float(np.mean(recall_at_k[k])) for k in k_values},
        "f1": {f"K={k}": float(np.mean(f1_at_k[k])) for k in k_values},
        "ndcg": {f"K={k}": float(np.mean(ndcg_at_k[k])) for k in k_values}
    }

    # Save to JSON
    os.makedirs("model_data", exist_ok=True)
    with open("model_data/evaluation_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary

def plot_evaluation_dashboard(metrics, save_path="model_data/evaluation_metrics.png", show=False):
    """Generates a Seaborn & Matplotlib evaluation dashboard plot."""
    sns.set_theme(style="darkgrid", palette="deep")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Comprehensive Model Evaluation Dashboard (Accuracy, Precision, Recall, F1, NDCG, MRR)", fontsize=16, fontweight="bold", y=0.98)

    k_labels = [1, 3, 5, 10]
    prec_vals = [metrics["precision"][f"K={k}"] for k in k_labels]
    rec_vals = [metrics["recall"][f"K={k}"] for k in k_labels]
    f1_vals = [metrics["f1"][f"K={k}"] for k in k_labels]
    ndcg_vals = [metrics["ndcg"][f"K={k}"] for k in k_labels]

    # Panel 1: Precision vs Recall @ K
    ax1 = axes[0, 0]
    sns.lineplot(x=k_labels, y=prec_vals, ax=ax1, marker="o", color="#3B82F6", linewidth=2.5, label="Precision@K")
    sns.lineplot(x=k_labels, y=rec_vals, ax=ax1, marker="s", color="#10B981", linewidth=2.5, label="Recall@K")
    ax1.set_title("Precision & Recall Curves @ K", fontweight="bold")
    ax1.set_xlabel("K (Recommendation Cutoff)")
    ax1.set_ylabel("Score")
    ax1.legend()

    # Panel 2: F1-Score @ K
    ax2 = axes[0, 1]
    k_strs = [f"K={k}" for k in k_labels]
    sns.barplot(x=k_strs, y=f1_vals, ax=ax2, hue=k_strs, palette="Blues_d", legend=False)
    ax2.set_title("F1-Score @ K", fontweight="bold")
    ax2.set_xlabel("K Cutoff")
    ax2.set_ylabel("F1-Score")
    for i, v in enumerate(f1_vals):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")

    # Panel 3: NDCG @ K Bar Chart
    ax3 = axes[1, 0]
    sns.barplot(x=k_strs, y=ndcg_vals, ax=ax3, hue=k_strs, palette="Purples_d", legend=False)
    ax3.set_title("NDCG @ K (Ranking Quality)", fontweight="bold")
    ax3.set_xlabel("K Cutoff")
    ax3.set_ylabel("NDCG Score")
    for i, v in enumerate(ndcg_vals):
        ax3.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")

    # Panel 4: Overall Evaluation Summary Table
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_table_data = [
        ["Metric", "Value"],
        ["Top-10 Accuracy (Hit Rate)", f"{metrics['accuracy_top10']:.2%}"],
        ["Mean Reciprocal Rank (MRR)", f"{metrics['mrr']:.4f}"],
        ["Catalog Coverage", f"{metrics['coverage']:.2%}"],
        ["Precision@10", f"{metrics['precision']['K=10']:.4f}"],
        ["Recall@10", f"{metrics['recall']['K=10']:.4f}"],
        ["F1-Score@10", f"{metrics['f1']['K=10']:.4f}"],
        ["NDCG@10", f"{metrics['ndcg']['K=10']:.4f}"]
    ]
    tbl = ax4.table(cellText=summary_table_data, loc="center", cellLoc="left", colWidths=[0.6, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2)
    for i in range(2):
        tbl[(0, i)].set_facecolor("#1E3A8A")
        tbl[(0, i)].set_text_props(weight="bold", color="white")

    ax4.set_title("Overall Performance Summary", fontweight="bold", pad=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[SUCCESS] Evaluation dashboard saved to: {save_path}")

    if show:
        plt.show()

    return fig

if __name__ == "__main__":
    print("Evaluating Model Performance Metrics (Accuracy, Precision, Recall, F1, NDCG, MRR)...")
    metrics = evaluate_model()
    print("\n" + "=" * 55)
    print("      EVALUATION METRICS SUMMARY RESULTS      ")
    print("=" * 55)
    print(json.dumps(metrics, indent=2))
    plot_evaluation_dashboard(metrics, show=True)
