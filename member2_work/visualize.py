"""
visualize.py  —  Member 2: Few-Shot Embedding & Metric Learning
t-SNE visualization of the learned embedding space.

Run:
    python visualize.py                     # uses best 5-shot checkpoint
    python visualize.py --checkpoint x.pt   # specific checkpoint
    python visualize.py --n_families 20     # show only top N families

Saves:
    outputs/plots/tsne_embedding_space.png   — t-SNE color-coded by family
    outputs/plots/training_curves_<tag>.png  — loss & accuracy curves per run
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from config import BEST_CHECKPOINT, PLOT_DIR, RESULTS_DIR, EVAL_SHOTS, EVAL_N_WAY
from dataset import MalwareDataset, load_data
from model import EmbeddingNetwork, build_model


# ── t-SNE Embedding Visualization ──────────────────────────────────────────────

@torch.no_grad()
def plot_tsne(
    model: EmbeddingNetwork,
    dataset: MalwareDataset,
    n_families: int = 20,
    max_samples_per_family: int = 50,
    device: torch.device = None,
    save_path: str = None,
):
    """
    Plot t-SNE visualization of the embedding space.
    Shows up to n_families families, each in a different color.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Select top-N families by sample count for cleaner visualization
    family_counts = {f: len(dataset.family_to_indices[f]) for f in dataset.families}
    top_families  = sorted(family_counts, key=lambda f: -family_counts[f])[:n_families]

    # Collect embeddings
    all_embeddings = []
    all_labels     = []

    for family in top_families:
        indices = dataset.family_to_indices[family]
        # Sample at most max_samples_per_family
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(indices), size=min(max_samples_per_family, len(indices)), replace=False)
        X_fam = dataset.X[indices[chosen]]
        x_t   = torch.tensor(X_fam, dtype=torch.float32).to(device)
        emb   = model(x_t).cpu().numpy()
        all_embeddings.append(emb)
        all_labels.extend([family] * len(emb))

    X_emb  = np.vstack(all_embeddings)     # (total_samples, 64)
    labels = np.array(all_labels)

    print(f"[t-SNE] Computing on {X_emb.shape[0]} samples, {n_families} families ...")
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
    X_2d   = tsne.fit_transform(X_emb)     # (total_samples, 2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    cmap    = cm.get_cmap("tab20", n_families)
    family_list = sorted(set(labels))

    for i, family in enumerate(family_list):
        mask = labels == family
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[cmap(i)], label=family, s=25, alpha=0.8, edgecolors="none"
        )

    ax.set_title(
        f"t-SNE of Learned Embedding Space\n"
        f"Top-{n_families} Malware Families | {X_emb.shape[0]} samples",
        color="white", fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("t-SNE Component 1", color="#aaaaaa")
    ax.set_ylabel("t-SNE Component 2", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    legend = ax.legend(
        loc="upper right", fontsize=7, framealpha=0.3,
        facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white",
        ncol=2
    )

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, "tsne_embedding_space.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[t-SNE] Saved: {save_path}")
    return save_path


# ── Training Curve Plots ────────────────────────────────────────────────────────

def plot_training_curves(log_csv_path: str = None, save_dir: str = None):
    """
    Plot training/validation loss and accuracy curves.
    Can plot a single CSV or all CSVs in RESULTS_DIR.
    """
    if save_dir is None:
        save_dir = PLOT_DIR

    if log_csv_path:
        csv_files = [log_csv_path]
    else:
        csv_files = [
            os.path.join(RESULTS_DIR, f)
            for f in os.listdir(RESULTS_DIR)
            if f.startswith("training_log_") and f.endswith(".csv")
        ]

    if not csv_files:
        print("[plot_training_curves] No training logs found in", RESULTS_DIR)
        return

    for csv_path in csv_files:
        df  = pd.read_csv(csv_path)
        tag = os.path.basename(csv_path).replace("training_log_", "").replace(".csv", "")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0f0f1a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#0f0f1a")
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        # Loss
        ax1.plot(df["epoch"], df["train_loss"], color="#4fc3f7", linewidth=2, label="Train Loss")
        ax1.plot(df["epoch"], df["val_loss"],   color="#ef9a9a", linewidth=2, label="Val Loss",   linestyle="--")
        ax1.set_title(f"Loss  [{tag}]", color="white", fontsize=12)
        ax1.set_xlabel("Epoch", color="#aaaaaa")
        ax1.set_ylabel("Loss",  color="#aaaaaa")
        ax1.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")
        ax1.grid(alpha=0.2, color="#333355")

        # Accuracy
        ax2.plot(df["epoch"], df["train_acc"], color="#a5d6a7", linewidth=2, label="Train Acc")
        ax2.plot(df["epoch"], df["val_acc"],   color="#ffcc80", linewidth=2, label="Val Acc",   linestyle="--")
        ax2.set_title(f"Accuracy  [{tag}]", color="white", fontsize=12)
        ax2.set_xlabel("Epoch", color="#aaaaaa")
        ax2.set_ylabel("Accuracy", color="#aaaaaa")
        ax2.set_ylim(0, 1)
        ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white")
        ax2.grid(alpha=0.2, color="#333355")

        plt.suptitle(
            f"Training Curves — {tag.replace('_', ' ').replace('way ', '-way ').replace('shot', '-shot')}",
            color="white", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"training_curves_{tag}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"[plot_training_curves] Saved: {save_path}")


# ── Shot Comparison Bar Chart ───────────────────────────────────────────────────

def plot_shot_comparison(results_csv: str = None, save_path: str = None):
    """
    Bar chart comparing accuracy at 1/5/10/20-shot settings.
    """
    if results_csv is None:
        results_csv = os.path.join(RESULTS_DIR, "evaluation_results.csv")

    if not os.path.exists(results_csv):
        print(f"[plot_shot_comparison] File not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    shots = df["k_shot"].tolist()
    accs  = (df["mean_acc"] * 100).tolist()
    ci_lo = (df["mean_acc"] - df["ci_low"]) * 100
    ci_hi = (df["ci_high"] - df["mean_acc"]) * 100

    colors = ["#ef9a9a", "#80cbc4", "#a5d6a7", "#4fc3f7"]
    bars   = ax.bar(
        [str(k) + "-shot" for k in shots], accs,
        color=colors[:len(shots)], width=0.5, alpha=0.85,
        yerr=[ci_lo, ci_hi], capsize=6, error_kw={"color": "white", "linewidth": 1.5}
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold"
        )

    ax.set_title(
        f"{EVAL_N_WAY}-Way Few-Shot Accuracy vs Number of Support Examples",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Shot Setting", color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Accuracy (%)", color="#aaaaaa", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(alpha=0.2, color="#333355")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, "shot_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot_shot_comparison] Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize few-shot malware embedding")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--n_families", type=int, default=20,
                        help="Number of families to show in t-SNE")
    parser.add_argument("--skip_tsne", action="store_true",
                        help="Skip t-SNE (slow for large data)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Training curves (no model needed) ─────────────────────────────────────
    print("\n[1] Plotting training curves ...")
    plot_training_curves()

    # ── Shot comparison bar chart ──────────────────────────────────────────────
    print("\n[2] Plotting shot comparison ...")
    plot_shot_comparison()

    # ── t-SNE visualization ────────────────────────────────────────────────────
    if not args.skip_tsne:
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = BEST_CHECKPOINT.replace(".pt", f"_{EVAL_N_WAY}way_5shot.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = BEST_CHECKPOINT.replace(".pt", "_5way_5shot.pt")

        if os.path.exists(ckpt_path):
            print("\n[3] Computing t-SNE visualization ...")
            from model import build_model
            ckpt = torch.load(ckpt_path, map_location=device)
            model = build_model(device)
            model.load_state_dict(ckpt["model_state_dict"])

            X_train, y_train, X_test, y_test = load_data()
            dataset = MalwareDataset(X_test, y_test, min_samples=5)
            plot_tsne(model, dataset, n_families=args.n_families, device=device)
        else:
            print(f"\n[3] Skipping t-SNE — checkpoint not found at {ckpt_path}")
            print("     Run train.py first.")
