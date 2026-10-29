"""
evaluate.py  —  Member 2: Few-Shot Embedding & Metric Learning
Systematic evaluation of the trained embedding across all shot settings.

Run:
    python evaluate.py                          # evaluates best checkpoint
    python evaluate.py --checkpoint path/to.pt  # specific checkpoint

Output:
    outputs/results/evaluation_results.csv   — per-shot accuracy table
    outputs/results/per_family_accuracy.csv  — accuracy broken down by family
    Printed confidence intervals using 95% CI over evaluation episodes
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EVAL_SHOTS, EVAL_N_WAY, EVAL_EPISODES, SEED,
    BEST_CHECKPOINT, RESULTS_DIR,
)
from dataset import MalwareDataset, EpisodeSampler, load_data
from model import EmbeddingNetwork, build_model
from loss import prototypical_loss, compute_prototypes, euclidean_distance


# ── Load a trained checkpoint ───────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device) -> EmbeddingNetwork:
    """Load model weights from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[load_checkpoint] Loaded from {checkpoint_path}")
    print(f"  Saved at epoch {ckpt.get('epoch', '?')} | "
          f"val_acc={ckpt.get('val_acc', '?'):.4f} | "
          f"{ckpt.get('n_way','?')}-way {ckpt.get('k_shot','?')}-shot")
    return model


# ── N-way K-shot evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_few_shot(
    model: EmbeddingNetwork,
    dataset: MalwareDataset,
    n_way: int,
    k_shot: int,
    n_query: int = 10,
    n_episodes: int = EVAL_EPISODES,
    device: torch.device = None,
    seed: int = SEED,
) -> dict:
    """
    Evaluate the model over n_episodes of N-way K-shot classification.
    Returns a dict with mean accuracy and 95% confidence interval.
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure we have enough families
    if dataset.n_families < n_way:
        print(f"  WARNING: Only {dataset.n_families} families, reducing n_way to {dataset.n_families}")
        n_way = dataset.n_families

    model.eval()
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot, n_query=n_query)
    rng = np.random.default_rng(seed)

    accs = []
    for _ in range(n_episodes):
        support_x, query_x, query_y = sampler.sample_episode(rng)
        support_x = support_x.to(device)
        query_x   = query_x.to(device)
        query_y   = query_y.to(device)

        support_emb = model(support_x)
        query_emb   = model(query_x)
        _, acc = prototypical_loss(support_emb, query_emb, query_y)
        accs.append(acc)

    accs_arr = np.array(accs)
    mean_acc = accs_arr.mean()
    # 95% confidence interval
    ci = scipy_stats.t.interval(
        0.95, df=len(accs_arr) - 1,
        loc=mean_acc,
        scale=scipy_stats.sem(accs_arr)
    )

    return {
        "n_way": n_way,
        "k_shot": k_shot,
        "mean_acc": float(mean_acc),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "std": float(accs_arr.std()),
        "n_episodes": n_episodes,
    }


# ── Per-family accuracy ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_family(
    model: EmbeddingNetwork,
    dataset: MalwareDataset,
    k_shot: int,
    n_query: int = 10,
    device: torch.device = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    For each family individually: use k_shot support samples,
    build a prototype, and classify the remaining samples.
    This gives per-family accuracy — useful to see which families are hard.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)

    rows = []
    for family in dataset.families:
        samples = dataset.family_to_indices[family]
        n_total = len(samples)
        if n_total < k_shot + 1:
            continue   # not enough samples to evaluate

        # Use k_shot as support, rest as query
        idx = rng.permutation(n_total)
        support_idx = samples[idx[:k_shot]]
        query_idx   = samples[idx[k_shot:min(k_shot + n_query, n_total)]]

        support_x = torch.tensor(dataset.X[support_idx], dtype=torch.float32).to(device)
        query_x   = torch.tensor(dataset.X[query_idx],   dtype=torch.float32).to(device)

        support_emb = model(support_x).unsqueeze(0)    # (1, K, D)
        query_emb   = model(query_x)                    # (Q, D)

        prototype = compute_prototypes(support_emb).squeeze(0)   # (D,)
        distances = euclidean_distance(query_emb, prototype.unsqueeze(0))  # (Q, 1)
        # Since there's only 1 prototype, all queries are 'classified' as this family
        # We use this as a reconstruction accuracy — measures intra-family cohesion
        # Intra-family distance = mean distance from query to its own prototype
        mean_dist = distances.mean().item()

        rows.append({
            "family": family,
            "n_samples": n_total,
            "k_shot": k_shot,
            "mean_intra_dist": round(mean_dist, 4),
        })

    return pd.DataFrame(rows).sort_values("mean_intra_dist")


# ── Full evaluation pipeline ────────────────────────────────────────────────────

def run_full_evaluation(checkpoint_path: str = None):
    """
    Run complete evaluation across all shot settings and save results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoint
    if checkpoint_path is None:
        # Look for 5-shot best checkpoint (default)
        checkpoint_path = BEST_CHECKPOINT.replace(".pt", "_5way_5shot.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = BEST_CHECKPOINT.replace(".pt", f"_{EVAL_N_WAY}way_5shot.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint found. Please run train.py first.\n"
                f"Expected: {checkpoint_path}"
            )

    model = load_checkpoint(checkpoint_path, device)

    # Load test data (families not seen during this training epoch)
    X_train, y_train, X_test, y_test = load_data()
    test_dataset = MalwareDataset(X_test, y_test, min_samples=5)

    print(f"\n[Evaluation] Test dataset: {test_dataset.n_families} families")
    print(f"[Evaluation] Running {len(EVAL_SHOTS)} shot settings: {EVAL_SHOTS}")

    # ── Shot experiments ──────────────────────────────────────────────────────
    results = []
    for k_shot in EVAL_SHOTS:
        print(f"\n  Evaluating {EVAL_N_WAY}-way {k_shot}-shot ...")
        res = evaluate_few_shot(
            model, test_dataset,
            n_way=EVAL_N_WAY, k_shot=k_shot,
            n_episodes=EVAL_EPISODES, device=device
        )
        results.append(res)
        print(f"    Accuracy: {res['mean_acc']*100:.2f}%  "
              f"[95% CI: {res['ci_low']*100:.2f}% – {res['ci_high']*100:.2f}%]")

    # Save results
    results_df = pd.DataFrame(results)
    results_df["mean_acc_pct"]  = (results_df["mean_acc"] * 100).round(2)
    results_df["ci_low_pct"]    = (results_df["ci_low"]   * 100).round(2)
    results_df["ci_high_pct"]   = (results_df["ci_high"]  * 100).round(2)

    out_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n[Evaluation] Results saved: {out_path}")

    # Print summary table
    print("\n" + "="*65)
    print(f"  EVALUATION RESULTS — {EVAL_N_WAY}-way Few-Shot Classification")
    print("="*65)
    print(f"  {'K-Shot':<10} {'Accuracy':>12}  {'95% CI':>24}")
    print(f"  {'-'*50}")
    for row in results:
        print(f"  {row['k_shot']}-shot      "
              f"{row['mean_acc']*100:>10.2f}%  "
              f"[{row['ci_low']*100:.2f}% – {row['ci_high']*100:.2f}%]")
    print("="*65)

    # Per-family intra-cohesion analysis (k=5)
    print("\n[Evaluation] Computing per-family cohesion (5-shot) ...")
    per_fam_df = evaluate_per_family(model, test_dataset, k_shot=5, device=device)
    per_fam_path = os.path.join(RESULTS_DIR, "per_family_cohesion.csv")
    per_fam_df.to_csv(per_fam_path, index=False)
    print(f"  Saved: {per_fam_path}")
    print(f"\n  Most cohesive families (smallest intra-distance):")
    print(per_fam_df.head(10).to_string(index=False))
    print(f"\n  Least cohesive families (largest intra-distance):")
    print(per_fam_df.tail(10).to_string(index=False))

    return results_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate few-shot malware embedding model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint file")
    args = parser.parse_args()
    run_full_evaluation(checkpoint_path=args.checkpoint)
