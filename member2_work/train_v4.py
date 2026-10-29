"""
train_v4.py  —  Member 2: Few-Shot Embedding & Metric Learning (V4 — Best Possible)

Improvements over V3:
  1. ATTENTION PROTOTYPE  — instead of simple mean of support embeddings,
     use a learned attention weight per support sample (hard samples
     get downweighted, easy ones get upweighted). This gives a smarter
     prototype that is more robust to noisy support samples.

  2. MIXUP AUGMENTATION   — randomly blend two training episodes' features
     and their labels. Proven to improve generalization significantly.

  3. VARIABLE N-WAY       — instead of fixed 10-way, each episode randomly
     picks N from {5, 10, 15}. The model learns to discriminate different
     numbers of classes, making the embedding more general.

  4. GRADIENT CLIPPING    — clips gradients to max norm 5.0 to prevent
     any single bad episode from causing a large update.

Run:
    python member2_work/train_v4.py

Saves best checkpoint to:
    outputs/checkpoints/best_embedding_model_v4.pt
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT_RATE,
    LEARNING_RATE, WEIGHT_DECAY, FEATURE_JITTER_STD,
    MIN_SAMPLES_PER_FAMILY, SEED,
    BEST_CHECKPOINT, FINAL_CHECKPOINT, RESULTS_DIR,
)
from model import EmbeddingNetwork, build_model
from loss import euclidean_distance
from train import SplitMalwareDataset, EpisodeSampler, set_seed


# ── Improvement 1: Attention Prototype Network ─────────────────────────────────

class AttentionEmbeddingNetwork(nn.Module):
    """
    Extends EmbeddingNetwork with a small attention head that
    weights each support sample's contribution to the prototype.

    Instead of: prototype = mean(support_embeddings)
    We compute:  attention_score = softmax(W * embed + b)
                 prototype = sum(attention_score * embed)

    This lets the network learn to ignore noisy/outlier support samples.
    """
    def __init__(self):
        super().__init__()
        # Core embedding network (same as V3)
        self.encoder = EmbeddingNetwork(
            input_dim=INPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
            embedding_dim=EMBEDDING_DIM,
            dropout_rate=DROPOUT_RATE,
        )
        # Attention head: maps embedding → scalar score
        self.attention = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Initialize attention weights small so it starts close to uniform
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.encoder(x)

    def compute_prototype(self, support_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted prototype from support embeddings.
        Args:
            support_emb: (K, D)  — K support embeddings for one class
        Returns:
            prototype: (D,)      — attention-weighted mean
        """
        scores = self.attention(support_emb)           # (K, 1)
        weights = F.softmax(scores, dim=0)             # (K, 1) — sum to 1
        prototype = (weights * support_emb).sum(dim=0) # (D,)
        return prototype

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Improvement 2: Mixup Augmentation ──────────────────────────────────────────

def mixup_episode(support_x, query_x, query_y, alpha=0.3):
    """
    Mixup: blend random pairs of query samples and their labels.
    Creates virtual training samples: x_mix = λ*x_a + (1-λ)*x_b

    We only mix QUERY samples (not support) to keep prototypes clean.
    Labels become soft (one-hot smoothed by λ).

    Args:
        support_x: (N, K, D)
        query_x:   (N*Q, D)
        query_y:   (N*Q,)  — class indices 0..N-1
        alpha:     Beta distribution parameter (higher = more mixing)
    Returns:
        support_x (unchanged), query_x_mixed, soft_labels (N*Q, N)
    """
    n_classes = support_x.shape[0]
    n_query   = query_x.shape[0]
    device    = query_x.device

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)   # always >= 0.5 so primary class still dominates

    # Shuffle query set for mixing
    perm = torch.randperm(n_query, device=device)
    query_x_mixed = lam * query_x + (1 - lam) * query_x[perm]

    # Create soft labels: one-hot for original class + one-hot for mixed class
    one_hot_a = F.one_hot(query_y,          num_classes=n_classes).float()
    one_hot_b = F.one_hot(query_y[perm],    num_classes=n_classes).float()
    soft_labels = lam * one_hot_a + (1 - lam) * one_hot_b   # (N*Q, N)

    return support_x, query_x_mixed, soft_labels


# ── Improvement 3: Variable N-way Sampler ──────────────────────────────────────

class VariableNWaySampler:
    """
    Each episode randomly picks N_WAY from {5, 10, 15}.
    Forces the model to handle different numbers of classes per episode,
    making the embedding more generally useful.
    """
    def __init__(self, X, family_to_indices, families, n_way_options,
                 k_shot, n_query):
        self.X                  = X
        self.family_to_indices  = family_to_indices
        self.families           = families
        self.n_way_options      = n_way_options   # e.g., [5, 10, 15]
        self.k_shot             = k_shot
        self.n_query            = n_query
        # Keep only families with enough samples
        self.eligible_families  = [
            f for f in families
            if len(family_to_indices.get(f, [])) >= k_shot + n_query
        ]
        print(f"[VariableNWaySampler] {len(self.eligible_families)} eligible families | "
              f"n_way options: {n_way_options}")

    def sample_episode(self, rng, n_way=None):
        if n_way is None:
            n_way = int(rng.choice(self.n_way_options))
        n_way = min(n_way, len(self.eligible_families))

        chosen_families = rng.choice(self.eligible_families, size=n_way, replace=False)
        support_list, query_x_list, query_y_list = [], [], []

        for cls_idx, family in enumerate(chosen_families):
            idxs = self.family_to_indices[family]
            n = len(idxs)
            needed = self.k_shot + self.n_query
            chosen = rng.choice(n, size=min(needed, n), replace=(n < needed))
            support_list.append(self.X[idxs[chosen[:self.k_shot]]])
            query_x_list.append(self.X[idxs[chosen[self.k_shot:self.k_shot + self.n_query]]])
            query_y_list.extend([cls_idx] * self.n_query)

        support_x = torch.tensor(np.stack(support_list), dtype=torch.float32)
        query_x   = torch.tensor(np.vstack(query_x_list), dtype=torch.float32)
        query_y   = torch.tensor(query_y_list, dtype=torch.long)
        return support_x, query_x, query_y


# ── Loss with attention prototypes ─────────────────────────────────────────────

def attention_proto_loss(model, support_x, query_x, query_y,
                         temperature=10.0, soft_labels=None):
    """
    Compute prototypical loss using attention-weighted prototypes.
    Optionally accepts soft_labels (from mixup) instead of hard labels.
    """
    n_way = support_x.shape[0]
    k     = support_x.shape[1]

    # Embed support samples
    support_flat = support_x.view(-1, support_x.shape[-1])       # (N*K, D)
    support_emb  = model.encoder(support_flat)                   # (N*K, D)
    support_emb  = support_emb.view(n_way, k, -1)               # (N, K, D)

    # Compute attention-weighted prototypes per class
    prototypes = torch.stack([
        model.compute_prototype(support_emb[i]) for i in range(n_way)
    ])   # (N, D)

    # Embed query
    query_emb = model.encoder(query_x)   # (N*Q, D)
    query_emb = F.normalize(query_emb, p=2, dim=-1)

    # Distance to prototypes
    sq_dist = euclidean_distance(query_emb, prototypes)   # (N*Q, N)
    logits  = -sq_dist * temperature

    # Loss
    log_probs = F.log_softmax(logits, dim=-1)   # (N*Q, N)
    if soft_labels is not None:
        loss = -(soft_labels * log_probs).sum(dim=-1).mean()
    else:
        loss = F.nll_loss(log_probs, query_y)

    # Hard accuracy (always use argmax vs true labels)
    acc = (logits.argmax(dim=-1) == query_y).float().mean().item()
    return loss, acc


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_epoch_v4(model, sampler, optimizer, device, n_episodes,
                   jitter_std=FEATURE_JITTER_STD, temperature=10.0,
                   use_mixup=True, mixup_alpha=0.3, max_grad_norm=5.0):
    model.train()
    rng = np.random.default_rng()
    total_loss, total_acc = 0.0, 0.0

    for _ in range(n_episodes):
        support_x, query_x, query_y = sampler.sample_episode(rng)
        support_x = support_x.to(device)
        query_x   = query_x.to(device)
        query_y   = query_y.to(device)

        # Feature jitter
        if jitter_std > 0:
            support_x = support_x + torch.randn_like(support_x) * jitter_std
            query_x   = query_x   + torch.randn_like(query_x)   * jitter_std

        # Mixup augmentation
        soft_labels = None
        if use_mixup and np.random.random() < 0.5:   # 50% chance per episode
            support_x, query_x, soft_labels = mixup_episode(
                support_x, query_x, query_y, alpha=mixup_alpha
            )

        loss, acc = attention_proto_loss(model, support_x, query_x, query_y,
                                          temperature=temperature,
                                          soft_labels=soft_labels)
        optimizer.zero_grad()
        loss.backward()
        # Improvement 4: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += acc

    return total_loss / n_episodes, total_acc / n_episodes


@torch.no_grad()
def val_epoch_v4(model, sampler, device, n_episodes=150, temperature=10.0):
    model.eval()
    rng = np.random.default_rng(SEED + 999)
    total_loss, total_acc = 0.0, 0.0

    for _ in range(n_episodes):
        support_x, query_x, query_y = sampler.sample_episode(rng, n_way=10)
        support_x = support_x.to(device)
        query_x   = query_x.to(device)
        query_y   = query_y.to(device)
        loss, acc = attention_proto_loss(model, support_x, query_x, query_y,
                                          temperature=temperature)
        total_loss += loss.item()
        total_acc  += acc

    return total_loss / n_episodes, total_acc / n_episodes


# ── Full V4 Training Run ───────────────────────────────────────────────────────

def run_training_v4(k_shot=20, n_episodes=500, n_epochs=80,
                    temperature=10.0, use_mixup=True):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  V4 TRAINING: variable-Nway {k_shot}-shot | "
          f"attention-proto | mixup={use_mixup} | T={temperature}")
    print(f"{'='*65}")

    # Load data
    X_train = np.load(os.path.join(os.path.dirname(__file__), '..', 'dataset',
                                    'member2_handoff', 'X_train.npy'))
    y_train = np.load(os.path.join(os.path.dirname(__file__), '..', 'dataset',
                                    'member2_handoff', 'y_train.npy'), allow_pickle=True)

    ds = SplitMalwareDataset(X_train, y_train, val_fraction=0.20, seed=SEED)

    # Variable N-way sampler
    N_WAY_OPTIONS = [5, 10, 15]
    train_sampler = VariableNWaySampler(
        ds.X, ds.train_indices, ds.families,
        n_way_options=N_WAY_OPTIONS, k_shot=k_shot, n_query=15
    )
    val_sampler = VariableNWaySampler(
        ds.X, ds.val_indices, ds.families,
        n_way_options=[10], k_shot=k_shot, n_query=15
    )

    # Build attention model
    model = AttentionEmbeddingNetwork().to(device)
    print(f"[AttentionEmbeddingNetwork] {model.count_parameters():,} trainable parameters")
    print(f"  Attention head adds {sum(p.numel() for p in model.attention.parameters()):,} extra parameters")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    log_rows     = []
    tag          = f"v4_{k_shot}shot"
    best_ckpt    = BEST_CHECKPOINT.replace(".pt", f"_{tag}.pt")
    patience     = 20
    no_improve   = 0

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch_v4(model, train_sampler, optimizer, device,
                                          n_episodes, temperature=temperature,
                                          use_mixup=use_mixup)
        va_loss, va_acc = val_epoch_v4(model, val_sampler, device,
                                        n_episodes=150, temperature=temperature)
        scheduler.step()
        lr  = scheduler.get_last_lr()[0]
        gap = abs(tr_acc - va_acc)

        log_rows.append({"epoch": epoch, "k_shot": k_shot,
                          "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4),
                          "gap": round(gap, 4), "lr": lr})

        marker = "<<BEST>>" if va_acc > best_val_acc else ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": va_acc, "k_shot": k_shot,
                        "model_type": "AttentionEmbeddingNetwork",
                        "temperature": temperature}, best_ckpt)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{n_epochs}  "
              f"train={tr_acc:.4f}  val={va_acc:.4f}  "
              f"gap={gap:.4f}  lr={lr:.6f}  {marker}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(RESULTS_DIR, f"training_log_{tag}.csv")
    log_df.to_csv(log_path, index=False)

    print(f"\n  Best val acc:  {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"  Checkpoint:    {best_ckpt}")
    return best_val_acc


if __name__ == "__main__":
    run_training_v4(k_shot=20, n_episodes=500, n_epochs=80, temperature=10.0, use_mixup=True)
