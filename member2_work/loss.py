"""
loss.py  —  Member 2: Few-Shot Embedding & Metric Learning
Prototypical Network loss function.

How it works (step by step):
1. For each of the N classes, compute its prototype =
   mean of the K support embeddings.
2. For each query embedding, compute squared Euclidean distance
   to every prototype.
3. Treat negative distances as logits → softmax → cross-entropy loss.
   (Minimizing this loss pushes queries closer to their true prototype
   and further from other prototypes.)
4. Also return accuracy over query examples for logging.

Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

import torch
import torch.nn.functional as F


def prototypical_loss(
    support_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Compute prototypical loss and accuracy for one episode.

    Args:
        support_embeddings : (N, K, D)  — embeddings of support set
        query_embeddings   : (N*Q, D)   — embeddings of query set
        query_labels       : (N*Q,)     — integer class labels 0..N-1

    Returns:
        loss     : scalar tensor (cross-entropy over query set)
        accuracy : float (0.0 to 1.0, fraction of queries correct)
    """
    n_way, k_shot, emb_dim = support_embeddings.shape

    # ── Step 1: Compute prototypes ─────────────────────────────────────────────
    # Average support embeddings per class → prototype vector per class
    # Shape: (N, D)
    prototypes = support_embeddings.mean(dim=1)

    # ── Step 2: Squared Euclidean distances ───────────────────────────────────
    # query_embeddings: (N*Q, D)
    # prototypes:       (N, D)
    # Output distances: (N*Q, N)
    #
    # Using the expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    # (since embeddings are L2-normalized, ||a||^2 = ||b||^2 = 1,
    #  so distance = 2 - 2*cosine_similarity — but we keep the general form
    #  for correctness even if normalization changes)
    n_queries = query_embeddings.size(0)

    # Expand for broadcasting: (N*Q, 1, D) - (1, N, D) → (N*Q, N, D)
    diffs = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)  # (N*Q, N, D)
    sq_distances = (diffs ** 2).sum(dim=-1)                          # (N*Q, N)

    # ── Step 3: Loss = cross-entropy over (- distances) ───────────────────────
    # Negative distances as logits: smaller distance → higher logit → preferred class
    logits = -sq_distances                                            # (N*Q, N)
    loss = F.cross_entropy(logits, query_labels)

    # ── Step 4: Accuracy ──────────────────────────────────────────────────────
    predictions = logits.argmax(dim=-1)                               # (N*Q,)
    accuracy = (predictions == query_labels).float().mean().item()

    return loss, accuracy


def compute_prototypes(support_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute class prototypes from support embeddings.
    Exposed separately so Member 3 can reuse this function directly.

    Args:
        support_embeddings: (N, K, D)  or  (K, D) for a single class
    Returns:
        prototypes: (N, D)             or  (D,) for a single class
    """
    return support_embeddings.mean(dim=-2)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance between two sets of vectors.

    Args:
        a: (M, D)
        b: (N, D)
    Returns:
        distances: (M, N) — dist[i,j] = ||a[i] - b[j]||^2
    """
    diffs = a.unsqueeze(1) - b.unsqueeze(0)   # (M, N, D)
    return (diffs ** 2).sum(dim=-1)            # (M, N)
