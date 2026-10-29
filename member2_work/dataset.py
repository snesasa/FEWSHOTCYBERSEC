"""
dataset.py  —  Member 2: Few-Shot Embedding & Metric Learning
Episode Sampler for N-way K-shot prototypical network training.

Key design decisions:
- BENIGN samples are excluded from episodic training (we learn malware-family
  discriminative embeddings, not benign vs malware separation — that is
  Member 1's baseline task).
- Only families with >= MIN_SAMPLES_PER_FAMILY are used for episodic training
  so every episode can draw support + query without overlap.
- Unseen/withheld families are never added here; they are loaded separately
  in evaluate.py for the open-set adaptation experiment.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from config import MIN_SAMPLES_PER_FAMILY, SEED


# ── Malware-only dataset ────────────────────────────────────────────────────────

class MalwareDataset:
    """
    Holds all malware samples (no BENIGN) indexed by family name.
    Provides a clean interface for the episode sampler.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, min_samples: int = MIN_SAMPLES_PER_FAMILY):
        """
        Args:
            X: Feature matrix, shape (N, D)
            y: String labels array, shape (N,)
            min_samples: Minimum number of samples a family needs to be included
        """
        # Build per-family index list (excluding BENIGN)
        family_to_indices = defaultdict(list)
        for i, label in enumerate(y):
            if label != "BENIGN":
                family_to_indices[label].append(i)

        # Filter out families with too few samples
        self.family_to_indices = {
            fam: np.array(idxs)
            for fam, idxs in family_to_indices.items()
            if len(idxs) >= min_samples
        }

        self.X = X
        self.families = sorted(self.family_to_indices.keys())
        self.n_families = len(self.families)

        print(f"[MalwareDataset] {self.n_families} families, "
              f"{sum(len(v) for v in self.family_to_indices.values())} malware samples "
              f"(min_samples >= {min_samples}, BENIGN excluded)")

    def get_family_samples(self, family: str) -> np.ndarray:
        """Return all feature vectors for a given family."""
        return self.X[self.family_to_indices[family]]

    def sample_from_family(self, family: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n indices from a family without replacement."""
        indices = self.family_to_indices[family]
        chosen = rng.choice(len(indices), size=n, replace=False)
        return self.X[indices[chosen]]


# ── Episode Sampler ─────────────────────────────────────────────────────────────

class EpisodeSampler:
    """
    Generates N-way K-shot episodes for prototypical network training.

    Each episode:
      - Randomly selects N families
      - Draws K support samples per family  (the model 'sees' these)
      - Draws Q query samples per family    (the model must classify these)
      - Returns tensors ready for the model

    Returns:
        support_x : (N, K, D) float tensor
        query_x   : (N*Q, D) float tensor
        query_y   : (N*Q,)   long tensor  — class indices 0..N-1
    """

    def __init__(self, dataset: MalwareDataset, n_way: int, k_shot: int, n_query: int):
        self.dataset = dataset
        self.n_way   = n_way
        self.k_shot  = k_shot
        self.n_query = n_query

        # Validate that every family has enough samples
        required = k_shot + n_query
        bad = [f for f in dataset.families
               if len(dataset.family_to_indices[f]) < required]
        if bad:
            print(f"[EpisodeSampler] WARNING: {len(bad)} families have < {required} samples "
                  f"and will sometimes cause replacement sampling: {bad[:5]} ...")

    def sample_episode(self, rng: np.random.Generator):
        """
        Sample one episode.
        Returns: support_x, query_x, query_y  (torch tensors, CPU)
        """
        # 1. Pick N random families
        chosen_families = rng.choice(self.dataset.families, size=self.n_way, replace=False)

        support_list = []
        query_x_list = []
        query_y_list = []

        for class_idx, family in enumerate(chosen_families):
            indices = self.dataset.family_to_indices[family]
            n_available = len(indices)
            needed = self.k_shot + self.n_query
            replace = n_available < needed

            # Sample support + query indices (no overlap)
            if replace:
                # rare: family has fewer samples than needed
                all_chosen = rng.choice(n_available, size=needed, replace=True)
            else:
                all_chosen = rng.choice(n_available, size=needed, replace=False)

            support_indices = indices[all_chosen[:self.k_shot]]
            query_indices   = indices[all_chosen[self.k_shot:]]

            support_list.append(self.dataset.X[support_indices])   # (K, D)
            query_x_list.append(self.dataset.X[query_indices])     # (Q, D)
            query_y_list.extend([class_idx] * self.n_query)

        support_x = torch.tensor(np.stack(support_list), dtype=torch.float32)  # (N, K, D)
        query_x   = torch.tensor(np.vstack(query_x_list), dtype=torch.float32) # (N*Q, D)
        query_y   = torch.tensor(query_y_list, dtype=torch.long)                # (N*Q,)

        return support_x, query_x, query_y

    def __iter__(self):
        """Allows using EpisodeSampler in a for-loop (yields one episode per call)."""
        rng = np.random.default_rng(SEED)
        while True:
            yield self.sample_episode(rng)


# ── Convenience loader ──────────────────────────────────────────────────────────

def load_data():
    """
    Load train and test numpy arrays from the Member 1 handoff.
    Returns: X_train, y_train, X_test, y_test
    """
    from config import X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True)
    X_test  = np.load(X_TEST_PATH)
    y_test  = np.load(Y_TEST_PATH, allow_pickle=True)
    print(f"[load_data] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test
