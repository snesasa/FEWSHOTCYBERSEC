"""
embedder.py  —  Member 2: Few-Shot Embedding & Metric Learning
Clean, self-contained API for Member 3.

This is the ONLY file Member 3 needs to import to use your embedding.
Usage:
    from embedder import load_embedder, preprocess_features

    embed = load_embedder("outputs/checkpoints/best_embedding_model_5way_5shot.pt")
    features_scaled = preprocess_features(raw_features)   # numpy (N, 100)
    embeddings = embed(features_scaled)                    # numpy (N, 64)

The embed() function always returns L2-normalized 64-dim embeddings on CPU.
"""

import os
import sys
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmbeddingNetwork
from config import (
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT_RATE,
    PREPROCESSOR_PATH, BEST_CHECKPOINT, CHECKPOINT_DIR,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_embedder(checkpoint_path: str = None, device: str = "cpu"):
    """
    Load the trained embedding model and return a callable embed() function.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
                         If None, looks for the default best checkpoint.
        device:          "cpu" or "cuda". Default is "cpu" for portability.

    Returns:
        embed: callable  (numpy_array) → numpy_array
               Input shape:  (N, 100)  — N preprocessed malware feature vectors
               Output shape: (N, 64)   — N L2-normalized embeddings
    """
    if checkpoint_path is None:
        # Try common checkpoint names
        candidates = [
            os.path.join(CHECKPOINT_DIR, "best_embedding_model_5way_5shot.pt"),
            os.path.join(CHECKPOINT_DIR, "best_embedding_model_5way_1shot.pt"),
            BEST_CHECKPOINT,
        ]
        checkpoint_path = next((p for p in candidates if os.path.exists(p)), None)
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Please run train.py first.\n"
                f"Looked in: {CHECKPOINT_DIR}"
            )

    _device = torch.device(device)

    # Load model
    ckpt  = torch.load(checkpoint_path, map_location=_device)
    model = EmbeddingNetwork(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        embedding_dim=EMBEDDING_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[load_embedder] Loaded: {os.path.basename(checkpoint_path)}")
    print(f"[load_embedder] Trained as {ckpt.get('n_way','?')}-way {ckpt.get('k_shot','?')}-shot | "
          f"val_acc={ckpt.get('val_acc', '?'):.4f}")
    print(f"[load_embedder] Input dim: {INPUT_DIM} → Output dim: {EMBEDDING_DIM} (L2-normalized)")

    @torch.no_grad()
    def embed(features: np.ndarray) -> np.ndarray:
        """
        Convert preprocessed malware feature vectors to embeddings.

        Args:
            features: numpy array of shape (N, 100) — float32 or float64
                      MUST be preprocessed using preprocess_features() first.

        Returns:
            embeddings: numpy array of shape (N, 64) — L2-normalized float32
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        x   = torch.tensor(features, dtype=torch.float32).to(_device)
        emb = model(x)
        return emb.cpu().numpy()

    return embed


def preprocess_features(
    raw_features: np.ndarray,
    preprocessor_path: str = PREPROCESSOR_PATH,
) -> np.ndarray:
    """
    Apply Member 1's feature preprocessor (scaler) to raw features.
    Call this BEFORE passing features to embed().

    Args:
        raw_features:      numpy array of shape (N, 100) — raw / unscaled
        preprocessor_path: path to feature_preprocessor.joblib from member2_handoff

    Returns:
        scaled_features: numpy array of shape (N, 100) — standardized float32
    """
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found at: {preprocessor_path}\n"
            f"Copy feature_preprocessor.joblib from the member2_handoff directory."
        )
    preprocessor = joblib.load(preprocessor_path)
    scaled = preprocessor.transform(raw_features).astype(np.float32)
    return scaled


def get_embedding_info() -> dict:
    """
    Return metadata about the embedding interface.
    Member 3 can call this to confirm the API contract.
    """
    return {
        "input_dim":     INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "normalized":    True,         # L2-norm applied → unit hypersphere
        "distance_metric": "euclidean", # preferred for prototypical networks
        "architecture":  f"{INPUT_DIM} → {HIDDEN_DIMS} → {EMBEDDING_DIM} (L2-norm)",
        "note": (
            "Embeddings are L2-normalized. Euclidean distance between "
            "normalized vectors equals sqrt(2 - 2*cosine_similarity). "
            "Use euclidean distance for prototype comparison."
        ),
    }


# ── Quick demo / smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  embedder.py — API smoke test")
    print("=" * 55)

    # Show API info
    info = get_embedding_info()
    print("\n[API Info]")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Try loading the embedder
    print("\n[Loading Embedder]")
    try:
        embed = load_embedder()

        # Generate fake test features (same shape as real data)
        fake_features = np.random.randn(10, INPUT_DIM).astype(np.float32)
        embeddings    = embed(fake_features)

        print(f"\n[Smoke Test]")
        print(f"  Input:  {fake_features.shape}")
        print(f"  Output: {embeddings.shape}")
        l2_norms = np.linalg.norm(embeddings, axis=1)
        print(f"  L2 norms (should all be ~1.0): {l2_norms.round(4)}")
        print("\n✅ embedder.py is working correctly.")
    except FileNotFoundError as e:
        print(f"\n⚠  {e}")
        print("   Run train.py first to generate a checkpoint, then test again.")
