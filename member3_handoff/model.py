"""
model.py  —  Member 2: Few-Shot Embedding & Metric Learning
PyTorch embedding network.

Architecture:  Input(100) → Dense(256, BN, ReLU, Dropout)
                           → Dense(128, BN, ReLU, Dropout)
                           → Dense(64)
                           → L2-Normalize
                           → 64-dim unit-sphere embedding

Why L2-normalize?
  When all embeddings live on the unit hypersphere, Euclidean distance
  between them is equivalent to cosine distance (d = sqrt(2 - 2*cos(θ))).
  This makes the embedding space geometry very clean and the prototypical
  distance meaningful regardless of scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_DIM, HIDDEN_DIMS, EMBEDDING_DIM, DROPOUT_RATE


class EmbeddingNetwork(nn.Module):
    """
    Dense feed-forward embedding network with BatchNorm and Dropout.
    Maps a malware feature vector → 64-dimensional L2-normalized embedding.
    """

    def __init__(
        self,
        input_dim: int   = INPUT_DIM,
        hidden_dims: list = HIDDEN_DIMS,
        embedding_dim: int = EMBEDDING_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim

        # Build the encoder: input → hidden layers → embedding
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ]
            prev_dim = h_dim

        # Final projection to embedding space (no activation — L2-norm follows)
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor, shape (batch_size, input_dim) or (N, K, input_dim)
        Returns:
            Embeddings, L2-normalized, same shape with last dim = embedding_dim
        """
        original_shape = x.shape
        # Flatten episode dims if needed (e.g., (N, K, D) → (N*K, D))
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])

        z = self.encoder(x)                     # (batch, embedding_dim)
        z = F.normalize(z, p=2, dim=-1)         # L2-normalize onto unit sphere

        # Restore episode shape if needed
        if len(original_shape) > 2:
            z = z.view(*original_shape[:-1], self.embedding_dim)

        return z

    def get_embedding_size(self) -> int:
        return self.embedding_dim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Convenience: build model from config ───────────────────────────────────────

def build_model(device: torch.device = None) -> EmbeddingNetwork:
    """Build and return the embedding network on the specified device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingNetwork().to(device)
    print(f"[EmbeddingNetwork] {model.count_parameters():,} trainable parameters")
    print(f"[EmbeddingNetwork] Architecture: {INPUT_DIM} -> {HIDDEN_DIMS} -> {EMBEDDING_DIM} (L2-norm)")
    print(f"[EmbeddingNetwork] Device: {device}")
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_model()
    dummy = torch.randn(8, INPUT_DIM)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"L2 norms (should all be 1.0): {out.norm(dim=-1)}")
