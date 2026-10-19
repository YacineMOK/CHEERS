"""Trainable feature-value embeddings for the CHEERS item representation."""

from __future__ import annotations

import math
from collections.abc import Mapping

from torch import Tensor, nn


class _FeatureValueEmbedding(nn.Module):
    """Own the embedding matrix for every possible value of one feature."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, received {vocab_size}."
            )
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, received {embedding_dim}."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize values isotropically with expected squared norm near one."""

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.embedding_dim),
        )

    def forward(self) -> Tensor:
        """Return the complete matrix with shape (V_f, d_f)."""

        return self.embedding.weight


class ItemEncoder(nn.Module):
    """Return all trainable feature-value embedding matrices.

    The loader supplies only vocabulary sizes. Item composition is performed
    later by CriteriaEncoder through the normalized incidence matrices.
    """

    def __init__(
        self,
        vocab_sizes: Mapping[str, int],
        feature_dims: Mapping[str, int],
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("feature_dims must contain at least one feature.")

        missing = set(feature_dims) - set(vocab_sizes)
        extra = set(vocab_sizes) - set(feature_dims)
        if missing or extra:
            raise ValueError(
                "vocab_sizes and feature_dims must contain the same features; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )

        self.feature_names = tuple(feature_dims)
        self.feature_dims = {
            feature: int(feature_dims[feature]) for feature in self.feature_names
        }
        self.vocab_sizes = {
            feature: int(vocab_sizes[feature]) for feature in self.feature_names
        }

        self.feature_embeddings = nn.ModuleDict(
            {
                feature: _FeatureValueEmbedding(
                    vocab_size=self.vocab_sizes[feature],
                    embedding_dim=self.feature_dims[feature],
                )
                for feature in self.feature_names
            }
        )

    def forward(self) -> dict[str, Tensor]:
        """Return one complete feature-value embedding matrix per feature."""

        return {
            feature: self.feature_embeddings[feature]()
            for feature in self.feature_names
        }

    def embedding_matrix(self, feature: str) -> Tensor:
        """Return one feature-value embedding matrix for inspection."""

        return self.feature_embeddings[feature]()

    def extra_repr(self) -> str:
        return f"features={self.feature_names}"
