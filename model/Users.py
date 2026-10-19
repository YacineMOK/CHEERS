"""
User representation module for the CHEERS prototype.

Provides only initial look-up user embeddings.

Adjust to include init user features if needed/have them in the future.
"""

from __future__ import annotations

import math

from torch import Tensor, nn


class UserEncoder(nn.Module):
    """Learn one unconstrained latent representation per user.

    Parameters
    ----------
    n_users:
        Number of known users.
    embedding_dim:
        Dimension of the representation returned for every user.
    init_std:
        Standard deviation of the isotropic Gaussian initialization.  When it
        is omitted, ``1 / sqrt(embedding_dim)`` is used, so an embedding has
        expected squared L2 norm close to one without constraining it to the
        unit sphere during training.
    """

    def __init__(
        self,
        n_users: int,
        embedding_dim: int = 64,
        init_std: float | None = None,
    ) -> None:
        super().__init__()

        if n_users <= 0:
            raise ValueError(f"n_users must be positive, received {n_users}.")
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, received {embedding_dim}."
            )
        if init_std is not None and init_std <= 0:
            raise ValueError(f"init_std must be positive, received {init_std}.")

        self.n_users = n_users
        self.embedding_dim = embedding_dim
        self.init_std = (1.0 / math.sqrt(embedding_dim) if init_std is None else init_std)

        self.embedding = nn.Embedding(n_users, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize users isotropically without privileging a direction."""

        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)

    def forward(self, user_ids: Tensor) -> Tensor:
        """Return user representations with shape ``(*user_ids.shape, D_u)``."""

        return self.embedding(user_ids)

    def extra_repr(self) -> str:
        return (
            f"n_users={self.n_users}, embedding_dim={self.embedding_dim}, "
            f"init_std={self.init_std:.4g}"
        )
