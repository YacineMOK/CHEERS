"""Additive and monotone 2-additive preference models for CHEERS.

Both modules receive a user representation and a vector of learned criterion activations.  

1. The additive model produces a normalized nonnegative weight vector.  
2. The Choquet model produces the Mobius coefficients of a normalized, monotone 2-additive capacity and **uses the exact minimum operator**.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _expand_before_last(tensor: Tensor, target_ndim: int) -> Tensor:
    """Add candidate axes before the coefficient dimension for broadcasting."""

    if tensor.ndim > target_ndim:
        raise ValueError(
            f"Cannot broadcast a {tensor.ndim}-D coefficient tensor to a "
            f"{target_ndim}-D criterion tensor."
        )
    while tensor.ndim < target_ndim:
        tensor = tensor.unsqueeze(-2)
    return tensor


class AdditivePreference(nn.Module):
    """Normalized, nonnegative additive preference model.

    At initialization, the raw logits are close to zero.  
    Applying softplus and joint normalization therefore starts the model close to the neutral capacity ``a_k = 1 / n_criteria`` while retaining small user-dependent perturbations.
    """

    def __init__(
        self,
        user_dim: int,
        n_criteria: int,
        projection_gain: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if user_dim <= 0:
            raise ValueError(f"user_dim must be positive, received {user_dim}.")
        if n_criteria <= 0:
            raise ValueError(
                f"n_criteria must be positive, received {n_criteria}."
            )
        if projection_gain <= 0:
            raise ValueError(
                f"projection_gain must be positive, received {projection_gain}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, received {eps}.")

        self.user_dim = user_dim
        self.n_criteria = n_criteria
        self.projection_gain = projection_gain
        self.eps = eps

        self.singleton_projection = nn.Linear(user_dim, n_criteria)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start near the uniform additive capacity, not a peaked preference."""

        nn.init.xavier_uniform_(
            self.singleton_projection.weight,
            gain=self.projection_gain,
        )
        nn.init.zeros_(self.singleton_projection.bias)

    def coefficients(self, user_repr: Tensor) -> Tensor:
        """Return additive weights with shape ``(..., n_criteria)``."""

        raw_singletons = self.singleton_projection(user_repr)
        positive_singletons = F.softplus(raw_singletons)
        normalizer = positive_singletons.sum(dim=-1, keepdim=True)
        return positive_singletons / normalizer.clamp_min(self.eps)

    def forward(
        self,
        user_repr: Tensor,
        criteria_values: Tensor,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, Dict[str, Tensor]]:
        """Aggregate criterion activations using user-specific additive weights."""

        if criteria_values.shape[-1] != self.n_criteria:
            raise ValueError(
                "The last criterion dimension must equal n_criteria: "
                f"expected {self.n_criteria}, received "
                f"{criteria_values.shape[-1]}."
            )

        weights = self.coefficients(user_repr)
        broadcast_weights = _expand_before_last(weights, criteria_values.ndim)
        singleton_contributions = broadcast_weights * criteria_values
        scores = singleton_contributions.sum(dim=-1)

        if not return_details:
            return scores

        details = {
            "singleton_coefficients": weights,
            "singleton_contributions": singleton_contributions,
        }
        return scores, details


class Choquet2AdditivePreference(nn.Module):
    """User-specific normalized and monotone 2-additive Choquet model.

    Let ``b_kl`` denote unconstrained pairwise Mobius coefficients.  Singleton
    coefficients are constructed as

        a_k* = softplus(raw_a_k) + sum_l relu(-b_kl).

    Hence ``a_k* + sum_{l: b_kl < 0} b_kl > 0`` for every criterion, which is
    the necessary-and-sufficient monotonicity condition in the 2-additive case.
    All singleton and pairwise coefficients are then divided by the same
    positive total mass, giving a normalized capacity.
    """

    def __init__(
        self,
        user_dim: int,
        n_criteria: int,
        singleton_projection_gain: float = 0.1,
        interaction_init_std: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if user_dim <= 0:
            raise ValueError(f"user_dim must be positive, received {user_dim}.")
        if n_criteria < 2:
            raise ValueError(
                "Choquet2AdditivePreference requires at least two criteria; "
                f"received {n_criteria}."
            )
        if singleton_projection_gain <= 0:
            raise ValueError(
                "singleton_projection_gain must be positive, received "
                f"{singleton_projection_gain}."
            )
        if interaction_init_std < 0:
            raise ValueError(
                "interaction_init_std cannot be negative, received "
                f"{interaction_init_std}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, received {eps}.")

        self.user_dim = user_dim
        self.n_criteria = n_criteria
        self.n_pairs = n_criteria * (n_criteria - 1) // 2
        self.singleton_projection_gain = singleton_projection_gain
        self.interaction_init_std = interaction_init_std
        self.eps = eps

        self.singleton_projection = nn.Linear(user_dim, n_criteria)
        self.interaction_projection = nn.Linear(user_dim, self.n_pairs)

        pair_indices = torch.triu_indices(
            row=n_criteria,
            col=n_criteria,
            offset=1,
        )
        self.register_buffer("pair_left", pair_indices[0], persistent=True)
        self.register_buffer("pair_right", pair_indices[1], persistent=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the model at, or very close to, its additive submodel."""

        nn.init.xavier_uniform_(
            self.singleton_projection.weight,
            gain=self.singleton_projection_gain,
        )
        nn.init.zeros_(self.singleton_projection.bias)

        if self.interaction_init_std == 0.0:
            nn.init.zeros_(self.interaction_projection.weight)
        else:
            nn.init.normal_(
                self.interaction_projection.weight,
                mean=0.0,
                std=self.interaction_init_std,
            )
        nn.init.zeros_(self.interaction_projection.bias)

    def _incident_negative_mass(self, pair_coefficients: Tensor) -> Tensor:
        """Sum ``relu(-b_kl)`` over all pairs incident to each criterion."""

        negative_mass = F.relu(-pair_coefficients)
        incident_mass = pair_coefficients.new_zeros(
            (*pair_coefficients.shape[:-1], self.n_criteria)
        )

        index_shape = (1,) * (pair_coefficients.ndim - 1) + (self.n_pairs,)
        left = self.pair_left.view(index_shape).expand_as(pair_coefficients)
        right = self.pair_right.view(index_shape).expand_as(pair_coefficients)

        incident_mass.scatter_add_(-1, left, negative_mass)
        incident_mass.scatter_add_(-1, right, negative_mass)
        return incident_mass

    def coefficients(self, user_repr: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized singleton and pairwise Mobius coefficients."""

        raw_singletons = self.singleton_projection(user_repr)
        raw_pairs = self.interaction_projection(user_repr)

        singleton_mass = (
            F.softplus(raw_singletons)
            + self._incident_negative_mass(raw_pairs)
        )
        total_mass = (
            singleton_mass.sum(dim=-1, keepdim=True)
            + raw_pairs.sum(dim=-1, keepdim=True)
        )
        total_mass = total_mass.clamp_min(self.eps)

        singletons = singleton_mass / total_mass
        pairs = raw_pairs / total_mass
        return singletons, pairs

    def monotonicity_slack(
        self,
        singletons: Tensor,
        pairs: Tensor,
    ) -> Tensor:
        """Return the worst-case marginal contribution of every criterion."""

        return singletons - self._incident_negative_mass(pairs)

    def forward(
        self,
        user_repr: Tensor,
        criteria_values: Tensor,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, Dict[str, Tensor]]:
        """Evaluate the exact 2-additive Choquet integral."""

        if criteria_values.shape[-1] != self.n_criteria:
            raise ValueError(
                "The last criterion dimension must equal n_criteria: "
                f"expected {self.n_criteria}, received "
                f"{criteria_values.shape[-1]}."
            )

        singletons, pairs = self.coefficients(user_repr)
        broadcast_singletons = _expand_before_last(
            singletons,
            criteria_values.ndim,
        )
        broadcast_pairs = _expand_before_last(pairs, criteria_values.ndim)

        singleton_contributions = broadcast_singletons * criteria_values
        pair_minima = torch.minimum(
            criteria_values.index_select(-1, self.pair_left),
            criteria_values.index_select(-1, self.pair_right),
        )
        interaction_contributions = broadcast_pairs * pair_minima
        scores = (
            singleton_contributions.sum(dim=-1)
            + interaction_contributions.sum(dim=-1)
        )

        if not return_details:
            return scores

        details = {
            "singleton_coefficients": singletons,
            "interaction_coefficients": pairs,
            "singleton_contributions": singleton_contributions,
            "interaction_contributions": interaction_contributions,
            "pair_indices": torch.stack((self.pair_left, self.pair_right)),
            "monotonicity_slack": self.monotonicity_slack(singletons, pairs),
        }
        return scores, details
