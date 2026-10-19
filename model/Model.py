"""End-to-end CHEERS forward pass from user and item IDs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class CHEERSModel(nn.Module):
    """Compose user, feature-value, criterion, and preference modules."""

    def __init__(
        self,
        user_encoder: nn.Module,
        item_encoder: nn.Module,
        criteria_encoder: nn.Module,
        preference_model: nn.Module,
        maximum_prototype_cosine: float = 0.0,
    ) -> None:
        super().__init__()

        if not -1.0 <= maximum_prototype_cosine <= 1.0:
            raise ValueError(
                "maximum_prototype_cosine must lie in [-1, 1]; "
                f"received {maximum_prototype_cosine}."
            )

        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.criteria_encoder = criteria_encoder
        self.preference_model = preference_model
        self.maximum_prototype_cosine = maximum_prototype_cosine

    def catalogue_criteria(
        self,
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Compute criterion representations for every catalogue item."""

        feature_value_embeddings = self.item_encoder()
        return self.criteria_encoder(
            feature_value_embeddings,
            return_details=return_details,
        )

    def _criterion_usage_loss(
        self,
        criteria_details: Mapping[str, Any],
    ) -> Tensor:
        """Return the mean KL divergence from uniform prototype usage.

        For feature ``f``, usage is averaged uniformly over its feature-value
        vocabulary, not over item occurrences. Consequently, frequent values do
        not dominate this regularizer merely because they occur in more items.
        """

        assignments_by_feature = criteria_details.get(
            "assignments_by_feature"
        )
        if not isinstance(assignments_by_feature, Mapping):
            raise KeyError(
                "Criteria details must contain an "
                "'assignments_by_feature' mapping to compute loss_use."
            )

        feature_losses: list[Tensor] = []
        for feature, assignments in assignments_by_feature.items():
            if not isinstance(assignments, Tensor) or assignments.ndim != 2:
                raise ValueError(
                    f"Assignments for feature {feature!r} must be a "
                    "two-dimensional Tensor."
                )

            n_criteria = assignments.shape[-1]
            if n_criteria <= 0:
                raise ValueError(
                    f"Feature {feature!r} must contain at least one criterion."
                )

            marginal_usage = assignments.mean(dim=0)
            if n_criteria == 1:
                feature_losses.append(marginal_usage.new_zeros(()))
                continue

            eps = getattr(self.criteria_encoder, "eps", 1e-8)
            log_usage = marginal_usage.clamp_min(eps).log()
            kl_from_uniform = (
                marginal_usage * (log_usage + math.log(n_criteria))
            ).sum()
            feature_losses.append(kl_from_uniform)

        if not feature_losses:
            raise ValueError(
                "At least one feature assignment matrix is required to "
                "compute loss_use."
            )
        return torch.stack(feature_losses).mean()

    def _prototype_separation_loss(self) -> Tensor:
        """Delegate margin-based prototype separation to CriteriaEncoder."""

        separation_loss = getattr(
            self.criteria_encoder,
            "prototype_separation_loss",
            None,
        )
        if separation_loss is None or not callable(separation_loss):
            raise TypeError(
                "criteria_encoder must define prototype_separation_loss() "
                "to compute loss_sep."
            )
        return separation_loss(
            maximum_cosine=self.maximum_prototype_cosine
        )

    @staticmethod
    def _pair_sparsity_loss(
        preference_details: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor:
        """Return mean per-user L1 mass of singleton and pair coefficients.

        Missing or empty coefficient tensors contribute zero.
        Uses the normalized coefficients employed by the scorer.
        """
        loss = reference.new_zeros(())

        for key in (
            "singleton_coefficients",
            "interaction_coefficients",
        ):
            coefficients = preference_details.get(key)

            if coefficients is None:
                continue

            if not isinstance(coefficients, Tensor):
                raise TypeError(
                    f"'{key}' must be a Tensor when present."
                )

            if coefficients.numel() == 0:
                continue

            loss = loss + coefficients.abs().sum(dim=-1).mean()

        return loss
    # def _pair_sparsity_loss(
    #     preference_details: Mapping[str, Any],
    #     reference: Tensor,
    # ) -> Tensor:
    #     """Return mean per-user L1 mass of normalized pair coefficients.

    #     Additive preference models have no interaction coefficients, so their
    #     pairwise loss is the scalar zero. The loss is computed from normalized
    #     coefficients, matching the coefficients used by the scorer.
    #     """

    #     pair_coefficients = preference_details.get(
    #         "interaction_coefficients"
    #     )
    #     if pair_coefficients is None:
    #         return reference.new_zeros(())
    #     if not isinstance(pair_coefficients, Tensor):
    #         raise TypeError(
    #             "'interaction_coefficients' must be a Tensor when present."
    #         )
    #     if pair_coefficients.numel() == 0:
    #         return pair_coefficients.new_zeros(())
    #     return pair_coefficients.abs().sum(dim=-1).mean()

    def _regularization_losses(
        self,
        criteria_details: Mapping[str, Any],
        preference_details: Mapping[str, Any],
        reference: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the three unweighted losses used for model ablations."""

        return {
            "loss_use": self._criterion_usage_loss(criteria_details),
            "loss_sep": self._prototype_separation_loss(),
            "loss_pair": self._pair_sparsity_loss(
                preference_details,
                reference,
            ),
        }

    def forward(
        self,
        user_ids: Tensor,
        positive_item_ids: Tensor,
        negative_item_ids: Tensor | None = None,
        return_details: bool = False,
        return_regularization_losses: bool = False,
    ) -> Tensor | tuple[Any, ...]:
        """Score items and optionally return details and regularization losses.

        With ``return_regularization_losses=True``, the last returned object is
        a dictionary containing the unweighted scalar tensors ``loss_use``,
        ``loss_sep``, and ``loss_pair``. When ``return_details=True`` as well,
        that same dictionary also contains the pre-existing inspection details.
        The default return signatures are unchanged.
        """

        user_representations = self.user_encoder(user_ids)

        need_criteria_details = (
            return_details or return_regularization_losses
        )
        if need_criteria_details:
            all_item_criteria, criteria_details = self.catalogue_criteria(
                return_details=True
            )
        else:
            all_item_criteria = self.catalogue_criteria()

        positive_criteria = all_item_criteria[positive_item_ids]
        need_positive_preference_details = (
            return_details or return_regularization_losses
        )
        if need_positive_preference_details:
            positive_scores, positive_preference_details = self.preference_model(
                user_representations,
                positive_criteria,
                return_details=True,
            )
        else:
            positive_scores = self.preference_model(
                user_representations,
                positive_criteria,
            )

        auxiliary_outputs: dict[str, Any] = {}
        if return_details:
            auxiliary_outputs.update(
                {
                    "criteria": criteria_details,
                    "positive_preference": positive_preference_details,
                    "positive_criteria": positive_criteria,
                }
            )
        if return_regularization_losses:
            auxiliary_outputs.update(
                self._regularization_losses(
                    criteria_details,
                    positive_preference_details,
                    positive_scores,
                )
            )

        if negative_item_ids is None:
            if not auxiliary_outputs:
                return positive_scores
            return positive_scores, auxiliary_outputs

        negative_criteria = all_item_criteria[negative_item_ids]
        if return_details:
            negative_scores, negative_preference_details = self.preference_model(
                user_representations,
                negative_criteria,
                return_details=True,
            )
            auxiliary_outputs.update(
                {
                    "negative_preference": negative_preference_details,
                    "negative_criteria": negative_criteria,
                }
            )
        else:
            negative_scores = self.preference_model(
                user_representations,
                negative_criteria,
            )

        if not auxiliary_outputs:
            return positive_scores, negative_scores
        return positive_scores, negative_scores, auxiliary_outputs
