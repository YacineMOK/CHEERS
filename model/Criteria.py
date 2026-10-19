"""Prototype criteria and sparse item composition for CHEERS.

For feature f, the module computes

    Q_f = softmax(cosine(E_f, P_f) / temperature_f)
    X_f = A_f Q_f

where E_f contains all feature-value embeddings, P_f contains criterion
prototypes, and A_f is the loader's row-normalized item-value incidence matrix.
The concatenation of all X_f matrices represents the complete item catalogue.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NumberByFeature = int | float | Mapping[str, int | float]


def _expand_feature_configuration(
    value: NumberByFeature,
    feature_names: tuple[str, ...],
    name: str,
) -> dict[str, int | float]:
    """Expand a scalar or validate a complete per-feature configuration."""

    if isinstance(value, Mapping):
        missing = set(feature_names) - set(value)
        extra = set(value) - set(feature_names)
        if missing or extra:
            raise ValueError(
                f"{name} must define exactly the configured features; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )
        return {feature: value[feature] for feature in feature_names}
    return {feature: value for feature in feature_names}


class CriteriaEncoder(nn.Module):
    """Build criterion representations for every item in the catalogue.

    Parameters
    ----------
    feature_dims:
        Embedding dimension of each feature-value vocabulary.
    n_criteria:
        Number of criterion prototypes per feature.
    normalized_incidence_matrices:
        Row-normalized item-by-feature-value matrices produced by
        Loader.get_normalized_incidence_matrices(). They may be sparse or dense.
        Missing feature values correspond to all-zero rows and therefore
        produce all-zero criterion blocks.
    temperature:
        Positive softmax temperature, shared or feature-specific.
    """

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        n_criteria: int | Mapping[str, int],
        normalized_incidence_matrices: Mapping[str, Tensor], # "Feature" -> "N-by-V_f" row-normalized incidence matrix
        temperature: float | Mapping[str, float] = 1,
        learnable_temperature: bool = False,
        min_temperature: float = 0.03,
        max_temperature: float = 1.0,
        prototype_init: str = "auto",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("feature_dims must contain at least one feature.")
        if any("." in feature for feature in feature_dims):
            raise ValueError(
                "Feature names cannot contain '.', because they are used as "
                "PyTorch parameter keys."
            )
        if prototype_init not in {"auto", "orthogonal", "normal"}:
            raise ValueError(
                "prototype_init must be 'auto', 'orthogonal', or "
                f"'normal'; received {prototype_init!r}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, received {eps}.")
        if learnable_temperature and not (
            0 < min_temperature < max_temperature
        ):
            raise ValueError(
                "Learnable temperatures require "
                "0 < min_temperature < max_temperature."
            )

        self.feature_names = tuple(feature_dims)
        self.feature_dims = {
            feature: int(feature_dims[feature]) for feature in self.feature_names
        }
        if any(dimension <= 0 for dimension in self.feature_dims.values()):
            raise ValueError("Every feature embedding dimension must be positive.")

        self._validate_mapping_keys(
            normalized_incidence_matrices,
            "normalized incidence matrices",
        )
        self._incidence_buffer_names: dict[str, str] = {}
        self.n_feature_values_by_feature: dict[str, int] = {}
        expected_n_items: int | None = None

        for index, feature in enumerate(self.feature_names):
            incidence = normalized_incidence_matrices[feature]
            if incidence.ndim != 2:
                raise ValueError(
                    f"Feature {feature!r} incidence matrix must be two-dimensional."
                )
            if incidence.layout not in {
                torch.strided,
                torch.sparse_coo,
                torch.sparse_csr,
            }:
                raise ValueError(
                    f"Feature {feature!r} uses unsupported layout "
                    f"{incidence.layout}."
                )
            if incidence.layout == torch.sparse_coo:
                incidence = incidence.coalesce()
            if not torch.is_floating_point(incidence):
                incidence = incidence.float()

            n_items, n_values = incidence.shape
            if n_values <= 0:
                raise ValueError(
                    f"Feature {feature!r} must contain at least one value."
                )
            if expected_n_items is None:
                expected_n_items = n_items
            elif n_items != expected_n_items:
                raise ValueError(
                    "All incidence matrices must contain the same number of "
                    f"items; expected {expected_n_items}, but feature "
                    f"{feature!r} contains {n_items}."
                )

            buffer_name = f"_incidence_{index}"
            self.register_buffer(
                buffer_name,
                incidence.detach(),
                persistent=False,
            )
            self._incidence_buffer_names[feature] = buffer_name
            self.n_feature_values_by_feature[feature] = n_values

        if expected_n_items is None:
            raise ValueError("No incidence matrices were provided.")
        self.n_items = expected_n_items

        criteria_config = _expand_feature_configuration(
            n_criteria,
            self.feature_names,
            "n_criteria",
        )
        self.n_criteria_by_feature = {
            feature: int(criteria_config[feature])
            for feature in self.feature_names
        }
        if any(count <= 0 for count in self.n_criteria_by_feature.values()):
            raise ValueError("Every feature must have at least one criterion.")

        temperature_config = _expand_feature_configuration(
            temperature,
            self.feature_names,
            "temperature",
        )
        initial_temperatures = torch.tensor(
            [
                float(temperature_config[feature])
                for feature in self.feature_names
            ],
            dtype=torch.float32,
        )
        if torch.any(initial_temperatures <= 0):
            raise ValueError("Every temperature must be strictly positive.")

        self.learnable_temperature = learnable_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.prototype_init = prototype_init
        self.eps = eps
        self._feature_to_index = {
            feature: index
            for index, feature in enumerate(self.feature_names)
        }

        self.prototypes = nn.ParameterDict(
            {
                feature: nn.Parameter(
                    torch.empty(
                        self.n_criteria_by_feature[feature],
                        self.feature_dims[feature],
                    )
                )
                for feature in self.feature_names
            }
        )

        if learnable_temperature:
            if torch.any(initial_temperatures <= min_temperature) or torch.any(
                initial_temperatures >= max_temperature
            ):
                raise ValueError(
                    "Every learnable initial temperature must lie strictly "
                    "between min_temperature and max_temperature."
                )
            scaled = (
                (initial_temperatures - min_temperature)
                / (max_temperature - min_temperature)
            )
            self.raw_temperatures = nn.Parameter(torch.logit(scaled))
            self.register_buffer("fixed_temperatures", None, persistent=False)
        else:
            self.register_parameter("raw_temperatures", None)
            self.register_buffer(
                "fixed_temperatures",
                initial_temperatures,
                persistent=True,
            )

        criterion_slices: dict[str, slice] = {}
        start = 0
        for feature in self.feature_names:
            stop = start + self.n_criteria_by_feature[feature]
            criterion_slices[feature] = slice(start, stop)
            start = stop
        self.criterion_slices = criterion_slices
        self.n_criteria_total = start

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize distinct prototype directions."""

        with torch.no_grad():
            for feature in self.feature_names:
                prototypes = self.prototypes[feature]
                n_criteria, dimension = prototypes.shape
                use_orthogonal = self.prototype_init == "orthogonal" or (
                    self.prototype_init == "auto" and n_criteria <= dimension
                )

                if use_orthogonal:
                    if n_criteria > dimension:
                        raise ValueError(
                            "Row-orthogonal initialization requires "
                            f"K_f <= d_f, but feature {feature!r} has "
                            f"K_f={n_criteria} and d_f={dimension}."
                        )
                    nn.init.orthogonal_(prototypes)
                else:
                    nn.init.normal_(
                        prototypes,
                        mean=0.0,
                        std=1.0 / math.sqrt(dimension),
                    )
                    prototypes.copy_(
                        F.normalize(prototypes, p=2, dim=-1, eps=self.eps)
                    )

    def temperatures(self) -> Tensor:
        """Return current per-feature temperatures in feature order."""

        if self.raw_temperatures is None:
            return self.fixed_temperatures
        scale = torch.sigmoid(self.raw_temperatures)
        return self.min_temperature + (
            self.max_temperature - self.min_temperature
        ) * scale

    def temperature_for(self, feature: str) -> Tensor:
        """Return one feature's temperature."""

        self._validate_feature(feature)
        return self.temperatures()[self._feature_to_index[feature]]

    def incidence_matrix(self, feature: str) -> Tensor:
        """Return one fixed normalized item-value incidence matrix."""

        self._validate_feature(feature)
        return getattr(self, self._incidence_buffer_names[feature])

    def normalized_prototypes(self, feature: str) -> Tensor:
        """Return prototypes projected onto the unit sphere."""

        self._validate_feature(feature)
        return F.normalize(
            self.prototypes[feature],
            p=2,
            dim=-1,
            eps=self.eps,
        )

    def assign_feature_values(
        self,
        feature: str,
        value_embeddings: Tensor,
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Convert all V_f feature values into K_f criterion memberships."""

        self._validate_feature(feature)
        expected_shape = (
            self.n_feature_values_by_feature[feature],
            self.feature_dims[feature],
        )
        if value_embeddings.shape != expected_shape:
            raise ValueError(
                f"Feature {feature!r} expects embedding matrix shape "
                f"{expected_shape}, received {tuple(value_embeddings.shape)}."
            )

        normalized_values = F.normalize(
            value_embeddings,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        prototypes = self.normalized_prototypes(feature)
        cosine_similarity = normalized_values @ prototypes.transpose(0, 1)
        logits = cosine_similarity / self.temperature_for(feature)
        assignments = torch.softmax(logits, dim=-1)

        if return_logits:
            return assignments, logits
        return assignments

    def compose_items(self, feature: str, assignments: Tensor) -> Tensor:
        """Compute X_f = A_f Q_f for the complete item catalogue."""

        self._validate_feature(feature)
        expected_shape = (
            self.n_feature_values_by_feature[feature],
            self.n_criteria_by_feature[feature],
        )
        if assignments.shape != expected_shape:
            raise ValueError(
                f"Feature {feature!r} expects assignment matrix shape "
                f"{expected_shape}, received {tuple(assignments.shape)}."
            )

        incidence = self.incidence_matrix(feature)
        if incidence.device != assignments.device:
            raise RuntimeError(
                "ItemEncoder and CriteriaEncoder must be on the same device."
            )
        if incidence.dtype != assignments.dtype:
            incidence = incidence.to(dtype=assignments.dtype)

        if incidence.layout == torch.strided:
            return incidence @ assignments
        return torch.sparse.mm(incidence, assignments)

    def forward(
        self,
        feature_value_embeddings: Mapping[str, Tensor],
        return_details: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Return the N-by-K criterion representation of the full catalogue."""

        self._validate_mapping_keys(
            feature_value_embeddings,
            "feature-value embeddings",
        )

        criteria_by_feature: dict[str, Tensor] = {}
        assignments_by_feature: dict[str, Tensor] = {}
        logits_by_feature: dict[str, Tensor] = {}
        availability_by_feature: dict[str, Tensor] = {}

        for feature in self.feature_names:
            assignments, logits = self.assign_feature_values(
                feature,
                feature_value_embeddings[feature],
                return_logits=True,
            )
            criteria_block = self.compose_items(feature, assignments)

            criteria_by_feature[feature] = criteria_block
            assignments_by_feature[feature] = assignments
            logits_by_feature[feature] = logits
            availability_by_feature[feature] = (
                criteria_block.sum(dim=-1) > self.eps
            )

        criteria = torch.cat(
            [
                criteria_by_feature[feature]
                for feature in self.feature_names
            ],
            dim=-1,
        )
        if not return_details:
            return criteria

        details: dict[str, Any] = {
            "criteria_by_feature": criteria_by_feature,
            "assignments_by_feature": assignments_by_feature,
            "logits_by_feature": logits_by_feature,
            "availability_by_feature": availability_by_feature,
            "temperatures": {
                feature: self.temperature_for(feature)
                for feature in self.feature_names
            },
            "criterion_slices": self.criterion_slices,
        }
        return criteria, details

    def prototype_separation_loss(self, maximum_cosine: float = 0.0) -> Tensor:
        """Penalize prototype pairs that are more similar than a margin."""

        losses: list[Tensor] = []
        for feature in self.feature_names:
            prototypes = self.normalized_prototypes(feature)
            n_criteria = prototypes.shape[0]
            if n_criteria < 2:
                continue
            cosine = prototypes @ prototypes.transpose(0, 1)
            upper = torch.triu_indices(
                n_criteria,
                n_criteria,
                offset=1,
                device=cosine.device,
            )
            excessive_similarity = F.relu(
                cosine[upper[0], upper[1]] - maximum_cosine
            )
            losses.append(excessive_similarity.square().mean())

        if losses:
            return torch.stack(losses).mean()
        return next(iter(self.prototypes.values())).new_zeros(())

    def assignment_diagnostics(
        self,
        assignments_by_feature: Mapping[str, Tensor],
    ) -> dict[str, dict[str, Tensor]]:
        """Summarize prototype use over every feature-value vocabulary."""

        self._validate_mapping_keys(assignments_by_feature, "assignments")
        diagnostics: dict[str, dict[str, Tensor]] = {}

        for feature in self.feature_names:
            assignments = assignments_by_feature[feature]
            expected_shape = (
                self.n_feature_values_by_feature[feature],
                self.n_criteria_by_feature[feature],
            )
            if assignments.shape != expected_shape:
                raise ValueError(
                    f"Feature {feature!r} expects assignments with shape "
                    f"{expected_shape}, received {tuple(assignments.shape)}."
                )

            marginal_usage = assignments.mean(dim=0)
            value_entropy = -(
                assignments.clamp_min(self.eps)
                * assignments.clamp_min(self.eps).log()
            ).sum(dim=-1)
            conditional_entropy = value_entropy.mean()
            marginal_entropy = -(
                marginal_usage.clamp_min(self.eps)
                * marginal_usage.clamp_min(self.eps).log()
            ).sum()

            n_criteria = self.n_criteria_by_feature[feature]
            entropy_scale = math.log(n_criteria) if n_criteria > 1 else 1.0
            diagnostics[feature] = {
                "marginal_usage": marginal_usage,
                "marginal_entropy": marginal_entropy / entropy_scale,
                "conditional_entropy": conditional_entropy / entropy_scale,
                "effective_criteria": marginal_entropy.exp(),
                "mean_max_assignment": assignments.max(dim=-1).values.mean(),
            }

        return diagnostics

    def prototype_cosine_matrices(self) -> dict[str, Tensor]:
        """Return prototype cosine-similarity matrices for inspection."""

        return {
            feature: self.normalized_prototypes(feature)
            @ self.normalized_prototypes(feature).transpose(0, 1)
            for feature in self.feature_names
        }

    def _validate_feature(self, feature: str) -> None:
        if feature not in self._feature_to_index:
            raise KeyError(
                f"Unknown feature {feature!r}; expected one of "
                f"{self.feature_names}."
            )

    def _validate_mapping_keys(
        self,
        mapping: Mapping[str, Any],
        name: str,
    ) -> None:
        missing = set(self.feature_names) - set(mapping)
        extra = set(mapping) - set(self.feature_names)
        if missing or extra:
            raise ValueError(
                f"{name} must define exactly the configured features; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )

    def extra_repr(self) -> str:
        return (
            f"features={self.feature_names}, n_items={self.n_items}, "
            f"n_criteria_total={self.n_criteria_total}, "
            f"learnable_temperature={self.learnable_temperature}, "
            f"prototype_init={self.prototype_init!r}"
        )
