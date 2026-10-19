import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import torch

def plot_user_weights(
    user_singletons,
    user_ids=None,
    n_regions=None,
    feature_names=None,
    figsize=None,
):
    """Plot users × regions, optionally grouping regions by feature."""
    if hasattr(user_singletons, "detach"):
        weights = user_singletons.detach().float().cpu().numpy()
    else:
        weights = np.asarray(user_singletons)

    if weights.ndim != 2 or 0 in weights.shape:
        raise ValueError("user_singletons must be a nonempty 2D array.")

    n_users, n_columns = weights.shape

    if user_ids is None:
        ids = np.arange(n_users)
    else:
        if hasattr(user_ids, "detach"):
            user_ids = user_ids.detach().cpu().numpy()
        ids = np.atleast_1d(user_ids)

    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError("user_ids must contain integer row indices.")
    if np.any(ids < 0) or np.any(ids >= n_users):
        raise ValueError(f"user_ids must be between 0 and {n_users - 1}.")

    if n_regions is None:
        if feature_names is not None:
            raise ValueError("Provide n_regions when using feature_names.")
        labels = [f"{r}" for r in range(n_columns)]
    else:
        if n_regions < 1 or n_columns % n_regions:
            raise ValueError("n_regions must divide the number of columns.")

        n_features = n_columns // n_regions
        names = list(feature_names) if feature_names is not None else [
            f"{f}" for f in range(n_features)
        ]

        if len(names) != n_features:
            raise ValueError("feature_names must have one name per feature.")

        labels = [f"{f}:{r}" for f in names for r in range(n_regions)]

    # figsize = (max(9, 0.45 * n_columns), max(3, 0.25 * len(ids)))
    figsize = (12, 8)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        weights[ids],
        ax=ax,
        cmap="viridis",
        vmin=0,  vmax=1,
        # vmax=float(weights.max()),
        xticklabels=labels,
        yticklabels=False,
        cbar_kws={"label": "Weight", "shrink": 0.85},
    )

    # Label only the requested user IDs, wherever they appear.
    tick_rows = np.flatnonzero(np.isin(ids, [0, 25, 50, 75]))
    ax.set_yticks(tick_rows + 0.5)  # Center labels within rows.
    ax.set_yticklabels(ids[tick_rows], rotation=0, fontsize=10)

    # Separate feature blocks.
    if n_regions is not None:
        for boundary in range(n_regions, n_columns, n_regions):
            ax.axvline(boundary, color="white", linewidth=1)

    # ax.set(xlabel="All feature prototypes", ylabel="User ID")
    ax.set_xlabel("All feature prototypes", fontsize=15)
    ax.set_ylabel("User ID", fontsize=15)
    ax.tick_params(axis="x", labelrotation=90, labelsize=14)
    ax.tick_params(axis="y", labelsize=18)
    fig.suptitle("CHEERS-2: Heatmap of learned singleton weights", fontsize=16)
    fig.tight_layout()

    return fig, ax


def plot_embeddings_tsne(
    prototype_embeddings,
    feature_value_embeddings,
    feature_value_names=None,
    feature_name="a",
    perplexity=5,
    metric="euclidean",
    random_state=42,
    figsize=(8, 6),
):
    def as_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    prototypes = as_numpy(prototype_embeddings)
    values = as_numpy(feature_value_embeddings)

    if prototypes.ndim != 2 or values.ndim != 2:
        raise ValueError("Both embedding matrices must be 2D.")
    if prototypes.shape[1] != values.shape[1]:
        raise ValueError("Both matrices must have the same embedding dimension.")
    if 0 in prototypes.shape or 0 in values.shape:
        raise ValueError("Both embedding matrices must be nonempty.")
    if perplexity <= 0:
        raise ValueError("perplexity must be positive.")

    n_values = len(values)
    names = (
        [f"{feature_name}{i}" for i in range(n_values)]
        if feature_value_names is None
        else list(feature_value_names)
    )
    if len(names) != n_values:
        raise ValueError("Provide one name per feature-value row.")

    # Fit a single projection containing both types of embeddings.
    combined = np.vstack([values, prototypes])

    coordinates = TSNE(
        n_components=2,
        perplexity=min(float(perplexity), len(combined) - 1),
        metric=metric,
        init="pca" if combined.shape[1] >= 2 else "random",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(combined)

    values_2d = coordinates[:n_values]
    prototypes_2d = coordinates[n_values:]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        *values_2d.T,
        c="tab:blue", s=50, label="Feature values",
    )
    ax.scatter(
        *prototypes_2d.T,
        c="red", s=120, edgecolors="black", linewidths=0.5,
        label="Prototypes", zorder=3,
    )

    for name, xy in zip(names, values_2d):
        ax.annotate(
            str(name), xy,
            xytext=(-15, 7), textcoords="offset points", fontsize=15,
        )

    for i, xy in enumerate(prototypes_2d):
        ax.annotate(
            f"{i}", xy,
            xytext=(5, 5), textcoords="offset points",
            fontsize=17, color="red", fontweight="bold",
        )

    # ax.set(
    #     title="t-SNE: feature values and prototypes",
    #     xlabel="t-SNE 1",
    #     ylabel="t-SNE 2",
    # )

    fig.suptitle(f'CHEERS-2 - Feature {feature_name}\nt-SNE: feature values and prototypes', fontsize=20)
    plt.xlabel('t-SNE 1', fontsize=18)
    plt.ylabel('t-SNE 2', fontsize=18)
    ax.legend(fontsize=18)
    fig.tight_layout()

    return fig, ax