"""Unified dimensionality reduction interface for analysis notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DimensionalityResult:
    """Container for dimensionality reduction results."""

    embedding: np.ndarray
    model: Any
    method: str
    params: Dict[str, Any]


def _require_module(module_name: str, install_hint: str) -> None:
    raise ImportError(
        f"Missing optional dependency '{module_name}'. {install_hint}"
    )


def _build_pca(n_components: int, random_state: Optional[int], **kwargs: Any) -> Any:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        _require_module("scikit-learn", "Install with `pip install scikit-learn`.")
    if random_state is not None and "random_state" not in kwargs:
        kwargs["random_state"] = random_state
    return PCA(n_components=n_components, **kwargs)


def _build_kernel_pca(
    n_components: int, random_state: Optional[int], **kwargs: Any
) -> Any:
    try:
        from sklearn.decomposition import KernelPCA
    except ImportError as exc:
        _require_module("scikit-learn", "Install with `pip install scikit-learn`.")
    if random_state is not None and "random_state" not in kwargs:
        kwargs["random_state"] = random_state
    return KernelPCA(n_components=n_components, **kwargs)


def _build_tsne(n_components: int, random_state: Optional[int], **kwargs: Any) -> Any:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        _require_module("scikit-learn", "Install with `pip install scikit-learn`.")
    if random_state is not None and "random_state" not in kwargs:
        kwargs["random_state"] = random_state
    return TSNE(n_components=n_components, **kwargs)


def _build_fa_varimax(
    n_components: int, random_state: Optional[int], **kwargs: Any
) -> Any:
    try:
        from factor_analyzer import FactorAnalyzer
    except ImportError as exc:
        _require_module(
            "factor_analyzer",
            "Install with `pip install factor-analyzer`.",
        )
    if "rotation" not in kwargs:
        kwargs["rotation"] = "varimax"
    return FactorAnalyzer(n_factors=n_components, **kwargs)


def build_reducer(
    method: str,
    n_components: int = 2,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build a dimensionality reducer instance based on method name."""
    method_key = method.lower().strip()
    builders = {
        "pca": _build_pca,
        "kernel_pca": _build_kernel_pca,
        "kpca": _build_kernel_pca,
        "tsne": _build_tsne,
        "t-sne": _build_tsne,
        "fa_varimax": _build_fa_varimax,
        "fa": _build_fa_varimax,
    }
    if method_key not in builders:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Use one of: fa_varimax, pca, kernel_pca, tsne."
        )
    return builders[method_key](n_components, random_state, **kwargs)


def fit_transform(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> DimensionalityResult:
    """Fit and transform data using the selected dimensionality reducer."""
    reducer = build_reducer(
        method=method,
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )
    embedding = reducer.fit_transform(X)
    params = {"n_components": n_components, "random_state": random_state, **kwargs}
    return DimensionalityResult(
        embedding=np.asarray(embedding),
        model=reducer,
        method=method,
        params=params,
    )


def plot_embedding(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Quick scatter plot for 2D embeddings."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        _require_module("matplotlib", "Install with `pip install matplotlib`.")

    if ax is None:
        _, ax = plt.subplots()

    if embedding.shape[1] < 2:
        raise ValueError("Embedding must be at least 2D for plotting.")

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], **kwargs)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    return ax
