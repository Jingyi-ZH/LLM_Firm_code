"""Unified probability field visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from llm_belief.analysis.dimensionality import fit_transform
from llm_belief.models.scoring import score_profiles
from llm_belief.utils.paths import get_plot_path


@dataclass
class ProbabilityVizResult:
    reducer: Any
    embedding: np.ndarray
    probabilities: np.ndarray
    scores: np.ndarray
    representative_indices: Optional[np.ndarray] = None
    representative_coords: Optional[np.ndarray] = None
    representative_probs: Optional[np.ndarray] = None


def _require_module(module_name: str, install_hint: str) -> None:
    raise ImportError(f"Missing optional dependency '{module_name}'. {install_hint}")


def _as_array(X: Any) -> np.ndarray:
    if hasattr(X, "values"):
        return np.asarray(X.values)
    if hasattr(X, "detach"):
        return np.asarray(X.detach().cpu().numpy())
    return np.asarray(X)


def _split_test(
    X2: np.ndarray,
    probs: np.ndarray,
    test_idx: Optional[Sequence[int]],
    num_test: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Sequence[int]]:
    N = len(X2)
    if test_idx is None and num_test > 0:
        test_idx = list(range(N - num_test, N))
    test_idx = test_idx or []

    mask = np.zeros(N, dtype=bool)
    if test_idx:
        mask[np.asarray(test_idx, dtype=int)] = True

    X2_test = X2[mask] if test_idx else None
    probs_test = probs[mask] if test_idx else None
    X2_train = X2[~mask] if test_idx else X2
    probs_train = probs[~mask] if test_idx else probs
    return X2_train, X2_test, probs_train, probs_test, test_idx


def _build_surface(
    X2: np.ndarray, probs: np.ndarray, grid_res: int = 120, smooth_sigma: float = 0.0
):
    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
    except ImportError:
        _require_module("scipy", "Install with `pip install scipy`.")
    gx = np.linspace(X2[:, 0].min(), X2[:, 0].max(), grid_res)
    gy = np.linspace(X2[:, 1].min(), X2[:, 1].max(), grid_res)
    GX, GY = np.meshgrid(gx, gy)
    GZ = griddata(points=X2, values=probs, xi=(GX, GY), method="linear")
    if smooth_sigma and smooth_sigma > 0:
        GZ = gaussian_filter(GZ, sigma=float(smooth_sigma))
    return GX, GY, GZ


def _find_hotspot_representatives(
    X2: np.ndarray,
    probs: np.ndarray,
    method: str = "peaks",
    top_k: int = 10,
    grid_res: int = 200,
    smooth_sigma: float = 1.0,
    peak_min_dist: int = 5,
    quantile_thresh: float = 0.9,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter, maximum_filter, label
        from scipy.spatial import cKDTree
        from sklearn.cluster import DBSCAN, KMeans
    except ImportError:
        _require_module(
            "scipy/scikit-learn",
            "Install with `pip install scipy scikit-learn`.",
        )

    X2 = np.asarray(X2, dtype=float)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    if X2.shape[0] != probs.shape[0]:
        raise ValueError("X2 and probs must have the same length.")

    rng = np.random.RandomState(random_state)
    kdt = cKDTree(X2)

    if method == "peaks":
        gx = np.linspace(X2[:, 0].min(), X2[:, 0].max(), grid_res)
        gy = np.linspace(X2[:, 1].min(), X2[:, 1].max(), grid_res)
        GX, GY = np.meshgrid(gx, gy)
        GZ = griddata(points=X2, values=probs, xi=(GX, GY), method="linear")
        if smooth_sigma and smooth_sigma > 0:
            GZ = gaussian_filter(GZ, sigma=float(smooth_sigma))

        valid = np.isfinite(GZ)
        zvals = GZ[valid]
        if zvals.size == 0:
            return np.empty((0, 2)), np.array([], int), np.array([])

        thr = np.quantile(zvals, quantile_thresh)
        mask = valid & (GZ >= thr)

        neighborhood = maximum_filter(GZ, size=peak_min_dist, mode="nearest")
        peaks_mask = (GZ == neighborhood) & mask

        lbl, ncomp = label(peaks_mask)
        peak_coords = []
        for i in range(1, ncomp + 1):
            region = lbl == i
            if not np.any(region):
                continue
            zi = GZ.copy()
            zi[~region] = -np.inf
            flat = np.argmax(zi)
            yy, xx = np.unravel_index(flat, zi.shape)
            peak_coords.append((GX[yy, xx], GY[yy, xx], GZ[yy, xx]))

        peak_coords.sort(key=lambda t: t[2], reverse=True)
        peak_coords = peak_coords[:top_k]
        if not peak_coords:
            return np.empty((0, 2)), np.array([], int), np.array([])

        centers_2d = np.array([[px, py] for px, py, _ in peak_coords])
        _, idx = kdt.query(centers_2d, k=1)
        center_idx = idx.astype(int)
        center_prob = probs[center_idx]
        return centers_2d, center_idx, center_prob

    if method == "dbscan":
        thr = np.quantile(probs, quantile_thresh)
        sel = probs >= thr
        Xh = X2[sel]
        if Xh.shape[0] < max(dbscan_min_samples, 5):
            return np.empty((0, 2)), np.array([], int), np.array([])

        if dbscan_eps is None:
            sample = Xh[rng.choice(len(Xh), size=min(500, len(Xh)), replace=False)]
            tree = cKDTree(sample)
            d, _ = tree.query(sample, k=2)
            base = np.median(d[:, 1])
            dbscan_eps = max(1e-6, base)

        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(Xh)
        labels = db.labels_
        centers_2d, center_idx, center_prob = [], [], []
        for lb in set(labels):
            if lb == -1:
                continue
            cluster_pts = Xh[labels == lb]
            c = cluster_pts.mean(axis=0)
            _, i0 = kdt.query(c, k=1)
            centers_2d.append(c)
            center_idx.append(int(i0))
            center_prob.append(probs[i0])

        return np.array(centers_2d), np.array(center_idx, int), np.array(center_prob)

    if method == "wkmeans":
        w = probs - probs.min()
        if w.sum() == 0:
            w = np.ones_like(w)

        k = max(1, top_k)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X2, sample_weight=w)
        centers_2d = km.cluster_centers_
        _, idx = kdt.query(centers_2d, k=1)
        center_idx = idx.astype(int)
        center_prob = probs[center_idx]
        return centers_2d, center_idx, center_prob

    raise ValueError("represent_method must be one of {'peaks','dbscan','wkmeans'}.")


def visualize_probability_distribution(
    model=None,
    X_full,
    T: float = 1.0,
    *,
    method: str = "pca",
    dims: int = 3,
    test_idx: Optional[Sequence[int]] = None,
    num_test: int = 0,
    fig_name: Optional[str] = None,
    show: bool = True,
    grid_res: int = 120,
    random_state: int = 0,
    reducer_kwargs: Optional[dict] = None,
    scores: Optional[np.ndarray] = None,
    visualize: Optional[bool] = None,
    plot_kind: Optional[str] = None,
    smooth_sigma: float = 0.0,
    pca_result: Optional[Any] = None,
    fa_result: Optional[Any] = None,
    kpca_result: Optional[Any] = None,
    represent_method: Optional[str] = None,
    top_k: int = 10,
    quantile_thresh: float = 0.9,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 5,
    title: Optional[str] = None,
    score_model: Optional[str] = None,
) -> ProbabilityVizResult:
    """Visualize probability field with a unified interface.

    Args:
        model: Trained scoring model.
        X_full: Features (DataFrame or ndarray).
        T: Temperature for score normalization.
        method: Dimensionality method: pca | tsne | fa | kpca.
        dims: 3 for 3D surface, 2 for 2D heatmap.
        test_idx: Optional indices to highlight as test points.
        num_test: If test_idx not provided, use last N points as test.
        fig_name: Output filename under plot/ (optional).
        show: Whether to display the plot.
        grid_res: Resolution for grid interpolation.
        random_state: Random state for reducers.
        reducer_kwargs: Extra kwargs passed to reducer.
    """
    if plot_kind is not None:
        plot_kind = str(plot_kind).strip().lower()
        if plot_kind == "heatmap":
            dims = 2
        elif plot_kind == "3d":
            dims = 3
        else:
            raise ValueError("plot_kind must be '3d' or 'heatmap'.")

    if dims not in {2, 3}:
        raise ValueError("dims must be 2 or 3.")

    X_arr = _as_array(X_full)
    if scores is None:
        if model is None:
            raise ValueError("Provide either model or scores.")
        scores, probs = score_profiles(model, X_arr, T)
    else:
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(X_arr):
            raise ValueError("scores must match number of samples in X_full.")
        scores_stable = scores - scores.max()
        w = np.exp(scores_stable)
        probs = w / w.sum()

    reducer_kwargs = reducer_kwargs or {}
    if pca_result or fa_result or kpca_result:
        if pca_result is not None:
            X2_train = np.asarray(pca_result.scores_train_)
            X2_test = np.asarray(getattr(pca_result, "scores_test_", None))
            reducer = pca_result
        elif fa_result is not None:
            X2_train = np.asarray(fa_result.scores_rot_train_)
            X2_test = np.asarray(getattr(fa_result, "scores_rot_test_", None))
            reducer = fa_result
        else:
            X2_train = np.asarray(kpca_result.scores_train_)
            X2_test = np.asarray(getattr(kpca_result, "scores_test_", None))
            reducer = kpca_result

        X2 = (
            np.concatenate([X2_train, X2_test], axis=0)
            if X2_test is not None
            else X2_train
        )
        result = type("ReducerResult", (), {"model": reducer})
    else:
        result = fit_transform(
            X_arr,
            method=method,
            n_components=2,
            random_state=random_state,
            **reducer_kwargs,
        )
        X2 = np.asarray(result.embedding)

    X2_train, X2_test, probs_train, probs_test, test_idx = _split_test(
        X2, probs, test_idx, num_test
    )

    if visualize is not None:
        show = visualize

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _require_module("matplotlib", "Install with `pip install matplotlib`.")

    if dims == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        if X2_test is not None:
            ax.scatter(
                X2_test[:, 0],
                X2_test[:, 1],
                probs_test,
                s=30,
                c="red",
                marker="^",
                alpha=0.9,
                label="Test",
            )
            for xi, yi, zi, mn in zip(
                X2_test[:, 0], X2_test[:, 1], probs_test, test_idx
            ):
                ax.text(xi, yi, zi, str(mn), color="black", fontsize=8)

        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("P(sample)")
        title_text = title or f"Probability field ({method.upper()}, 3D)"
        if score_model:
            title_text += f"\\n{score_model}"
        ax.set_title(title_text)
        if X2_test is not None:
            ax.legend()

        GX, GY, GZ = _build_surface(
            X2_train, probs_train, grid_res=grid_res, smooth_sigma=smooth_sigma
        )
        ax.plot_surface(
            GX, GY, np.nan_to_num(GZ, nan=np.nan), alpha=0.4, linewidth=0
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        GX, GY, GZ = _build_surface(
            X2_train, probs_train, grid_res=grid_res, smooth_sigma=smooth_sigma
        )
        cf = ax.contourf(GX, GY, GZ, levels=30, cmap="viridis")
        fig.colorbar(cf, ax=ax, label="P(sample)")
        if X2_test is not None:
            ax.scatter(
                X2_test[:, 0],
                X2_test[:, 1],
                s=40,
                c="red",
                marker="^",
                alpha=0.9,
                label="Test",
            )
            for xi, yi, mn in zip(X2_test[:, 0], X2_test[:, 1], test_idx):
                ax.text(xi, yi, str(mn), color="black", fontsize=8)
            ax.legend()
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        title_text = title or f"Probability field ({method.upper()}, 2D)"
        if score_model:
            title_text += f"\\n{score_model}"
        ax.set_title(title_text)

    if fig_name:
        out_path = get_plot_path(fig_name)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)

    if show:
        plt.show()
    else:
        plt.close(fig)

    rep_coords = rep_idx = rep_probs = None
    if represent_method and str(represent_method).lower() != "none":
        rep_coords, rep_idx, rep_probs = _find_hotspot_representatives(
            X2=X2,
            probs=probs,
            method=str(represent_method).lower(),
            top_k=top_k,
            grid_res=grid_res,
            smooth_sigma=smooth_sigma,
            quantile_thresh=quantile_thresh,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            random_state=random_state,
        )

    return ProbabilityVizResult(
        reducer=result.model,
        embedding=X2,
        probabilities=probs,
        scores=scores,
        representative_indices=rep_idx,
        representative_coords=rep_coords,
        representative_probs=rep_probs,
    )
