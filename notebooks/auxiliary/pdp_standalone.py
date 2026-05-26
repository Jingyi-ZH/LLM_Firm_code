"""Standalone partial dependence plot utilities.

These functions work with the score models used in this project: a trained
PyTorch model maps one preprocessed profile row to one scalar score.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import torch

class _QuietTuple(tuple):
    def __new__(cls, *items):
        return super().__new__(cls, items)

    def __repr__(self) -> str:
        return ""

    def _repr_html_(self) -> str:
        return ""


class _QuietDict(dict):
    def __repr__(self) -> str:
        return ""

    def _repr_html_(self) -> str:
        return ""


def _as_dataframe(X: pd.DataFrame, feature_cols: Sequence[str] | None = None) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X_raw must be a pandas DataFrame with original feature columns.")
    X = X.copy()
    return X.loc[:, list(feature_cols)].copy() if feature_cols is not None else X


def _score_model(model: torch.nn.Module, X_proc, batch_size: int = 4096) -> np.ndarray:
    """Return model scores as a 1D numpy array."""
    if isinstance(X_proc, pd.DataFrame):
        X_proc = torch.from_numpy(X_proc.values).float()
    elif isinstance(X_proc, np.ndarray):
        X_proc = torch.from_numpy(X_proc).float()
    elif not torch.is_tensor(X_proc):
        X_proc = torch.as_tensor(X_proc).float()

    device = next(model.parameters()).device
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(X_proc), batch_size):
            xb = X_proc[start : start + batch_size].to(device)
            scores.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(scores)


def _grid_values(
    X_raw: pd.DataFrame,
    feature: str,
    *,
    feature_types: Mapping[str, str] | None = None,
    categories_map: Mapping[str, Sequence] | None = None,
    grid_resolution: int = 25,
    percentiles: tuple[float, float] = (0.05, 0.95),
) -> np.ndarray:
    feature_types = feature_types or {}
    categories_map = categories_map or {}

    if feature_types.get(feature) == "categorical":
        if feature in categories_map:
            return np.asarray(list(categories_map[feature]), dtype=object)
        return np.asarray(pd.unique(X_raw[feature].dropna()), dtype=object)

    values = pd.to_numeric(X_raw[feature], errors="coerce").dropna().to_numpy()
    lo, hi = np.quantile(values, percentiles)
    return np.linspace(lo, hi, grid_resolution)


def _safe_model_name(model: torch.nn.Module) -> str:
    try:
        return model._get_name()
    except Exception:
        return model.__class__.__name__


def _make_vivid_colors(n: int):
    if n <= 0:
        return []
    palettes = ["tab10", "Set1", "Dark2", "Set2", "Set3", "tab20"]
    colors = []
    palette_idx = 0
    while len(colors) < n:
        cmap = plt.get_cmap(palettes[palette_idx % len(palettes)])
        for color_idx in range(getattr(cmap, "N", 10)):
            colors.append(cmap(color_idx))
            if len(colors) >= n:
                break
        palette_idx += 1
    return colors[:n]


def _bright_palette():
    return [
        "#E41A1C",
        "#377EB8",
        "#FF7F00",
        "#984EA3",
        "#1FA3A3",
        "#F781BF",
        "#A65628",
        "#BFBFBF",
    ]


def _feature_kind(feature: str, feature_types: Mapping[str, str] | None) -> str:
    return "categorical" if (feature_types or {}).get(feature) == "categorical" else "numeric"


def _draw_pdp1d_axis(
    ax: plt.Axes,
    pdp: pd.DataFrame,
    feature: str,
    *,
    is_cat: bool,
    y_min: float,
    y_max: float,
    title_fontsize: int = 14,
    label_fontsize: int | None = None,
    tick_fontsize: int | None = None,
) -> dict[str, float]:
    cat_h = {}
    if is_cat:
        x = np.arange(len(pdp))
        bars = ax.bar(x, pdp["pdp"], alpha=0.15)
        for bar in bars:
            h = bar.get_height()
            xc = bar.get_x() + bar.get_width() / 2
            cat_h[str(pdp.iloc[int(round(xc))][feature])] = h
            half_width = bar.get_width() / 2
            ax.plot([xc - half_width, xc + half_width], [h, h], lw=2, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pdp[feature]], rotation=0, ha="center")
        ax.set_xlabel("Category", fontsize=label_fontsize)
    else:
        x = pdp[feature].astype(float).to_numpy()
        y = pdp["pdp"].astype(float).to_numpy()
        ax.plot(x, y, lw=2)
        ax.fill_between(x, y, alpha=0.15)
        ax.set_xlabel(f"{feature.capitalize()}", fontsize=label_fontsize)

    ax.set_title(f"{feature.capitalize()}", loc="left", fontsize=title_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.5)
    return cat_h


def _overlay_pdp1d_marks(
    ax: plt.Axes,
    pdp: pd.DataFrame,
    X_mark: pd.DataFrame | None,
    feature: str,
    *,
    is_cat: bool,
    row_colors,
    marker_size: int,
    marker_alpha: float,
    jitter: float,
    y_min: float,
    y_max: float,
    cat_h: Mapping[str, float] | None = None,
    annotate: bool = False,
) -> None:
    if X_mark is None or X_mark.empty or feature not in X_mark.columns:
        return

    labels = X_mark["_pdp_profile_label"] if "_pdp_profile_label" in X_mark.columns else X_mark.index.to_series().astype(str)
    if is_cat:
        cat_to_x = {str(value): i for i, value in enumerate(pdp[feature])}
        for rid, (_, row) in enumerate(X_mark.iterrows()):
            key = str(row[feature])
            if key not in cat_to_x:
                continue
            xi = cat_to_x[key] + np.random.uniform(-jitter, jitter)
            yi = (cat_h or {}).get(key, float(pdp.loc[pdp[feature].astype(str) == key, "pdp"].iloc[0]))
            ax.scatter(
                xi,
                yi,
                s=marker_size,
                color=row_colors[rid],
                alpha=marker_alpha,
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
            if annotate:
                ax.annotate(str(labels.iloc[rid]), (xi, yi), textcoords="offset points", xytext=(3, 3), fontsize=7)
    else:
        grid = pdp[feature].astype(float).to_numpy()
        pd_vals = pdp["pdp"].astype(float).to_numpy()
        xt = pd.to_numeric(X_mark[feature], errors="coerce").to_numpy()
        valid = np.isfinite(xt) & (xt >= grid.min()) & (xt <= grid.max())
        same_v = {}
        for rid, value in enumerate(xt):
            if not valid[rid]:
                continue
            yv = float(np.interp(value, grid, pd_vals))
            same_v[value] = same_v.get(value, -1) + 1
            offset = 0.05 * (y_max - y_min) * same_v[value]
            ax.scatter(
                value,
                yv - offset,
                s=marker_size,
                color=row_colors[rid],
                alpha=marker_alpha,
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
            if annotate:
                ax.annotate(str(labels.iloc[rid]), (value, yv - offset), textcoords="offset points", xytext=(3, 3), fontsize=7)


def _legend_indices_for_marks(
    X_mark: pd.DataFrame | None,
    legend_profile_ids: Sequence | str | None = None,
    *,
    profile_id_col: str = "profile_id",
) -> list[int]:
    if X_mark is None or X_mark.empty:
        return []

    selector = _normalize_profile_selector(legend_profile_ids)
    if selector == "none":
        return []
    if selector is None:
        return list(range(len(X_mark)))

    labels = (
        X_mark["_pdp_profile_label"].astype(str)
        if "_pdp_profile_label" in X_mark.columns
        else X_mark.index.to_series().astype(str)
    )
    if selector in {"iphone16", "iphone17", "both"}:
        lowered_labels = labels.str.lower()
        if selector == "both":
            mask = lowered_labels.str.contains("iphone 16|iphone16|iphone 17|iphone17", regex=True)
        elif selector == "iphone16":
            mask = lowered_labels.str.contains("iphone 16|iphone16", regex=True)
        else:
            mask = lowered_labels.str.contains("iphone 17|iphone17", regex=True)
        return np.flatnonzero(mask.to_numpy()).tolist()

    id_values = (
        X_mark[profile_id_col].astype(str)
        if profile_id_col in X_mark.columns
        else pd.Series("", index=X_mark.index, dtype=str)
    )
    index_values = X_mark.index.to_series().astype(str)
    mask = id_values.isin(selector) | index_values.isin(selector) | labels.isin(selector)
    return np.flatnonzero(mask.to_numpy()).tolist()


def _legend_handles_for_marks(
    X_mark: pd.DataFrame | None,
    row_colors,
    *,
    legend_profile_ids: Sequence | str | None = None,
    profile_id_col: str = "profile_id",
    markersize: int = 8,
):
    if X_mark is None or X_mark.empty:
        return []
    labels = X_mark["_pdp_profile_label"] if "_pdp_profile_label" in X_mark.columns else X_mark.index.to_series().astype(str)
    return [
        mlines.Line2D([], [], color=row_colors[i], marker="o", linestyle="None", label=str(labels.iloc[i]), markersize=markersize)
        for i in _legend_indices_for_marks(X_mark, legend_profile_ids, profile_id_col=profile_id_col)
    ]


def _normalize_profile_selector(profile_ids: Sequence | str | None) -> set[str] | str | None:
    if profile_ids is None:
        return None
    if isinstance(profile_ids, str):
        selector = profile_ids.strip()
        lowered = selector.lower().replace(" ", "")
        if lowered in {"none", "no", "false", "off", "不标注"}:
            return "none"
        if lowered in {"iphone16", "iphone17", "both"}:
            return lowered
        return {selector}
    return {str(profile_id) for profile_id in profile_ids}


def _profiles_to_mark(
    X_test: pd.DataFrame | None,
    profile_ids: Sequence | str | None = None,
    *,
    profile_id_col: str = "profile_id",
    profile_label_col: str | None = None,
) -> pd.DataFrame | None:
    """Select profiles that should be marked on PDP plots.

    profile_ids can be a list of IDs, or one of: "iphone16", "iphone17",
    "both", "none". IDs are matched against profile_id_col when present,
    otherwise against the DataFrame index. Labels default to profile_label_col,
    then profile_id_col, then the index.
    """
    if X_test is None:
        return None

    X_mark = X_test.copy()
    selector = _normalize_profile_selector(profile_ids)
    if selector == "none":
        return X_mark.iloc[0:0].copy()

    if profile_label_col is not None and profile_label_col in X_mark.columns:
        labels = X_mark[profile_label_col].astype(str)
    elif profile_id_col in X_mark.columns:
        labels = X_mark[profile_id_col].astype(str)
    else:
        labels = X_mark.index.to_series().astype(str)
    X_mark = X_mark.assign(_pdp_profile_label=labels.to_numpy())

    if selector is None:
        return X_mark

    if selector in {"iphone16", "iphone17", "both"}:
        lowered_labels = X_mark["_pdp_profile_label"].str.lower()
        if selector == "both":
            mask = lowered_labels.str.contains("iphone 16|iphone16|iphone 17|iphone17", regex=True)
        elif selector == "iphone16":
            mask = lowered_labels.str.contains("iphone 16|iphone16", regex=True)
        else:
            mask = lowered_labels.str.contains("iphone 17|iphone17", regex=True)
        return X_mark.loc[mask].copy()

    id_values = (
        X_mark[profile_id_col].astype(str)
        if profile_id_col in X_mark.columns
        else X_mark.index.to_series().astype(str)
    )
    return X_mark.loc[id_values.isin(selector)].copy()


def partial_dependence_1d(
    model: torch.nn.Module,
    X_raw: pd.DataFrame,
    feature: str,
    preprocessor: Callable[[pd.DataFrame], torch.Tensor],
    *,
    feature_cols: Sequence[str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    categories_map: Mapping[str, Sequence] | None = None,
    grid_values: Sequence | None = None,
    grid_resolution: int = 25,
) -> pd.DataFrame:
    """Compute 1D PDP values.

    Returns a DataFrame with columns: feature, pdp, score_std, n.
    """
    X_base = _as_dataframe(X_raw, feature_cols)
    if feature not in X_base.columns:
        raise ValueError(f"Feature {feature!r} is not in X_raw.")

    values = (
        np.asarray(list(grid_values), dtype=object)
        if grid_values is not None
        else _grid_values(
            X_base,
            feature,
            feature_types=feature_types,
            categories_map=categories_map,
            grid_resolution=grid_resolution,
        )
    )

    rows = []
    for value in values:
        X_tmp = X_base.copy()
        X_tmp[feature] = value
        scores = _score_model(model, preprocessor(X_tmp))
        rows.append(
            {
                feature: value,
                "pdp": float(scores.mean()),
                "score_std": float(scores.std(ddof=0)),
                "n": int(scores.size),
            }
        )
    return pd.DataFrame(rows)


def partial_dependence_2d(
    model: torch.nn.Module,
    X_raw: pd.DataFrame,
    features: tuple[str, str],
    preprocessor: Callable[[pd.DataFrame], torch.Tensor],
    *,
    feature_cols: Sequence[str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    categories_map: Mapping[str, Sequence] | None = None,
    grid_resolution: int = 15,
) -> pd.DataFrame:
    """Compute 2D PDP values.

    Returns a DataFrame with columns: feature_x, feature_y, pdp, score_std, n.
    """
    fx, fy = features
    X_base = _as_dataframe(X_raw, feature_cols)
    x_values = _grid_values(
        X_base,
        fx,
        feature_types=feature_types,
        categories_map=categories_map,
        grid_resolution=grid_resolution,
    )
    y_values = _grid_values(
        X_base,
        fy,
        feature_types=feature_types,
        categories_map=categories_map,
        grid_resolution=grid_resolution,
    )

    rows = []
    for xv in x_values:
        for yv in y_values:
            X_tmp = X_base.copy()
            X_tmp[fx] = xv
            X_tmp[fy] = yv
            scores = _score_model(model, preprocessor(X_tmp))
            rows.append(
                {
                    fx: xv,
                    fy: yv,
                    "pdp": float(scores.mean()),
                    "score_std": float(scores.std(ddof=0)),
                    "n": int(scores.size),
                }
            )
    return pd.DataFrame(rows)


def plot_pdp1d_singlefeature(
    model: torch.nn.Module,
    X_raw: pd.DataFrame,
    feature: str,
    preprocessor: Callable[[pd.DataFrame], torch.Tensor],
    *,
    feature_cols: Sequence[str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    categories_map: Mapping[str, Sequence] | None = None,
    X_test: pd.DataFrame | None = None,
    mark_profile_ids: Sequence | str | None = None,
    legend_profile_ids: Sequence | str | None = None,
    profile_id_col: str = "profile_id",
    profile_label_col: str | None = None,
    annotate_marked_profiles: bool = False,
    grid_values: Sequence | None = None,
    grid_resolution: int = 50,
    marker_size: int = 22,
    marker_alpha: float = 0.95,
    jitter: float = 0.15,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    lgd_flag: bool = False,
    return_data: bool = False,
) -> tuple[pd.DataFrame, plt.Figure, plt.Axes] | None:
    """Compute and plot a single-feature PDP in the original PDP notebook style."""
    X_mark = _profiles_to_mark(
        X_test,
        mark_profile_ids,
        profile_id_col=profile_id_col,
        profile_label_col=profile_label_col,
    )
    X_for_pdp = _as_dataframe(X_raw, feature_cols)
    if X_mark is not None and not X_mark.empty:
        X_for_pdp = pd.concat([X_for_pdp, X_mark.loc[:, X_for_pdp.columns]], ignore_index=True)

    pdp = partial_dependence_1d(
        model,
        X_for_pdp,
        feature,
        preprocessor,
        feature_cols=None,
        feature_types=feature_types,
        categories_map=categories_map,
        grid_values=grid_values,
        grid_resolution=grid_resolution,
    )

    is_cat = _feature_kind(feature, feature_types) == "categorical"
    y_min = float(pdp["pdp"].min() - 0.2)
    y_max = float(pdp["pdp"].max() + 0.2)
    row_colors = _make_vivid_colors(0 if X_mark is None else len(X_mark))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    cat_h = _draw_pdp1d_axis(ax, pdp, feature, is_cat=is_cat, y_min=y_min, y_max=y_max)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=14)
    _overlay_pdp1d_marks(
        ax,
        pdp,
        X_mark,
        feature,
        is_cat=is_cat,
        row_colors=row_colors,
        marker_size=marker_size,
        marker_alpha=marker_alpha,
        jitter=jitter,
        y_min=y_min,
        y_max=y_max,
        cat_h=cat_h,
        annotate=annotate_marked_profiles,
    )

    lgd = None
    if lgd_flag and X_mark is not None and not X_mark.empty:
        lgd = fig.legend(
            handles=_legend_handles_for_marks(
                X_mark,
                row_colors,
                legend_profile_ids=legend_profile_ids,
                profile_id_col=profile_id_col,
            ),
            loc="upper left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=False,
            title="Real iPhone models",
            ncol=1,
        )
    fig.tight_layout(rect=[0, 0, 0.9 if lgd_flag else 1.0, 1])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_args = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
        if lgd is not None:
            save_args["bbox_extra_artists"] = [lgd]
        fig.savefig(save_path, **save_args)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (pdp, fig, ax) if return_data else _QuietTuple(pdp, fig, ax)


def plot_pdp1d_grid(
    model: torch.nn.Module,
    X_raw: pd.DataFrame,
    features: Sequence[str],
    preprocessor: Callable[[pd.DataFrame], torch.Tensor],
    *,
    feature_cols: Sequence[str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    categories_map: Mapping[str, Sequence] | None = None,
    X_test: pd.DataFrame | None = None,
    mark_profile_ids: Sequence | str | None = None,
    legend_profile_ids: Sequence | str | None = None,
    profile_id_col: str = "profile_id",
    profile_label_col: str | None = None,
    annotate_marked_profiles: bool = False,
    grid_resolution: int = 50,
    ncols: int = 2,
    marker_size: int = 22,
    marker_alpha: float = 0.95,
    jitter: float = 0.15,
    save_path: str | Path | None = None,
    show: bool = True,
    return_data: bool = False,
    title_fontsize: int = 10,
    label_fontsize: int = 9,
    tick_fontsize: int = 8,
    legend_fontsize: int = 8,
) -> tuple[dict[str, pd.DataFrame], plt.Figure, np.ndarray] | None:
    """Compute and plot a 1D PDP grid in the original PDP notebook style."""
    X_mark = _profiles_to_mark(
        X_test,
        mark_profile_ids,
        profile_id_col=profile_id_col,
        profile_label_col=profile_label_col,
    )
    X_for_pdp = _as_dataframe(X_raw, feature_cols)
    if X_mark is not None and not X_mark.empty:
        X_for_pdp = pd.concat([X_for_pdp, X_mark.loc[:, X_for_pdp.columns]], ignore_index=True)

    results = {}
    all_vals = []
    for feature in features:
        pdp = partial_dependence_1d(
            model,
            X_for_pdp,
            feature,
            preprocessor,
            feature_cols=None,
            feature_types=feature_types,
            categories_map=categories_map,
            grid_resolution=grid_resolution,
        )
        results[feature] = pdp
        all_vals.append(pdp["pdp"].to_numpy())

    global_min = float(np.min(np.concatenate(all_vals)) - 0.2)
    global_max = float(np.max(np.concatenate(all_vals)) + 0.2)
    row_colors = _make_vivid_colors(0 if X_mark is None else len(X_mark))

    nrows = int(np.ceil(len(features) / ncols))
    figsize = (9, 16) if len(features) == 10 and ncols == 2 else (4.5 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, feature in zip(axes, features):
        pdp = results[feature]
        is_cat = _feature_kind(feature, feature_types) == "categorical"
        cat_h = _draw_pdp1d_axis(
            ax,
            pdp,
            feature,
            is_cat=is_cat,
            y_min=global_min,
            y_max=global_max,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
        )
        if ax not in axes[::ncols]:
            ax.set_yticklabels([])
        _overlay_pdp1d_marks(
            ax,
            pdp,
            X_mark,
            feature,
            is_cat=is_cat,
            row_colors=row_colors,
            marker_size=marker_size,
            marker_alpha=marker_alpha,
            jitter=jitter,
            y_min=global_min,
            y_max=global_max,
            cat_h=cat_h,
            annotate=annotate_marked_profiles,
        )

    for ax in axes[len(features) :]:
        ax.axis("off")

    lgd = None
    if X_mark is not None and not X_mark.empty:
        lgd = fig.legend(
            handles=_legend_handles_for_marks(
                X_mark,
                row_colors,
                legend_profile_ids=legend_profile_ids,
                profile_id_col=profile_id_col,
            ),
            loc="upper left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=False,
            title="Real iPhone models",
            ncol=1,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_args = {"dpi": 300, "bbox_inches": "tight"}
        if lgd is not None:
            save_args["bbox_extra_artists"] = [lgd]
        fig.savefig(save_path, **save_args)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return (results, fig, axes) if return_data else _QuietTuple(results, fig, axes)


def plot_pdp2d_grid(
    model: torch.nn.Module,
    X_raw: pd.DataFrame,
    features: Sequence[str],
    preprocessor: Callable[[pd.DataFrame], torch.Tensor],
    *,
    feature_cols: Sequence[str] | None = None,
    feature_types: Mapping[str, str] | None = None,
    categories_map: Mapping[str, Sequence] | None = None,
    X_test: pd.DataFrame | None = None,
    mark_profile_ids: Sequence | str | None = None,
    legend_profile_ids: Sequence | str | None = None,
    profile_id_col: str = "profile_id",
    profile_label_col: str | None = None,
    annotate_marked_profiles: bool | None = None,
    grid_resolution: int = 30,
    max_pairs: int | None = None,
    marker_size: int = 16,
    save_dir: str | Path | None = None,
    show: bool = True,
    cmap: str = "viridis",
    test_marker: str = "o",
    test_size: int | None = None,
    test_alpha: float = 0.95,
    test_annotate: bool = False,
    test_annotate_fontsize: int = 7,
    test_legend: bool = True,
    test_max_points: int = 60,
    jitter_frac_x: float = 0.012,
    jitter_frac_y: float = 0.012,
    jitter_seed: int = 42,
    legend_out: bool = True,
    legend_cols: int = 1,
    legend_fontsize: int = 8,
    legend_markerscale: float = 2.0,
    legend_bbox_to_anchor: tuple[float, float] = (-0.02, 0.5),
    vmin_vmax_mode: str = "global",
    return_data: bool = False,
) -> dict[tuple[str, str], pd.DataFrame] | None:
    """Compute and plot pairwise 2D PDPs in one grid, matching the PDP notebook style."""
    if test_size is not None:
        marker_size = test_size
    if annotate_marked_profiles is not None:
        test_annotate = annotate_marked_profiles

    X_mark = _profiles_to_mark(
        X_test,
        mark_profile_ids,
        profile_id_col=profile_id_col,
        profile_label_col=profile_label_col,
    )
    X_for_pdp = _as_dataframe(X_raw, feature_cols)
    if X_mark is not None and not X_mark.empty:
        X_for_pdp = pd.concat([X_for_pdp, X_mark.loc[:, X_for_pdp.columns]], ignore_index=True)

    pairs = list(combinations(features, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    results: dict[tuple[str, str], pd.DataFrame] = {}
    pivots = []
    global_min, global_max = np.inf, -np.inf
    for fx, fy in pairs:
        pdp = partial_dependence_2d(
            model,
            X_for_pdp,
            (fx, fy),
            preprocessor,
            feature_cols=None,
            feature_types=feature_types,
            categories_map=categories_map,
            grid_resolution=grid_resolution,
        )
        results[(fx, fy)] = pdp
        pivot = pdp.pivot(index=fx, columns=fy, values="pdp")
        pivots.append((fx, fy, pivot))
        if vmin_vmax_mode == "global":
            global_min = min(global_min, float(pivot.values.min()))
            global_max = max(global_max, float(pivot.values.max()))

    ncols = 5 if len(features) == 10 and max_pairs is None else min(5, max(1, int(np.ceil(np.sqrt(len(pairs))))))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig_width = 5 * ncols * (1.1522 if ncols == 5 else 1.0)
    fig_height = 5 * nrows
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.3, hspace=0.3)
    axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]

    rng = np.random.RandomState(jitter_seed)
    colors = _bright_palette()
    labels = None
    selected_mark = X_mark
    if selected_mark is not None and len(selected_mark) > test_max_points:
        selected_mark = selected_mark.iloc[rng.choice(len(selected_mark), size=test_max_points, replace=False)].copy()
    if selected_mark is not None and not selected_mark.empty:
        labels = selected_mark["_pdp_profile_label"] if "_pdp_profile_label" in selected_mark.columns else selected_mark.index.to_series().astype(str)

    last_im = None
    legend_handles = []
    legend_indices = set(_legend_indices_for_marks(selected_mark, legend_profile_ids, profile_id_col=profile_id_col))
    for idx, (ax, (fx, fy, pivot)) in enumerate(zip(axes, pivots)):
        y_values = pivot.index.tolist()
        x_values = pivot.columns.tolist()
        x_is_cat = _feature_kind(fy, feature_types) == "categorical"
        y_is_cat = _feature_kind(fx, feature_types) == "categorical"
        vmin = global_min if vmin_vmax_mode == "global" else float(pivot.values.min())
        vmax = global_max if vmin_vmax_mode == "global" else float(pivot.values.max())

        if x_is_cat and y_is_cat:
            extent = [-0.5, len(x_values) - 0.5, -0.5, len(y_values) - 0.5]
            ax.set_xticks(np.arange(len(x_values)))
            ax.set_xticklabels([str(x) for x in x_values], rotation=0, ha="center")
            ax.set_yticks(np.arange(len(y_values)))
            ax.set_yticklabels([str(y) for y in y_values], rotation=90, ha="center")
        elif x_is_cat:
            extent = [-0.5, len(x_values) - 0.5, float(np.min(y_values)), float(np.max(y_values))]
            ax.set_xticks(np.arange(len(x_values)))
            ax.set_xticklabels([str(x) for x in x_values], rotation=0, ha="center")
        elif y_is_cat:
            extent = [float(np.min(x_values)), float(np.max(x_values)), -0.5, len(y_values) - 0.5]
            ax.set_yticks(np.arange(len(y_values)))
            ax.set_yticklabels([str(y) for y in y_values], rotation=90, ha="center")
        else:
            x_arr = np.asarray(x_values, dtype=float)
            y_arr = np.asarray(y_values, dtype=float)
            dx = (x_arr.max() - x_arr.min()) / (len(x_arr) - 1) if len(x_arr) > 1 else 1.0
            dy = (y_arr.max() - y_arr.min()) / (len(y_arr) - 1) if len(y_arr) > 1 else 1.0
            extent = [x_arr.min() - dx / 2, x_arr.max() + dx / 2, y_arr.min() - dy / 2, y_arr.max() + dy / 2]

        im = ax.imshow(pivot.values, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        last_im = im
        ax.set_xlabel(fy)
        ax.set_ylabel(fx)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.autoscale(False)

        if selected_mark is not None and not selected_mark.empty and fx in selected_mark.columns and fy in selected_mark.columns:
            for k, (_, row) in enumerate(selected_mark.iterrows()):
                if x_is_cat:
                    try:
                        x = x_values.index(row[fy])
                    except ValueError:
                        continue
                    x_span = 1.0
                else:
                    x = float(row[fy])
                    x_span = float(np.max(x_values) - np.min(x_values)) if len(x_values) > 1 else 1.0
                if y_is_cat:
                    try:
                        y = y_values.index(row[fx])
                    except ValueError:
                        continue
                    y_span = 1.0
                else:
                    y = float(row[fx])
                    y_span = float(np.max(y_values) - np.min(y_values)) if len(y_values) > 1 else 1.0

                x_j = np.clip(x + (rng.rand() * 2 - 1) * jitter_frac_x * x_span, extent[0], extent[1])
                y_j = np.clip(y + (rng.rand() * 2 - 1) * jitter_frac_y * y_span, extent[2], extent[3])
                marker = "^" if k < 4 else test_marker
                color = colors[k % len(colors)]
                sc = ax.scatter(
                    x_j,
                    y_j,
                    marker=marker,
                    s=marker_size,
                    color=color,
                    edgecolor="k",
                    linewidths=0.6,
                    alpha=test_alpha,
                    zorder=3,
                    label=str(labels.iloc[k]) if labels is not None else str(k),
                )
                if idx == 0 and k in legend_indices:
                    legend_handles.append(sc)
                if test_annotate:
                    ax.annotate(
                        str(labels.iloc[k]) if labels is not None else str(k),
                        (x_j, y_j),
                        textcoords="offset points",
                        xytext=(np.random.choice([-1, 1]) * np.random.randint(2, 6), np.random.choice([-1, 1]) * np.random.randint(2, 6)),
                        fontsize=test_annotate_fontsize,
                        color="k",
                        ha="left",
                        va="bottom",
                        zorder=4,
                    )

    for ax in axes[len(pairs) :]:
        ax.axis("off")

    if test_legend and legend_out and legend_handles:
        fig.legend(
            handles=legend_handles,
            title="Index",
            loc="center left",
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=False,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            markerscale=legend_markerscale,
            ncol=legend_cols,
        )

    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Average Prediction", rotation=270, labelpad=15)
    fig.subplots_adjust(top=0.97, bottom=0.05)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base = save_dir / f"{_safe_model_name(model)}_pdp2d_grid"
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return results if return_data else _QuietDict(results)
