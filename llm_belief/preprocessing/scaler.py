"""Preprocessing utilities for model training."""

from typing import Tuple
import joblib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from llm_belief.utils.paths import get_data_path
from config import get_config


_LEGACY_NUM_COLS = [
    "battery life (in hours of video playback)",
    "screen size (in inches)",
    "thickness (in mm)",
    "front camera resolution (in MP)",
    "rear camera main lens resolution (in MP)",
    "rear camera longest focal length (in x)",
    "Geekbench multicore score",
    "RAM",
    "price",
]
_LEGACY_CAT_COLS = ["ultrawide camera"]


def _get_feature_columns() -> tuple[list[str], list[str], dict[str, dict]]:
    """Derive numeric/categorical feature columns from the active app spec.

    Returns:
        (num_cols, cat_cols, cat_encode_maps_by_colname)
    """
    try:
        attrs = get_config().get_attributes() or {}
    except Exception:
        attrs = {}

    if not isinstance(attrs, dict) or not attrs:
        return list(_LEGACY_NUM_COLS), list(_LEGACY_CAT_COLS), {}

    num_cols: list[str] = []
    cat_cols: list[str] = []
    encode_maps: dict[str, dict] = {}

    for key, cfg in attrs.items():
        if not isinstance(cfg, dict):
            continue
        col = cfg.get("name", key)
        if not isinstance(col, str):
            col = str(col)
        if bool(cfg.get("categorical", False)):
            cat_cols.append(col)
            emap = cfg.get("encode_map")
            if isinstance(emap, dict) and emap:
                encode_maps[col] = emap
        else:
            num_cols.append(col)

    # Backward-compatible default encoding for iPhone ultrawide
    if "ultrawide camera" in cat_cols and "ultrawide camera" not in encode_maps:
        encode_maps["ultrawide camera"] = {"not equipped": 1, "equipped": 0}

    return num_cols, cat_cols, encode_maps


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _, cat_cols, encode_maps = _get_feature_columns()
    for col in cat_cols:
        if col not in df.columns:
            continue
        emap = encode_maps.get(col)
        if isinstance(emap, dict) and emap:
            df[col] = df[col].map(emap).astype(int)
        else:
            # If no encode map is provided, keep behavior conservative.
            # Users should define `encode_map` for categorical attributes in the app spec.
            df[col] = pd.Categorical(df[col]).codes.astype(int)
    return df


def preprocess(df: pd.DataFrame, scaler_path: str = "scaler.pkl") -> torch.Tensor:
    """Fit scaler and transform features; saves scaler to data/."""
    df = _encode_categoricals(df)
    scaler = StandardScaler()
    num_cols, _, _ = _get_feature_columns()
    present_num = [c for c in num_cols if c in df.columns]
    if present_num:
        df[present_num] = scaler.fit_transform(df[present_num])
    scaler_file = get_data_path(scaler_path)
    joblib.dump(scaler, scaler_file)
    return torch.from_numpy(df.values).float()


def preprocess_test(df: pd.DataFrame, scaler_path: str = "scaler.pkl") -> torch.Tensor:
    """Transform features using saved scaler from data/."""
    df = _encode_categoricals(df)
    scaler_file = get_data_path(scaler_path)
    scaler = joblib.load(scaler_file)
    num_cols, _, _ = _get_feature_columns()
    present_num = [c for c in num_cols if c in df.columns]
    if present_num:
        df[present_num] = scaler.transform(df[present_num])
    return torch.from_numpy(df.values).float()
