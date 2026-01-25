"""Preprocessing utilities for model training."""

from typing import Tuple
import joblib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from llm_belief.utils.paths import get_data_path


NUM_COLS = [
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
CAT_COLS = ["ultrawide camera"]


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ultrawide camera" in df.columns:
        df["ultrawide camera"] = (
            df["ultrawide camera"].map({"not equipped": 1, "equipped": 0}).astype(int)
        )
    return df


def preprocess(df: pd.DataFrame, scaler_path: str = "scaler.pkl") -> torch.Tensor:
    """Fit scaler and transform features; saves scaler to data/."""
    df = _encode_categoricals(df)
    scaler = StandardScaler()
    df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
    scaler_file = get_data_path(scaler_path)
    joblib.dump(scaler, scaler_file)
    return torch.from_numpy(df.values).float()


def preprocess_test(df: pd.DataFrame, scaler_path: str = "scaler.pkl") -> torch.Tensor:
    """Transform features using saved scaler from data/."""
    df = _encode_categoricals(df)
    scaler_file = get_data_path(scaler_path)
    scaler = joblib.load(scaler_file)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    return torch.from_numpy(df.values).float()
