"""Resampling utilities for profile selection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from llm_belief.utils import get_data_path


def resample_profile_ids(
    scored_df: pd.DataFrame,
    n_makeup: int,
    sample_limit: int = 20000,
    seed: int = 2025,
    output_file: Optional[str] = None,
    use_existing: bool = True,
) -> np.ndarray:
    """Resample profile_id values from the first N scored profiles.

    Args:
        scored_df: Scored profiles DataFrame with a profile_id column.
        n_makeup: Number of profile_ids to sample.
        sample_limit: Max number of rows to sample from (default: 20000).
        seed: Random seed for reproducibility.
        output_file: Optional output filename under data/ for saving ids.
        use_existing: Reuse saved ids if the output file already exists.

    Returns:
        Numpy array of sampled profile_id values.
    """
    if "profile_id" not in scored_df.columns:
        raise ValueError("scored_df must include a 'profile_id' column")

    output_name = output_file or f"sample{n_makeup}_profile_ids.npy"
    output_path = get_data_path(output_name)
    if use_existing and output_path.is_file():
        return np.load(output_path, allow_pickle=True)

    limit = min(sample_limit, len(scored_df))
    if n_makeup > limit:
        raise ValueError(
            f"n_makeup ({n_makeup}) exceeds available profiles ({limit})"
        )

    candidates = scored_df.iloc[:limit]["profile_id"].to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(candidates, size=n_makeup, replace=False)
    np.save(output_path, sampled)
    return sampled
