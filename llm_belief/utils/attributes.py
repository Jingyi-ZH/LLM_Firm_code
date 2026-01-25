"""iPhone attribute definitions and profile generation utilities.

This module provides:
    - Attribute definitions loaded from config
    - Profile generation with jitter
    - Label randomization for pairwise comparisons
"""

import random
import itertools
from typing import Dict, List, Any, Tuple
import pandas as pd

import sys
from pathlib import Path

_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_config


def get_attributes() -> Dict[str, Dict]:
    """Get iPhone attributes from configuration.

    Returns:
        Dictionary of attribute configurations.
    """
    return get_config().get_attributes()


def get_attribute_values() -> Dict[str, List]:
    """Get attribute values as a simple dictionary.

    Returns:
        Dictionary mapping attribute names to their possible values.
    """
    attrs = get_attributes()
    return {
        attr_config['name']: attr_config['values']
        for attr_config in attrs.values()
    }


def get_real_profiles() -> Dict[str, Dict]:
    """Get real iPhone profiles from configuration.

    Returns:
        Dictionary of real iPhone profile specifications.
    """
    return get_config().get_real_profiles()


def format_attribute_value(attr_key: str, value: Any) -> str:
    """Format an attribute value using config display conventions."""
    unit_map = {
        "battery_life": "hours video playback",
        "screen_size": "inches",
        "thickness": "mm",
        "front_camera": "MP",
        "rear_camera": "MP",
        "focal_length": "x",
        "ram": "GB",
        "price": "$",
    }
    if attr_key == "price":
        return f"${value}"
    if attr_key in {"focal_length"}:
        return f"{value}x"
    if attr_key in unit_map:
        return f"{value} {unit_map[attr_key]}"
    return str(value)


def format_profile_for_prompt(profile: Dict[str, Any]) -> Dict[str, str]:
    """Convert a config-keyed profile into display-name/value strings."""
    attrs = get_attributes()
    formatted = {}
    for key, val in profile.items():
        display_name = attrs.get(key, {}).get("name", key)
        formatted[display_name] = format_attribute_value(key, val)
    return formatted


# ===================
# Label Generation
# ===================

LABELS = ["G", "H", "I", "J", "K", "L", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]


def random_label_only() -> List[str]:
    """Generate two random labels for pairwise comparison.

    Returns:
        List of two randomly chosen labels.
    """
    return random.sample(LABELS, 2)


def random_label_choice(options: List[Any]) -> Dict[str, Any]:
    """Randomly assign labels to options.

    Args:
        options: List of two options to label.

    Returns:
        Dictionary mapping labels to options.
    """
    labels = random.sample(LABELS, 2)
    shuffled_options = options[:]
    random.shuffle(shuffled_options)
    return {
        labels[0]: shuffled_options[0],
        labels[1]: shuffled_options[1],
    }


# ===================
# Jitter Functions
# ===================

def apply_jitter(attr_key: str, value: Any, rng: random.Random) -> Any:
    """Apply jitter to an attribute value.

    Args:
        attr_key: Attribute key in config (e.g., 'battery_life')
        value: Original value
        rng: Random number generator

    Returns:
        Jittered value (or original if jitter disabled)
    """
    attrs = get_attributes()
    if attr_key not in attrs:
        return value

    jitter_spec = attrs[attr_key].get('jitter', {})

    # Check if jitter is enabled
    if not jitter_spec.get('dist') and not jitter_spec.get('enabled', True):
        return value

    if 'dist' not in jitter_spec:
        return value

    # Check probability of no jitter
    if rng.random() < jitter_spec.get('p_no_jitter', 0.0):
        return value

    dist = jitter_spec['dist']
    scale = jitter_spec.get('scale', 1.0)
    round_digits = jitter_spec.get('round')
    clip = jitter_spec.get('clip')

    # Apply jitter based on distribution
    if dist == "uniform":
        noisy = value + rng.uniform(-scale, scale)
    else:  # normal
        noisy = value + rng.gauss(0, scale)

    # Apply clipping
    if clip:
        noisy = max(clip[0], min(clip[1], noisy))

    # Apply rounding
    if round_digits is not None:
        noisy = round(noisy, round_digits)

    return noisy


def jitter_profile(profile: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Apply jitter to all attributes in a profile.

    Args:
        profile: Dictionary of attribute name -> value
        rng: Random number generator

    Returns:
        Jittered profile dictionary
    """
    attrs = get_attributes()
    # Create mapping from attribute name to key
    name_to_key = {v['name']: k for k, v in attrs.items()}

    jittered = {}
    for name, value in profile.items():
        key = name_to_key.get(name)
        if key:
            jittered[name] = apply_jitter(key, value, rng)
        else:
            jittered[name] = value

    return jittered


# ===================
# Profile Generation
# ===================

def generate_all_profiles(seed: int = 42) -> List[Dict[str, Any]]:
    """Generate all possible profiles with jitter applied.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of profile dictionaries, shuffled
    """
    rng = random.Random(seed)
    attr_values = get_attribute_values()

    keys = list(attr_values.keys())
    values_lists = [attr_values[k] for k in keys]

    profiles = []
    for combo in itertools.product(*values_lists):
        base = dict(zip(keys, combo))
        profiles.append(jitter_profile(base, rng))

    rng.shuffle(profiles)
    return profiles


def rearrange_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """Rearrange and format a profiles DataFrame for display.

    Converts numeric values to human-readable strings with units.

    Args:
        df_input: DataFrame with raw profile values

    Returns:
        Formatted DataFrame
    """
    df = df_input.copy()

    # Set integer columns
    int_columns = [
        "battery life (in hours of video playback)",
        "front camera resolution (in MP)",
        "rear camera main lens resolution (in MP)",
        "rear camera longest focal length (in x)",
        "Geekbench multicore score",
        "RAM",
        "price",
    ]

    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Rename columns to shorter names
    rename_map = {
        "battery life (in hours of video playback)": "battery life",
        "screen size (in inches)": "screen size",
        "thickness (in mm)": "thickness",
        "front camera resolution (in MP)": "front camera resolution",
        "rear camera main lens resolution (in MP)": "rear camera main lens resolution",
        "rear camera longest focal length (in x)": "rear camera longest focal length",
    }
    df = df.rename(columns=rename_map)

    # Add units
    if "battery life" in df.columns:
        df["battery life"] = df["battery life"].apply(lambda x: f"{x} hours video playback")
    if "screen size" in df.columns:
        df["screen size"] = df["screen size"].apply(lambda x: f"{x} inches")
    if "thickness" in df.columns:
        df["thickness"] = df["thickness"].apply(lambda x: f"{x} mm")
    if "front camera resolution" in df.columns:
        df["front camera resolution"] = df["front camera resolution"].apply(lambda x: f"{x} MP")
    if "rear camera main lens resolution" in df.columns:
        df["rear camera main lens resolution"] = df["rear camera main lens resolution"].apply(
            lambda x: f"{x} MP"
        )
    if "rear camera longest focal length" in df.columns:
        df["rear camera longest focal length"] = df["rear camera longest focal length"].apply(
            lambda x: f"{x}x"
        )
    if "RAM" in df.columns:
        df["RAM"] = df["RAM"].apply(lambda x: f"{x} GB")
    if "price" in df.columns:
        df["price"] = df["price"].apply(lambda x: f"${x}")

    return df
