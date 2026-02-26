"""Prompt generation for pairwise comparison experiments.

This module provides functions to generate prompts for LLM belief elicitation
using pairwise conjoint comparisons.
"""

from typing import Sequence, List, Dict

from config import get_config

# User template: asks LLM to choose between two smartphone alternatives
USER_TEMPLATE = (
    "Please consider two smartphone alternatives, both with 256 GB storage "
    "capacity and black color:\n"
    "{l0}: {p0}\n"
    "{l1}: {p1}\n\n"
    "Return exactly one label: '{l0}' or '{l1}'. "
    "Output nothing else, no quotes, no punctuation, no spaces, and no line breaks."
)

# 10 system prompts with varying styles (conversational/academic)
# Designed to be neutral and avoid bias
SYSTEM_TEXTS: Dict[str, str] = {
    "prompt_0": (
        "You will be provided with two smartphone alternatives described by a set of "
        "attributes. Decide which one is more likely to appear in the next iPhone "
        "generation lineup (covering standard, Pro, Max, Plus, or Air) within 6 months."
    ),
    "prompt_1": (
        "You will be shown two smartphone profiles with specific attributes. "
        "Identify which is more plausible for Apple's next iPhone lineup "
        "(standard, Pro, Max, Plus, Air) within 6 months."
    ),
    "prompt_2": (
        "Two smartphone alternatives will be presented as stimuli. "
        "Select the alternative that more closely aligns with the next-generation "
        "iPhone models (standard, Pro, Max, Plus, Air) scheduled within 6 months."
    ),
    "prompt_3": (
        "You will receive two smartphone descriptions. "
        "Choose the one more likely to appear in Apple's upcoming iPhone series "
        "within 6 months."
    ),
    "prompt_4": (
        "In this task, two sets of smartphone specifications are provided. "
        "Determine which set better corresponds to Apple's forthcoming iPhone "
        "generation (6-month horizon)."
    ),
    "prompt_5": (
        "Participants are asked to evaluate two smartphone prototypes. "
        "Identify the prototype most consistent with the characteristics of "
        "the next iPhone lineup (within 6 months)."
    ),
    "prompt_6": (
        "Two smartphone alternatives defined by multiple attributes will be shown. "
        "Decide which alternative is more realistic to be included in Apple's "
        "next iPhone lineup within 6 months."
    ),
    "prompt_7": (
        "Two smartphone configurations are introduced as experimental stimuli. "
        "Determine which configuration more plausibly belongs to the next "
        "iPhone generation (6 months)."
    ),
    "prompt_8": (
        "This task presents two smartphone concepts. "
        "Select the concept more likely to be adopted in Apple's next iPhone "
        "generation within 6 months."
    ),
    "prompt_9": (
        "Two smartphone attribute sets will be evaluated. "
        "Decide which set is more likely to be represented in the upcoming "
        "iPhone generation (within 6 months)."
    ),
}

# Neutral criteria for reasoning
NEUTRAL_CRITERIA = (
    "Base your assessment on historical trajectories of iPhone development, "
    "market positioning dynamics, technical feasibility of the configurations, "
    "and considerations of plausible generational change patterns."
)

def _get_prompting_config() -> dict:
    try:
        return get_config().get_prompting() or {}
    except Exception:
        # Keep legacy behavior when config isn't available (e.g., during partial imports).
        return {}

def _variant_index(normalized_key: str) -> int:
    """Extract integer index from 'prompt_X'."""
    try:
        return int(str(normalized_key).split("_", 1)[1])
    except Exception:
        return 0


def _select_neutral_criteria(
    prompting: dict,
    normalized_key: str,
    fallback: str,
) -> str:
    """Select a neutral criteria variant from app spec.

    Supports:
      - prompting.neutral_criteria: str (legacy)
      - prompting.neutral_criteria_variants: list[str]
      - prompting.neutral_criteria_texts: dict[str, str] with keys like prompt_0..prompt_9
    """
    if not isinstance(prompting, dict):
        return fallback

    # Dict keyed by prompt variant name (preferred for explicit mapping)
    texts = prompting.get("neutral_criteria_texts")
    if isinstance(texts, dict) and texts:
        v = texts.get(normalized_key)
        if v is None:
            idx = _variant_index(normalized_key)
            v = texts.get(f"criteria_{idx}")
        if v is None:
            v = texts.get("default")
        if v is not None:
            return str(v)

    # List variants (indexed by prompt variant)
    variants = prompting.get("neutral_criteria_variants")
    if isinstance(variants, list) and variants:
        idx = _variant_index(normalized_key) % len(variants)
        return str(variants[idx])

    # Legacy string
    neutral = prompting.get("neutral_criteria")
    if neutral is not None:
        return str(neutral)

    return fallback


def _normalize_key(variant_key) -> str:
    """Normalize variant key to 'prompt_X' format."""
    s = str(variant_key)
    return s if s.startswith("prompt_") else f"prompt_{s}"


def get_prompt_variant(
    variant_key,
    pair: Sequence[str],
    labels: Sequence[str],
    date_override: str | None = None,
) -> List[Dict[str, str]]:
    """Generate a prompt variant for pairwise comparison.

    Args:
        variant_key: Prompt variant identifier (0-9 or 'prompt_0' to 'prompt_9')
        pair: Sequence of two profile strings to compare
        labels: Sequence of two labels for the profiles (e.g., ['G', 'H'])

    Returns:
        List of message dictionaries with 'role' and 'content' keys,
        suitable for OpenAI chat API.
    """
    prompting = _get_prompting_config()
    user_template = prompting.get("user_template", USER_TEMPLATE)
    system_texts = prompting.get("system_texts", SYSTEM_TEXTS)
    default_date = prompting.get("default_date", "2024-06-01")

    key = _normalize_key(variant_key)
    neutral_criteria = _select_neutral_criteria(
        prompting=prompting,
        normalized_key=key,
        fallback=NEUTRAL_CRITERIA,
    )
    date_text = date_override or default_date
    system_variant = system_texts.get(key, system_texts.get("prompt_0", SYSTEM_TEXTS["prompt_0"]))
    try:
        system_variant = str(system_variant).format(**prompting)
    except Exception:
        system_variant = str(system_variant)
    system_text = f"Assume the current date is {date_text}. " + system_variant

    template_vars = dict(prompting)
    template_vars.pop("system_texts", None)
    try:
        user_text = str(user_template).format(
            l0=str(labels[0]),
            l1=str(labels[1]),
            p0=str(pair[0]),
            p1=str(pair[1]),
            **template_vars,
        )
    except KeyError:
        # Backward-compatible fallback (old templates used simple replace)
        user_text = (
            str(user_template)
            .replace("{l0}", str(labels[0]))
            .replace("{l1}", str(labels[1]))
            .replace("{p0}", str(pair[0]))
            .replace("{p1}", str(pair[1]))
        )
    user_text = user_text + "\n\n" + str(neutral_criteria)

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def get_all_prompt_variants() -> Dict[str, str]:
    """Get all available prompt variants.

    Returns:
        Dictionary of prompt variant keys to their system texts.
    """
    return SYSTEM_TEXTS.copy()
