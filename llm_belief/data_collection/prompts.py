"""Prompt generation for pairwise comparison experiments.

This module provides functions to generate prompts for LLM belief elicitation
using pairwise conjoint comparisons.
"""

from typing import Sequence, List, Dict


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
    key = _normalize_key(variant_key)
    date_text = date_override or "2024-06-01"
    system_text = (
        f"Assume the current date is {date_text}. "
        + SYSTEM_TEXTS.get(key, SYSTEM_TEXTS["prompt_0"])
    )

    user_text = (
        USER_TEMPLATE.replace("{l0}", str(labels[0]))
        .replace("{l1}", str(labels[1]))
        .replace("{p0}", str(pair[0]))
        .replace("{p1}", str(pair[1]))
        + "\n\n"
        + NEUTRAL_CRITERIA
    )

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
