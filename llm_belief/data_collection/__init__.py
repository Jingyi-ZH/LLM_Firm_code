"""Data collection module for LLM belief elicitation.

This module provides:
    - PairwiseCollector: Main class for running pairwise comparison experiments
    - Prompt generation utilities
"""

from .collector import PairwiseCollector
from .prompts import get_prompt_variant

__all__ = ["PairwiseCollector", "get_prompt_variant"]
