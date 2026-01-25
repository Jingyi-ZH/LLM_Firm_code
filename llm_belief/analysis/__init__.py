"""Analysis module for LLM belief elicitation.

This module provides:
    - Dimensionality reduction (FA, PCA, Kernel PCA)
    - Partial Dependence Plots (PDP)
    - Integrated Gradients for interpretability
"""

from .dimensionality import DimensionalityResult, fit_transform, plot_embedding
from .visualization import ProbabilityVizResult, visualize_probability_distribution

__all__ = [
    "DimensionalityResult",
    "fit_transform",
    "plot_embedding",
    "ProbabilityVizResult",
    "visualize_probability_distribution",
]
