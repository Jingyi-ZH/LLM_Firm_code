"""Scoring models for LLM belief elicitation.

This module provides various models for scoring iPhone profiles
based on pairwise comparison data:
    - LogisticRegression: Simple linear model
    - MLPScorer: Multi-layer perceptron
    - MLPAttentionScore: MLP with attention mechanism
    - LinearInteractionModel: Adaptive Lasso with interactions
"""

from .scoring import LogisticRegression, MLPScorer, MLPAttentionScore
from .adalasso import LinearInteractionModel, train_pairwise_adalasso
from .xgboost_model import train_xgb_pairwise

__all__ = [
    "LogisticRegression",
    "MLPScorer",
    "MLPAttentionScore",
    "LinearInteractionModel",
    "train_pairwise_adalasso",
    "train_xgb_pairwise",
]
