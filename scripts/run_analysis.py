#!/usr/bin/env python3
"""CLI for running analysis pipelines.

This script provides a command-line interface for running
various analysis and visualization pipelines.

Usage:
    python scripts/run_analysis.py --pipeline dimensionality --method fa_varimax
    python scripts/run_analysis.py --pipeline pdp --model mlp
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))


def main():
    """Main entry point for analysis CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Belief Elicitation Analysis Pipeline",
    )

    parser.add_argument(
        "--pipeline",
        choices=["dimensionality", "pdp", "integrated_gradient", "all"],
        required=True,
        help="Analysis pipeline to run",
    )
    parser.add_argument(
        "--method",
        choices=["fa_varimax", "pca", "kernel_pca"],
        default="fa_varimax",
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "mlp", "mlp_attention", "adalasso", "xgboost", "all"],
        default="all",
        help="Model to analyze",
    )
    parser.add_argument(
        "--n-profiles",
        type=int,
        default=20000,
        help="Number of profiles to analyze",
    )

    args = parser.parse_args()

    print("Analysis CLI is a placeholder.")
    print("For now, please use the notebooks in analysis/")
    print(f"Selected pipeline: {args.pipeline}")
    print(f"Method: {args.method}")

    # TODO: Implement analysis pipelines after migrating analysis module
    # from llm_belief.analysis import DimensionalityReducer, plot_pdp1d_grid


if __name__ == "__main__":
    main()
