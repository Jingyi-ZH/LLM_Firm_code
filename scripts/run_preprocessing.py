#!/usr/bin/env python3
"""CLI for preprocessing tasks.

This script provides a command-line interface for running
preprocessing tasks such as profile generation.

Usage:
    # Generate profiles with default settings
    python scripts/run_preprocessing.py --task generate-profiles

    # Generate profiles with custom seed and output
    python scripts/run_preprocessing.py --task generate-profiles --seed 42 --output my_profiles.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from config import get_config
from llm_belief.preprocessing import ProfileGenerator


def main():
    """Main entry point for preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Belief Elicitation Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate profiles with default settings
    python scripts/run_preprocessing.py --task generate-profiles

    # Generate profiles with custom seed
    python scripts/run_preprocessing.py --task generate-profiles --seed 42

    # Generate profiles with custom output filename
    python scripts/run_preprocessing.py --task generate-profiles --output custom_profiles.csv
        """,
    )

    parser.add_argument(
        "--task",
        choices=["generate-profiles"],
        required=True,
        help="Preprocessing task to run",
    )
    cfg = get_config()
    default_seed = cfg.get("project", "random_seed", default=2025)
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="Random seed for reproducibility (default from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (optional)",
    )

    args = parser.parse_args()

    if args.task == "generate-profiles":
        generator = ProfileGenerator(seed=args.seed)
        output_path = generator.generate_csv(output_file=args.output)
        profile_count = generator.get_profile_count()
        print(f"Generated {profile_count} profiles")
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
