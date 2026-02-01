#!/usr/bin/env python3
"""CLI for LLM data collection.

This script provides a command-line interface for running various
pairwise comparison experiments.

Usage:
    # Basic pairwise comparison (10,000 pairs)
    python scripts/run_collection.py --experiment basic --start 0 --end 1000

    # Real vs. makeup profile comparison
    python scripts/run_collection.py --experiment fixreal --real-profile "iPhone 16 Pro"

    # Real vs. top-scored profiles
    python scripts/run_collection.py --experiment top --real-profile "iPhone 16 Pro" --n-top 50

    # Real vs. makeup with injected context
    python scripts/run_collection.py --experiment context --real-profile "iPhone 16 Pro" \
        --context data/re16.txt

    # RAG via RAG_langchain
    python scripts/run_collection.py --experiment rag --real-profile "iPhone 16 Pro" \
        --api-key-env OPENAI_API_KEY

    # Custom FAISS-based RAG
    python scripts/run_collection.py --experiment rag-faiss --real-profile "iPhone 16 Pro" \
        --rag-faiss path/to/index.faiss --rag-meta path/to/records.jsonl
"""

import argparse
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from llm_belief.data_collection import PairwiseCollector


def main():
    """Main entry point for data collection CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Belief Elicitation Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic pairwise comparison
    python scripts/run_collection.py --experiment basic --start 0 --end 100

    # Real vs. makeup profiles
    python scripts/run_collection.py --experiment fixreal --real-profile "iPhone 16 Pro"

    # Real vs. top-50 profiles
    python scripts/run_collection.py --experiment top --real-profile "iPhone 16 Pro" --n-top 50

    # Real vs. makeup with injected context
    python scripts/run_collection.py --experiment context --real-profile "iPhone 16 Pro" \
        --context data/re16.txt

    # RAG via RAG_langchain
    python scripts/run_collection.py --experiment rag --real-profile "iPhone 16 Pro" \
        --api-key-env OPENAI_API_KEY

    # Custom FAISS-based RAG
    python scripts/run_collection.py --experiment rag-faiss --real-profile "iPhone 16 Pro" \
        --rag-faiss path/to/index.faiss --rag-meta path/to/records.jsonl
        """,
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "basic",
            "fixreal",
            "top",
            "context",
            "rag",
            "rag-faiss",
        ],
        required=True,
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start index for basic experiment",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index for basic experiment (exclusive)",
    )
    parser.add_argument(
        "--real-profile",
        type=str,
        help="Real profile ID for fixreal/top experiments (e.g., 'iPhone 16 Pro')",
    )
    parser.add_argument(
        "--n-makeup",
        type=int,
        help="Number of makeup profiles for fixreal experiment",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        help="Number of top profiles for top experiment",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Context text file for context experiment (required)",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        help="Sampled profile ids file for context experiment (optional)",
    )
    parser.add_argument(
        "--scored-limit",
        type=int,
        help="Limit rows when loading scored profiles (optional)",
    )
    parser.add_argument(
        "--context-date",
        type=str,
        help="Context date for injected knowledge (optional)",
    )
    parser.add_argument(
        "--rag-faiss",
        type=str,
        help="Path to FAISS index for RAG (rag-faiss only; env RAG_FAISS)",
    )
    parser.add_argument(
        "--rag-meta",
        type=str,
        help="Path to metadata jsonl for RAG (rag-faiss only; env RAG_META)",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        help="Top-k chunks to retrieve (rag-faiss only)",
    )
    parser.add_argument(
        "--rag-per-chars",
        type=int,
        help="Max chars per retrieved chunk (rag-faiss only)",
    )
    parser.add_argument(
        "--rag-embed-model",
        type=str,
        help="Embedding model for retrieval (rag-faiss only)",
    )
    parser.add_argument(
        "--exclude-ids-file",
        type=str,
        help="Profile id file to exclude for RAG (rag-faiss only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (optional)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        help="Environment variable name for API key (optional)",
    )
    parser.add_argument(
        "--api-key-envs",
        type=str,
        help="Comma-separated API key env vars for parallel basic runs (optional)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        help="Override reasoning effort (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name (rag only)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature (rag only)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.experiment == "basic":
        if args.start is None or args.end is None:
            parser.error("--start and --end are required for basic experiment")
    elif args.experiment in ["fixreal", "top", "context", "rag", "rag-faiss"]:
        if args.real_profile is None:
            parser.error(f"--real-profile is required for {args.experiment} experiment")
    if args.experiment == "context" and not args.context:
        parser.error("--context is required for context experiment")

    # Run experiment
    if args.experiment == "basic":
        if args.api_key_envs:
            env_vars = [e.strip() for e in args.api_key_envs.split(",") if e.strip()]
            if not env_vars:
                parser.error("--api-key-envs is empty after parsing")
            total = args.end - args.start
            if total <= 0:
                parser.error("--end must be greater than --start")

            chunk = (total + len(env_vars) - 1) // len(env_vars)
            futures = []
            with ThreadPoolExecutor(max_workers=len(env_vars)) as executor:
                for i, env_var in enumerate(env_vars):
                    sub_start = args.start + i * chunk
                    sub_end = min(args.start + (i + 1) * chunk, args.end)
                    if sub_start >= sub_end:
                        continue
                    output_file = args.output
                    if output_file and len(env_vars) > 1:
                        stem, *rest = output_file.rsplit(".", 1)
                        suffix = f"_part{i+1}"
                        output_file = f"{stem}{suffix}.{rest[0]}" if rest else f"{stem}{suffix}"
                    collector = PairwiseCollector(api_key_env_var=env_var)
                    futures.append(
                        executor.submit(
                            collector.collect_basic,
                            start_idx=sub_start,
                            end_idx=sub_end,
                            reasoning_effort=args.reasoning_effort,
                            output_file=output_file,
                        )
                    )
                for future in as_completed(futures):
                    output_path = future.result()
                    print(f"\nOutput saved to: {output_path}")
            return

        collector = PairwiseCollector(api_key_env_var=args.api_key_env)
        output_path = collector.collect_basic(
            start_idx=args.start,
            end_idx=args.end,
            reasoning_effort=args.reasoning_effort,
            output_file=args.output,
        )
    elif args.experiment == "fixreal":
        collector = PairwiseCollector(api_key_env_var=args.api_key_env)
        output_path = collector.collect_fixreal(
            real_profile_id=args.real_profile,
            n_makeup=args.n_makeup,
            reasoning_effort=args.reasoning_effort,
            output_file=args.output,
        )
    elif args.experiment == "top":
        collector = PairwiseCollector(api_key_env_var=args.api_key_env)
        output_path = collector.collect_top(
            real_profile_id=args.real_profile,
            n_top=args.n_top,
            reasoning_effort=args.reasoning_effort,
            output_file=args.output,
        )
    elif args.experiment == "context":
        collector = PairwiseCollector(api_key_env_var=args.api_key_env)
        output_path = collector.collect_context_fixreal(
            real_profile_id=args.real_profile,
            context_file=args.context,
            sample_ids_file=args.sample_ids_file or "sample5k_profile_ids.npy",
            scored_limit=args.scored_limit or 20000,
            output_file=args.output,
            context_date=args.context_date or "2025-03-15",
            reasoning_effort=args.reasoning_effort,
        )
    elif args.experiment == "rag":
        rag_script = _project_root / "RAG_langchain" / "main_rag_langchain.py"
        if not rag_script.is_file():
            raise FileNotFoundError(f"RAG_langchain script not found: {rag_script}")
        account = args.api_key_env or "OPENAI_API_KEY"
        cmd = [
            sys.executable,
            str(rag_script),
            "--real_profile_id",
            args.real_profile,
            "--account",
            account,
        ]
        if args.reasoning_effort:
            cmd += ["--reasoning-effort", args.reasoning_effort]
        if args.model:
            cmd += ["--model", args.model]
        if args.temperature is not None:
            cmd += ["--temperature", str(args.temperature)]
        if args.n_makeup is not None:
            cmd += ["--n_makeup", str(args.n_makeup)]
        subprocess.run(cmd, check=True)
        output_path = None
    elif args.experiment == "rag-faiss":
        collector = PairwiseCollector(api_key_env_var=args.api_key_env)
        output_path = collector.collect_rag_fixreal(
            real_profile_id=args.real_profile,
            n_makeup=args.n_makeup,
            exclude_ids_file=args.exclude_ids_file or "fixreal_used_profile_ids.npy",
            rag_faiss=args.rag_faiss,
            rag_meta=args.rag_meta,
            rag_k=args.rag_k or 3,
            rag_per_chars=args.rag_per_chars or 1200,
            rag_embed_model=args.rag_embed_model or "text-embedding-3-small",
            reasoning_effort=args.reasoning_effort,
            output_file=args.output,
        )

    if output_path:
        print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
