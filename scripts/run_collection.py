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
from typing import Optional, List, Tuple, Dict, Any

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

import pandas as pd

from config import get_config
from llm_belief.data_collection import PairwiseCollector


def _is_output_dir_arg(output_arg: str) -> bool:
    if not output_arg:
        return False
    # Treat as directory when:
    # - user explicitly ends with a path separator
    # - path exists and is a directory
    # - no suffix and no dot in the final path segment (common "folder" case)
    if output_arg.endswith(("/", "\\")):
        return True
    p = Path(output_arg)
    if p.exists() and p.is_dir():
        return True
    return p.suffix == "" and "." not in p.name


def _resolve_output_dir(output_arg: str) -> Path:
    """Resolve an output directory path.

    - Absolute paths are used as-is.
    - Relative paths are interpreted under the configured output/ directory.
    """
    p = Path(output_arg)
    if p.is_absolute():
        out_dir = p
    else:
        out_dir = get_config().get_path("output_dir") / p
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_fixreal_real_profiles_csv(
    real_profile_arg: str,
) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """If real_profile_arg points to a CSV file, load real profiles from it.

    CSV requirements:
      - Column: real_profile_id
      - All attributes present as columns (either config keys like 'battery_life'
        or their config display names like 'battery life (in hours of video playback)').
    """
    path_arg = Path(real_profile_arg)
    if path_arg.suffix.lower() != ".csv":
        return None

    if path_arg.is_absolute():
        path = path_arg
    else:
        cwd_path = Path.cwd() / path_arg
        root_path = _project_root / path_arg
        if cwd_path.is_file():
            path = cwd_path
        elif root_path.is_file():
            path = root_path
        else:
            raise FileNotFoundError(
                f"--real-profile looks like a CSV path but was not found: {path_arg}"
            )

    if not path.is_file():
        raise FileNotFoundError(f"Real profile CSV not found: {path}")

    df = pd.read_csv(path)
    if "real_profile_id" not in df.columns:
        raise ValueError(f"CSV missing required column 'real_profile_id': {path}")

    attrs = get_config().get_attributes() or {}
    if not attrs:
        raise ValueError("No attributes found in config; cannot validate CSV columns.")

    attr_keys = list(attrs.keys())
    col_for_key: dict[str, str] = {}
    missing: list[str] = []
    for key in attr_keys:
        if key in df.columns:
            col_for_key[key] = key
            continue
        name = attrs.get(key, {}).get("name", key)
        if name in df.columns:
            col_for_key[key] = name
            continue
        missing.append(key)

    if missing:
        expected_key_cols = attr_keys
        expected_name_cols = [attrs.get(k, {}).get("name", k) for k in attr_keys]
        raise ValueError(
            "CSV missing required attribute columns.\n"
            f"  file: {path}\n"
            f"  missing (attribute keys): {missing}\n"
            f"  expected columns include either keys: {expected_key_cols}\n"
            f"  ...or display names: {expected_name_cols}"
        )

    profiles: list[tuple[str, dict]] = []
    for row_idx, row in df.iterrows():
        rid = row.get("real_profile_id")
        if pd.isna(rid) or str(rid).strip() == "":
            raise ValueError(f"Row {row_idx} has empty real_profile_id: {path}")
        real_profile_id = str(rid).strip()

        profile: dict[str, object] = {}
        for key, col in col_for_key.items():
            val = row.get(col)
            if pd.isna(val):
                raise ValueError(
                    f"Row {row_idx} missing value for '{col}' (attribute '{key}'): {path}"
                )
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            profile[key] = val

        profiles.append((real_profile_id, profile))

    return profiles


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
        help=(
            "Real profile ID for fixreal/top/context/rag/rag-faiss experiments (e.g., 'iPhone 16 Pro'). "
            "For fixreal only, you may also pass a CSV path to run fixreal once per row."
        ),
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
        "--logprobs",
        choices=["on", "off"],
        default=None,
        help="Enable or disable logprobs (collector experiments only)",
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
    elif args.experiment in ["fixreal", "top", "context", "rag-faiss"]:
        if args.real_profile is None:
            parser.error(f"--real-profile is required for {args.experiment} experiment")
    elif args.experiment == "rag":
        if args.real_profile is None:
            parser.error("--real-profile is required for rag experiment")
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
                    collector = PairwiseCollector(
                        api_key_env_var=env_var,
                        logprobs=args.logprobs,
                    )
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

        collector = PairwiseCollector(
            api_key_env_var=args.api_key_env,
            logprobs=args.logprobs,
        )
        output_path = collector.collect_basic(
            start_idx=args.start,
            end_idx=args.end,
            reasoning_effort=args.reasoning_effort,
            output_file=args.output,
        )
    elif args.experiment == "fixreal":
        collector = PairwiseCollector(
            api_key_env_var=args.api_key_env,
            logprobs=args.logprobs,
        )
        try:
            csv_profiles = _load_fixreal_real_profiles_csv(args.real_profile)
        except (FileNotFoundError, ValueError, TypeError) as exc:
            parser.error(str(exc))

        output_dir = None
        output_file = args.output
        if args.output and _is_output_dir_arg(args.output):
            output_dir = _resolve_output_dir(args.output)
            output_file = None

        cfg = get_config()
        resolved_n_makeup = args.n_makeup or cfg.get("collection", "default_n_makeup", default=5000)
        if csv_profiles is not None:
            if output_file:
                parser.error(
                    "--output cannot be used when --real-profile is a CSV path. "
                    "Use a folder path to write outputs under that folder."
                )
            for rid, profile in csv_profiles:
                safe_id = rid.replace(" ", "_")
                row_output_file = None
                if output_dir is not None:
                    row_output_file = str(output_dir / f"{safe_id}_fixreal{resolved_n_makeup}.csv")
                output_path = collector.collect_fixreal(
                    real_profile_id=rid,
                    n_makeup=args.n_makeup,
                    reasoning_effort=args.reasoning_effort,
                    output_file=row_output_file,
                    real_profile=profile,
                )
                print(f"\nOutput saved to: {output_path}")
            return

        output_path = collector.collect_fixreal(
            real_profile_id=args.real_profile,
            n_makeup=args.n_makeup,
            reasoning_effort=args.reasoning_effort,
            output_file=(
                str(output_dir / f"{args.real_profile.replace(' ', '_')}_fixreal{resolved_n_makeup}.csv")
                if output_dir is not None
                else output_file
            ),
        )
    elif args.experiment == "top":
        collector = PairwiseCollector(
            api_key_env_var=args.api_key_env,
            logprobs=args.logprobs,
        )
        output_path = collector.collect_top(
            real_profile_id=args.real_profile,
            n_top=args.n_top,
            reasoning_effort=args.reasoning_effort,
            output_file=args.output,
        )
    elif args.experiment == "context":
        collector = PairwiseCollector(
            api_key_env_var=args.api_key_env,
            logprobs=args.logprobs,
        )
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
        if args.logprobs is not None:
            cmd += ["--logprobs", args.logprobs]
        if args.model:
            cmd += ["--model", args.model]
        if args.temperature is not None:
            cmd += ["--temperature", str(args.temperature)]
        if args.n_makeup is not None:
            cmd += ["--n_makeup", str(args.n_makeup)]
        subprocess.run(cmd, check=True)
        output_path = None
    elif args.experiment == "rag-faiss":
        collector = PairwiseCollector(
            api_key_env_var=args.api_key_env,
            logprobs=args.logprobs,
        )
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
