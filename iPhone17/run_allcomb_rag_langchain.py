from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.environ.setdefault(
    "LLM_BELIEF_APP_SPEC_PATH",
    str(PROJECT_ROOT / "config" / "apps" / "iphone17.yaml"),
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from llm_belief.data_collection.collector import PairwiseCollector
from llm_belief.data_collection.prompts import get_prompt_variant
from llm_belief.utils.attributes import random_label_only

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from html_to_text import extract_title_and_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run allcomb with static context + LangChain RAG over iPhone 17 rumor HTML."
    )
    parser.add_argument(
        "--real-profile-csv",
        default=str(SCRIPT_DIR / "represents104.csv"),
        help="CSV of profiles for allcomb.",
    )
    parser.add_argument(
        "--html",
        default=str(PROJECT_ROOT / "data" / "external_knowledge" / "macrumor_iPhone17.html"),
        help="HTML file used as the RAG corpus.",
    )
    parser.add_argument(
        "--context",
        default=str(PROJECT_ROOT / "data" / "re16.txt"),
        help="Static text context file to prepend to every prompt.",
    )
    parser.add_argument(
        "--context-date",
        default="2025-03-17",
        help="Date injected into the base pairwise prompt.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "iPhone17" / "represents104_allcomb_RAG.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable name for the OpenAI API key. Defaults to config.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional override for reasoning effort.",
    )
    parser.add_argument(
        "--logprobs",
        choices=["on", "off"],
        default=None,
        help="Optional logprobs mode forwarded to PairwiseCollector.",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=3,
        help="Number of retrieved chunks per pairwise prompt.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for the RAG corpus.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for the RAG corpus.",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-large",
        help="Embedding model for LangChain retrieval.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    root_path = (PROJECT_ROOT / path).resolve()
    if root_path.exists():
        return root_path
    return root_path


def resolve_data_or_absolute(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    cfg = get_config()
    data_path = (cfg.get_path("data_dir") / path).resolve()
    if data_path.exists():
        return data_path
    root_path = (PROJECT_ROOT / path).resolve()
    if root_path.exists():
        return root_path
    return data_path


def sanitize_prompt_text(text: str) -> str:
    return re.sub(r"/(?:Users|home)/\S*", "[local-path]", text or "")


def load_profiles_csv(csv_path_str: str) -> List[Tuple[str, Dict[str, Any]]]:
    csv_path = resolve_path(csv_path_str)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Profile CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "real_profile_id" not in df.columns:
        raise ValueError(f"CSV missing required column 'real_profile_id': {csv_path}")

    attrs = get_config().get_attributes() or {}
    if not attrs:
        raise ValueError("No attributes found in active app spec.")

    attr_keys = list(attrs.keys())
    col_for_key: Dict[str, str] = {}
    missing: List[str] = []
    for key in attr_keys:
        if key in df.columns:
            col_for_key[key] = key
            continue
        display_name = attrs.get(key, {}).get("name", key)
        if display_name in df.columns:
            col_for_key[key] = display_name
            continue
        missing.append(key)

    if missing:
        raise ValueError(
            f"CSV missing required attribute columns {missing}: {csv_path}"
        )

    profiles: List[Tuple[str, Dict[str, Any]]] = []
    for row_idx, row in df.iterrows():
        rid = row.get("real_profile_id")
        if pd.isna(rid) or str(rid).strip() == "":
            raise ValueError(f"Row {row_idx} has empty real_profile_id: {csv_path}")

        profile: Dict[str, Any] = {}
        for key, col in col_for_key.items():
            val = row.get(col)
            if pd.isna(val):
                raise ValueError(
                    f"Row {row_idx} missing value for '{col}' / '{key}': {csv_path}"
                )
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            profile[key] = val
        profiles.append((str(rid).strip(), profile))

    return profiles


def load_static_context(context_path_str: str) -> str:
    context_path = resolve_data_or_absolute(context_path_str)
    if not context_path.is_file():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    return sanitize_prompt_text(context_path.read_text(encoding="utf-8"))


def load_rag_document(html_path_str: str) -> Document:
    html_path = resolve_path(html_path_str)
    if not html_path.is_file():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    clean_text = extract_title_and_content(raw_html)
    if not clean_text.strip():
        raise ValueError(f"No readable content extracted from {html_path}")

    return Document(
        page_content=clean_text,
        metadata={
            "source": str(html_path),
            "title": html_path.stem,
            "type": "html",
        },
    )


def build_vector_store(doc: Document, embed_model: str, chunk_size: int, chunk_overlap: int) -> InMemoryVectorStore:
    embeddings = OpenAIEmbeddings(model=embed_model)
    vector_store = InMemoryVectorStore(embeddings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents([doc])
    if not chunks:
        raise ValueError("No chunks produced from the RAG document.")
    vector_store.add_documents(chunks)
    return vector_store


def build_retrieval_context(vector_store: InMemoryVectorStore, query: str, rag_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    hits = vector_store.similarity_search(query, k=rag_k)
    blocks: List[str] = []
    meta_hits: List[Dict[str, Any]] = []
    for idx, doc in enumerate(hits, start=1):
        source = str(doc.metadata.get("source", ""))
        title = str(doc.metadata.get("title", "")) or Path(source).name or "(untitled)"
        source_label = Path(source).name if source else ""
        text = sanitize_prompt_text(doc.page_content.strip())
        header = f"[{idx}] {title}"
        if source_label and source_label != title:
            header = f"{header} ({source_label})"
        blocks.append(f"{header}\n{text}")
        meta_hits.append(
            {
                "id": idx,
                "title": title,
                "source": source_label,
                "start_index": doc.metadata.get("start_index"),
            }
        )
    return "\n\n---\n\n".join(blocks), meta_hits


def build_augmented_prompt(
    base_prompt: List[Dict[str, str]],
    static_context: str,
    retrieval_context: str,
) -> List[Dict[str, str]]:
    preamble = []
    if static_context.strip():
        preamble.append(
            {
                "role": "system",
                "content": (
                    "The following static external context is provided:\n"
                    f"{sanitize_prompt_text(static_context.strip())}\n"
                ),
            }
        )
    if retrieval_context.strip():
        preamble.append(
            {
                "role": "system",
                "content": (
                    "Retrieved external context from the iPhone 17 rumor HTML:\n"
                    f"{sanitize_prompt_text(retrieval_context.strip())}\n\n"
                    "Use the retrieved context only as reference and preserve uncertainty where the source is speculative."
                ),
            }
        )
    return preamble + base_prompt


def default_output_path() -> Path:
    out_dir = PROJECT_ROOT / "output" / "iPhone17"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "represents104_allcomb_RAG.csv"


def main() -> None:
    args = parse_args()

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    profiles = load_profiles_csv(args.real_profile_csv)
    static_context = load_static_context(args.context)
    rag_doc = load_rag_document(args.html)
    vector_store = build_vector_store(
        rag_doc,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    collector = PairwiseCollector(
        api_key_env_var=args.api_key_env,
        logprobs=args.logprobs,
    )

    ids: List[str] = []
    formatted_profiles: List[Dict[str, Any]] = []
    for rid, profile in profiles:
        ids.append(str(rid))
        formatted_profiles.append(
            collector._get_real_profile_formatted(rid, real_profile=profile)
        )

    total_pairs = len(profiles) * (len(profiles) - 1) // 2
    num_variants = int(collector.cfg.get("collection", "num_prompt_variants", default=10) or 10)
    if num_variants <= 0:
        num_variants = 10

    output_path = resolve_path(args.output) if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "model",
        "temperature",
        "pair_id",
        "pair",
        "prompt_variant",
        "prompt",
        "retrieval_context",
        "retrieval_hits",
        "prompt_response",
        "chosen_profile",
        "chosen_profile_id",
        "nonchosen_profile_id",
    ]
    if collector.logprobs_enabled:
        cols += ["prob_chosen", "prob_nochosen"]

    completed_pair_ids: set[int] = set()
    file_exists = output_path.is_file()
    if file_exists:
        try:
            existing = pd.read_csv(output_path, usecols=["pair_id"])
            completed_pair_ids.update(
                pd.to_numeric(existing["pair_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
        except Exception:
            completed_pair_ids = set()

    print(f"Loaded {len(profiles)} profiles -> {total_pairs} allcomb pairs")
    print(f"RAG corpus source: {resolve_path(args.html)}")
    print(f"Static context: {resolve_data_or_absolute(args.context)}")
    print(f"Output: {output_path}")
    if completed_pair_ids:
        print(f"Resuming: {len(completed_pair_ids)} pairs already completed")

    pair_id = 0
    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if (not file_exists) or output_path.stat().st_size == 0:
            writer.writeheader()

        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                if pair_id in completed_pair_ids:
                    pair_id += 1
                    continue

                labels = random_label_only()
                pair_ids = [ids[i], ids[j]]
                profiles_map = {
                    labels[0]: formatted_profiles[i],
                    labels[1]: formatted_profiles[j],
                }

                prompt_variant = pair_id % num_variants
                prompt_labels = labels if pair_id % 2 == 0 else [labels[1], labels[0]]
                prompt_pair = [profiles_map[prompt_labels[0]], profiles_map[prompt_labels[1]]]
                base_prompt = get_prompt_variant(
                    prompt_variant,
                    prompt_pair,
                    prompt_labels,
                    date_override=args.context_date,
                )

                query_text = " ".join(str(msg.get("content", "")) for msg in base_prompt).strip()
                retrieval_context, retrieval_hits = build_retrieval_context(
                    vector_store,
                    query_text,
                    rag_k=args.rag_k,
                )
                prompt = build_augmented_prompt(
                    base_prompt=base_prompt,
                    static_context=static_context,
                    retrieval_context=retrieval_context,
                )

                res, prob_chosen, prob_nochosen = collector._call_api(
                    prompt,
                    reasoning_effort=args.reasoning_effort,
                )

                chosen_profile = profiles_map.get(res)
                chosen_profile_id = None
                nonchosen_profile_id = None
                if chosen_profile is not None and res in labels:
                    chosen_idx = labels.index(res)
                    chosen_profile_id = pair_ids[chosen_idx]
                    nonchosen_profile_id = pair_ids[1 - chosen_idx]

                row = {
                    "model": collector.logprobs_model if collector.logprobs_enabled else collector.model,
                    "temperature": collector.logprobs_temperature if collector.logprobs_enabled else collector.temperature,
                    "pair_id": pair_id,
                    "pair": json.dumps(profiles_map, ensure_ascii=False),
                    "prompt_variant": prompt_variant,
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "retrieval_context": retrieval_context,
                    "retrieval_hits": json.dumps(retrieval_hits, ensure_ascii=False),
                    "prompt_response": res,
                    "chosen_profile": json.dumps(chosen_profile, ensure_ascii=False) if chosen_profile is not None else "",
                    "chosen_profile_id": chosen_profile_id,
                    "nonchosen_profile_id": nonchosen_profile_id,
                }
                if collector.logprobs_enabled:
                    row["prob_chosen"] = prob_chosen
                    row["prob_nochosen"] = prob_nochosen

                writer.writerow(row)
                f.flush()
                print(f"Completed pair {pair_id + 1}/{total_pairs}")
                pair_id += 1

    print("All pairs completed.")


if __name__ == "__main__":
    main()
