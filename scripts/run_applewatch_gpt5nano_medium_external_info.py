#!/usr/bin/env python3
"""Run the 80-row Apple Watch gpt-5-nano medium experiment with external information."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config  # noqa: E402


APPLEWATCH_DIR = PROJECT_ROOT / "AppleWatch"
BASELINE_CSV = (
    APPLEWATCH_DIR
    / "baseline_prediction"
    / "Applewatch_gpt5nano_t1_medium.csv"
)
EXTERNAL_INFO_FILES = (
    APPLEWATCH_DIR / "external_info" / "Verge20240906.txt",
    APPLEWATCH_DIR / "external_info" / "tomsguide20240831.txt",
    APPLEWATCH_DIR / "external_info" / "9to5mac20240906.txt",
)
OUTPUT_CSV = (
    APPLEWATCH_DIR
    / "external_info_output_10variants"
    / "ek_applewatch_gpt5nano_t1_medium.csv"
)
IN_PROGRESS_CSV = OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}.in_progress.csv")

FIELDNAMES = [
    "model",
    "temperature",
    "question",
    "prompt_variant",
    "prompt",
    "prompt_response",
    "answer_yes",
    "answer_no",
]


def normalize_yes_no(text: str) -> str:
    normalized = (text or "").strip().upper()
    if not normalized:
        return ""
    return normalized[0] if normalized[0] in {"Y", "N"} else ""


def build_external_info_message() -> dict[str, str]:
    sections = []
    for path in EXTERNAL_INFO_FILES:
        sections.append(f"--- {path.name} ---\n{path.read_text(encoding='utf-8').strip()}")
    return {
        "role": "user",
        "content": (
            "Use the following external information when answering the question.\n\n"
            + "\n\n".join(sections)
        ),
    }


def build_prompt(
    baseline_prompt_json: str,
    external_message: dict[str, str],
) -> list[dict[str, str]]:
    baseline_prompt: list[dict[str, Any]] = json.loads(baseline_prompt_json)
    if len(baseline_prompt) != 2:
        raise ValueError(f"Expected two baseline messages, found {len(baseline_prompt)}")
    if [message.get("role") for message in baseline_prompt] != ["system", "user"]:
        raise ValueError("Expected baseline roles [system, user]")
    return [
        {
            "role": "system",
            "content": str(baseline_prompt[0]["content"]),
        },
        external_message,
        {
            "role": "user",
            "content": str(baseline_prompt[1]["content"]),
        },
    ]


def load_baseline_rows() -> list[dict[str, str]]:
    with BASELINE_CSV.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                f"Baseline columns do not match expected schema: {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != 80:
        raise ValueError(f"Expected 80 baseline rows, found {len(rows)}")
    return rows


def load_completed(path: Path) -> set[tuple[str, int]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            (row["question"].strip(), int(row["prompt_variant"]))
            for row in csv.DictReader(file)
        }


def validate_completed_output(path: Path) -> None:
    baseline_rows = load_baseline_rows()
    baseline_by_key = {
        (row["question"].strip(), int(row["prompt_variant"])): row
        for row in baseline_rows
    }

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Output columns do not match baseline: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 80:
        raise ValueError(f"Expected 80 completed rows in {path}, found {len(rows)}")

    for row in rows:
        key = (row["question"].strip(), int(row["prompt_variant"]))
        baseline_prompt = json.loads(baseline_by_key[key]["prompt"])
        prompt = json.loads(row["prompt"])
        if prompt[0] != baseline_prompt[0] or prompt[2] != baseline_prompt[1]:
            raise ValueError(f"Baseline prompt changed for row: {key}")
        external_content = prompt[1]["content"]
        if not all(source.name in external_content for source in EXTERNAL_INFO_FILES):
            raise ValueError(f"Missing external-information source for row: {key}")


def main() -> None:
    for path in (BASELINE_CSV, *EXTERNAL_INFO_FILES):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = load_baseline_rows()
    external_message = build_external_info_message()
    completed = load_completed(IN_PROGRESS_CSV)
    pending = [
        row
        for row in rows
        if (row["question"].strip(), int(row["prompt_variant"])) not in completed
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = IN_PROGRESS_CSV.is_file() and IN_PROGRESS_CSV.stat().st_size > 0

    cfg = get_config()
    api_key_env_var = cfg.get("openai", "api_key_env_var", default="OPENAI_API_KEY")
    client = OpenAI(api_key=cfg.get_api_key(api_key_env_var))

    print(f"In-progress output: {IN_PROGRESS_CSV}")
    print(f"Final output: {OUTPUT_CSV}")
    print(f"Completed: {len(completed)}; pending: {len(pending)}")

    with IN_PROGRESS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for index, row in enumerate(pending, start=1):
            question = row["question"].strip()
            prompt_variant = int(row["prompt_variant"])
            prompt = build_prompt(row["prompt"], external_message)
            response = client.responses.create(
                model="gpt-5-nano",
                input=prompt,
                temperature=1.0,
                reasoning={"effort": "medium"},
            )
            output_text = response.output_text
            label = normalize_yes_no(output_text)
            writer.writerow(
                {
                    "model": "gpt-5-nano",
                    "temperature": 1.0,
                    "question": question,
                    "prompt_variant": prompt_variant,
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "prompt_response": output_text,
                    "answer_yes": 1 if label == "Y" else 0,
                    "answer_no": 1 if label == "N" else 0,
                }
            )
            file.flush()
            print(
                f"[{len(completed) + index:02d}/80] "
                f"variant={prompt_variant} label={label or 'INVALID'} question={question}"
            )

    validate_completed_output(IN_PROGRESS_CSV)
    IN_PROGRESS_CSV.replace(OUTPUT_CSV)
    print(f"Completed output atomically replaced: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
