from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5-nano"
TEMPERATURE = 1.0
REASONING_EFFORT = "default"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
QUESTIONS_CSV = SCRIPT_DIR / "questions.csv"
ANSWERS_CSV = SCRIPT_DIR / "answers.csv"
ENV_PATH = PROJECT_ROOT / ".env"


def load_api_client() -> OpenAI:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found. Expected it in {ENV_PATH} or the environment."
        )
    return OpenAI(api_key=api_key)


def read_questions() -> list[str]:
    if not QUESTIONS_CSV.is_file():
        raise FileNotFoundError(f"Question CSV not found: {QUESTIONS_CSV}")

    df = pd.read_csv(QUESTIONS_CSV)
    if "question" not in df.columns:
        raise ValueError(
            f"Expected a 'question' column in {QUESTIONS_CSV}, got {list(df.columns)}"
        )

    questions = [
        str(value).strip()
        for value in df["question"].tolist()
        if pd.notna(value) and str(value).strip()
    ]
    if not questions:
        raise ValueError(f"No non-empty questions found in {QUESTIONS_CSV}")
    return questions


def ask_question(client: OpenAI, question: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=question,
        temperature=TEMPERATURE,
    )
    return response.output_text.strip()


def main() -> None:
    client = load_api_client()
    questions = read_questions()

    rows: list[dict[str, str | float]] = []
    for question in questions:
        answer = ask_question(client, question)
        rows.append(
            {
                "questions": question,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "reasoning effort": REASONING_EFFORT,
                "api_answer": answer,
            }
        )

    with open(ANSWERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "questions",
                "model",
                "temperature",
                "reasoning effort",
                "api_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {ANSWERS_CSV}")


if __name__ == "__main__":
    main()
