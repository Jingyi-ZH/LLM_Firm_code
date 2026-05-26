from __future__ import annotations

import os
import re
from html import unescape
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5-nano"
TEMPERATURE = 1.0
MAX_SOURCE_CHARS = 120_000

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
HTML_PATH = PROJECT_ROOT / "data" / "external_knowledge" / "macrumor_iPhone17.html"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "external_knowledge" / "sum_macrumor_iPhone17.txt"
)


def load_client() -> OpenAI:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found. Expected it in {ENV_PATH} or the environment."
        )
    return OpenAI(api_key=api_key)


def html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script\b.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style\b.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)</div\s*>", "\n", text)
    text = re.sub(r"(?i)</li\s*>", "\n", text)
    text = re.sub(r"(?i)</h[1-6]\s*>", "\n\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def load_source_text() -> str:
    if not HTML_PATH.is_file():
        raise FileNotFoundError(f"HTML file not found: {HTML_PATH}")

    html = HTML_PATH.read_text(encoding="utf-8", errors="ignore")
    text = html_to_text(html)
    if not text:
        raise ValueError(f"No readable text extracted from {HTML_PATH}")
    return text[:MAX_SOURCE_CHARS]


def build_prompt(source_text: str) -> str:
    return (
        "You are summarizing an HTML page about iPhone 17 rumors.\n\n"
        "Task:\n"
        "1. Summarize only the important content from the page.\n"
        "2. Focus on rumored lineup names, release timing, design, display, chips, cameras, "
        "battery, pricing, and any notable uncertainty.\n"
        "3. Ignore site navigation, ads, menus, comments, legal text, and unrelated boilerplate.\n"
        "4. Write a clean plain-text summary with short section headers.\n"
        "5. If some claims are framed as rumors or uncertainty, preserve that uncertainty.\n\n"
        "Source text extracted from the HTML file:\n\n"
        f"{source_text}"
    )


def summarize_html(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=TEMPERATURE,
    )
    summary = response.output_text.strip()
    if not summary:
        raise RuntimeError("API returned an empty summary.")
    return summary


def main() -> None:
    client = load_client()
    source_text = load_source_text()
    prompt = build_prompt(source_text)
    summary = summarize_html(client, prompt)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(summary + "\n", encoding="utf-8")

    print(f"Wrote summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
