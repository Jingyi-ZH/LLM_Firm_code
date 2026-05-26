from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, SoupStrainer


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_HTML = PROJECT_ROOT / "data" / "external_knowledge" / "macrumor_iPhone17.html"
DEFAULT_TXT = PROJECT_ROOT / "data" / "external_knowledge" / "macrumor_iPhone17.txt"

TITLE_SELECTORS = [
    ".post-title",
    "h1.post-title",
    "article h1",
    "main h1",
    "h1",
    "header h1",
    "title",
]

CONTENT_SELECTORS = [
    ".post-content",
    "article .content",
    "article",
    "main .content",
    "main article",
    "main",
    "#content",
    ".content",
]

REMOVE_GLOBAL = [
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "header",
    "nav",
    "aside",
    "footer",
    "[role=banner]",
    "[role=navigation]",
    "[role=complementary]",
    "[role=contentinfo]",
    ".sidebar",
    ".site-nav",
    ".navbar",
    ".breadcrumbs",
    ".pagination",
    ".ad, .ads, .advert, .advertisement",
    ".cookie, .cookie-banner, .cookie-consent",
    ".comments, .comment, #comments",
]

REMOVE_MAIN = [
    ".toc, #toc, .table-of-contents",
    "footer",
    ".share, .social, .social-share",
    ".related, .related-posts",
]

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return ""


def _main_section(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    for selector in CONTENT_SELECTORS:
        section = soup.select_one(selector)
        if section is not None:
            return section
    return soup.body or soup


def _normalize_inline_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _extract_blocks(section: BeautifulSoup) -> list[str]:
    blocks: list[str] = []

    for node in section.find_all(BLOCK_TAGS):
        text = _normalize_inline_text(node.get_text(" ", strip=True))
        if not text:
            continue

        if node.name == "li":
            text = f"- {text}"

        if not blocks or blocks[-1] != text:
            blocks.append(text)

    if blocks:
        return blocks

    fallback = _normalize_inline_text(section.get_text(" ", strip=True))
    return [fallback] if fallback else []


def extract_title_and_content(html_text: str) -> str:
    body_only = SoupStrainer(name=lambda tag: tag == "body" or tag is None)
    soup = BeautifulSoup(html_text, "html.parser", parse_only=body_only)

    for selector in REMOVE_GLOBAL:
        for node in soup.select(selector):
            node.decompose()

    title = _first_text(soup, TITLE_SELECTORS)
    main_section = _main_section(soup)
    if main_section is None:
        return title.strip()

    for selector in REMOVE_MAIN:
        for node in main_section.select(selector):
            node.decompose()

    blocks = _extract_blocks(main_section)
    if title and blocks:
        if _normalize_inline_text(blocks[0]) == _normalize_inline_text(title):
            blocks = blocks[1:]

    content = "\n\n".join(blocks).strip()
    if title and content:
        return f"{title}\n\n{content}"
    if title:
        return title.strip()
    return content.strip()


def clean_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank_pending = False

    for line in lines:
        if not line:
            if cleaned:
                blank_pending = True
            continue
        if blank_pending:
            cleaned.append("")
            blank_pending = False
        cleaned.append(re.sub(r"\s+", " ", line))

    return "\n".join(cleaned).strip() + "\n"


def main() -> None:
    if not DEFAULT_HTML.is_file():
        raise FileNotFoundError(f"HTML file not found: {DEFAULT_HTML}")

    raw_html = DEFAULT_HTML.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_whitespace(extract_title_and_content(raw_html))

    DEFAULT_TXT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_TXT.write_text(cleaned_text, encoding="utf-8")

    print(f"Wrote cleaned text to {DEFAULT_TXT}")


if __name__ == "__main__":
    main()
