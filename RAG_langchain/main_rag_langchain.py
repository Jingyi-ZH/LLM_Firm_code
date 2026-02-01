import os
import ast
import pandas as pd
import sys
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import random
from openai import OpenAI
os.chdir(
    os.path.dirname(os.path.abspath(__file__))
)
import numpy as np
import argparse

_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import get_config
from llm_belief.preprocessing import resample_profile_ids
from llm_belief.utils.paths import get_data_path
def read_arguments():
    parser = argparse.ArgumentParser(
        description="Run task with start, end, and account parameters"
    )

    parser.add_argument("--start", type=int, help="start profile pair idx, e.g. 0")
    parser.add_argument("--end", type=int, help="end profile pair idx, e.g. 100")
    parser.add_argument(
        "--account", type=str, required=True, help="OpenAI account name"
    )
    parser.add_argument(
        "--real_profile_id", type=str, help="Real profile ID, e.g. iPhone 16 Pro"
    )
    parser.add_argument("--n_makeup", type=int, help="Number of makeup profiles")
    parser.add_argument("--n_top", type=int, help="Number of top scored profiles")
    parser.add_argument("--shift", type=int, default=0, help="Shift id for profiles")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Override reasoning effort",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature",
    )
    parser.add_argument(
        "--logprobs",
        choices=["on", "off"],
        default=None,
        help="Enable or disable logprobs",
    )

    args = parser.parse_args()
    if args.start is not None and args.end is None:
        parser.error("--start requires --end.")

    return args

args = read_arguments()
real_profile_id, account = (
    args.real_profile_id,
    args.account,
)

cfg = get_config()
n_makeup = args.n_makeup or cfg.get("collection", "default_n_makeup", default=5000)
sample_limit = cfg.get("collection", "fixreal_sample_limit", default=20000)
seed = cfg.get("project", "random_seed", default=2025)
random.seed(seed)

model_name = args.model or cfg.get("openai", "model", default="gpt-5-nano")
temperature = (
    args.temperature
    if args.temperature is not None
    else cfg.get("openai", "temperature", default=1.0)
)
reasoning_effort = (
    args.reasoning_effort
    if args.reasoning_effort is not None
    else cfg.get("openai", "reasoning_effort", default="medium")
)
logprobs_cfg = cfg.get("openai", "logprobs", default={})
default_logprobs_enabled = bool(logprobs_cfg.get("enabled", False))
if args.logprobs is None:
    logprobs_enabled = default_logprobs_enabled
else:
    logprobs_enabled = (args.logprobs == "on")
logprobs_model = logprobs_cfg.get("model", model_name)
logprobs_temperature = logprobs_cfg.get("temperature", 0.0)
logprobs_max_output_tokens = logprobs_cfg.get("max_output_tokens", 16)
logprobs_top_logprobs = logprobs_cfg.get("top_logprobs", 2)
logprobs_include = logprobs_cfg.get("include", ["message.output_text.logprobs"])
openai_client = OpenAI(api_key=os.environ.get(account))

llm = init_chat_model(
    model_name,
    temperature=temperature,
    reasoning_effort=reasoning_effort,
    model_provider="openai",
)

# llm = init_chat_model("gpt-5-nano",api_key=os.environ.get(account), temperature = 1, reasoning_effort=args.reasoning_effort, model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

from pathlib import Path
from typing import Optional, List, TypedDict
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def extract_title_and_content(
    html: str,
    title_selectors: Optional[List[str]] = None,
    content_selectors: Optional[List[str]] = None,
) -> str:
    """Extract title and main content from HTML while filtering common noise."""
    body_only = SoupStrainer(name=lambda tag: tag == "body" or tag is None)
    soup = BeautifulSoup(html, "html.parser", parse_only=body_only)

    for selector in [
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
    ]:
        for node in soup.select(selector):
            node.decompose()

    if not title_selectors:
        title_selectors = [
            ".post-title",
            "h1.post-title",
            "article h1",
            "main h1",
            "h1",
            "header h1",
            "title",
        ]

    title_text = ""
    for sel in title_selectors:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            title_text = node.get_text(separator=" ", strip=True)
            break

    if not content_selectors:
        content_selectors = [
            ".post-content",
            "article .content",
            "article",
            "main .content",
            "main article",
            "main",
            "#content",
            ".content",
        ]

    main_section = None
    for sel in content_selectors:
        main_section = soup.select_one(sel)
        if main_section:
            break
    if not main_section:
        main_section = soup.body or soup

    for selector in [
        ".toc, #toc, .table-of-contents",
        "footer",
        ".share, .social, .social-share",
        ".related, .related-posts",
    ]:
        for node in main_section.select(selector):
            node.decompose()

    content_text = main_section.get_text(separator="\n", strip=True)

    combined = title_text.strip()
    if combined and content_text:
        combined += "\n\n" + content_text
    elif not combined:
        combined = content_text

    return combined


def load_html_then_filter(
    html_path: str,
    title_selectors: Optional[List[str]] = None,
    content_selectors: Optional[List[str]] = None,
) -> List[Document]:
    """Load a local HTML file and return a cleaned Document."""
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    loader = TextLoader(str(path), autodetect_encoding=True)
    raw_docs = loader.load()
    if len(raw_docs) == 0:
        raise ValueError("No document loaded. Check the file path or encoding.")

    raw_html = raw_docs[0].page_content
    source_meta = raw_docs[0].metadata

    clean_text = extract_title_and_content(
        raw_html,
        title_selectors=title_selectors,
        content_selectors=content_selectors,
    )

    doc = Document(
        page_content=clean_text,
        metadata={
            **source_meta,
            "source": str(path),
            "extracted": True,
            "note": "title + main content only",
        },
    )
    return [doc]

# Correct imports for LangChain v0.2+
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ========== Your CSS selectors ==========
TITLE_SELECTORS = [
    ".post-title",
    "h1.post-title",
    "article h1",
    "main h1",
    "h1",
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


# ========== Utility functions ==========

def _read_text(path: Path) -> str:
    """Safely read text from a file with UTF-8, ignoring bad characters if needed."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _pick_first(soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
    """Return text from the first matching selector."""
    for css in selectors:
        node = soup.select_one(css)
        if node:
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return None


def _pick_all_join(soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
    """Return concatenated text from the first selector that matches any nodes."""
    for css in selectors:
        nodes = soup.select(css)
        if nodes:
            parts = []
            for n in nodes:
                txt = n.get_text("\n", strip=True)
                if txt:
                    parts.append(txt)
            joined = "\n\n".join(p for p in parts if p.strip())
            if joined.strip():
                return joined
    return None


def extract_html_with_selectors(
    file_path: Path,
    title_selectors: List[str],
    content_selectors: List[str],
) -> Optional[Document]:
    """Extract title and main content from one HTML file using CSS selectors."""
    raw = _read_text(file_path)
    soup = BeautifulSoup(raw, "lxml")

    # Try to extract title; fallback to <title> or filename
    title = _pick_first(soup, title_selectors)
    if not title:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        else:
            title = file_path.stem

    # Try to extract main content; fallback to all visible text
    content = _pick_all_join(soup, content_selectors)
    if not content:
        content = soup.get_text("\n", strip=True)

    content = content.strip()
    if not content:
        return None

    meta = {
        "source": str(file_path),
        "filename": file_path.name,
        "title": title,
        "type": "html",
    }
    return Document(page_content=content, metadata=meta)


def load_pdf_as_docs(file_path: Path) -> List[Document]:
    """Load one PDF as a list of page-level Documents."""
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for d in docs:
        d.metadata = {
            **d.metadata,
            "source": str(file_path),
            "filename": file_path.name,
            "title": file_path.stem,
            "type": "pdf",
        }
    return docs


def load_folder_pdfs_individual(pdf_path: Path) -> List[Document]:
    """Helper function: process one PDF file."""
    return load_pdf_as_docs(pdf_path)


def load_folder_mixed(
    folder: str,
    title_selectors: List[str],
    content_selectors: List[str],
    recursive: bool = True,
) -> List[Document]:
    """
    Load both HTML and PDF files from a folder (and subfolders if recursive=True),
    returning all as LangChain Documents.
    """
    base = Path(folder)
    pattern = "**/*" if recursive else "*"

    html_docs: List[Document] = []
    pdf_docs: List[Document] = []

    # --- Parse HTML files ---
    for p in base.glob(pattern):
        if p.is_file() and p.suffix.lower() in {".html", ".htm"}:
            doc = extract_html_with_selectors(p, title_selectors, content_selectors)
            if doc:
                html_docs.append(doc)

    # --- Parse PDF files ---
    for p in base.glob(pattern):
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdf_docs.extend(load_folder_pdfs_individual(p))

    return html_docs + pdf_docs


def split_docs(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """Optional: split long documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# retrieval and generation example
docs = load_folder_mixed(
    folder="./raw_material/iPhone16lineup/",
    title_selectors=TITLE_SELECTORS,
    content_selectors=CONTENT_SELECTORS,
    recursive=True,  # set False if you only want the current folder
)

# Example: inspect the first document
if not docs:
    print("No HTML or PDF files were parsed in this folder.")
else:
    print(f"Total documents loaded: {len(docs)}")
    print(f"Type: {docs[0].metadata.get('type')}")
    print(f"Source: {docs[0].metadata.get('source')}")
    print(f"Title: {docs[0].metadata.get('title')}")
    print(f"Character count: {len(docs[0].page_content)}")
    print("\n====== Preview (first 800 chars) ======\n")
    print(docs[0].page_content[:800])

# Splitting documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

# retrieval and generation
from langchain_classic import hub

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    prob_chosen: float
    prob_nochosen: float


def _get_field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_logprobs(response, chosen_text):
    output = _get_field(response, "output", []) or []
    if not output:
        return None, None
    content = _get_field(output[0], "content", []) or []
    if not content:
        return None, None
    logprobs = _get_field(content[0], "logprobs", None)
    if not logprobs:
        return None, None

    token_info = logprobs[0]
    token = _get_field(token_info, "token", "")
    logprob = _get_field(token_info, "logprob", None)
    top_logprobs = _get_field(token_info, "top_logprobs", []) or []

    chosen_norm = (chosen_text or "").strip()
    token_norm = (token or "").strip()
    prob_chosen = None
    prob_nochosen = None

    if logprob is not None and token_norm == chosen_norm:
        prob_chosen = float(np.exp(logprob))

    if top_logprobs:
        for item in top_logprobs:
            t = _get_field(item, "token", "")
            lp = _get_field(item, "logprob", None)
            if lp is None:
                continue
            if (t or "").strip() == chosen_norm:
                prob_chosen = float(np.exp(lp))
            elif prob_nochosen is None:
                prob_nochosen = float(np.exp(lp))

    if prob_nochosen is None and top_logprobs:
        for item in top_logprobs:
            t = _get_field(item, "token", "")
            lp = _get_field(item, "logprob", None)
            if lp is None:
                continue
            if (t or "").strip() != chosen_norm:
                prob_nochosen = float(np.exp(lp))
                break

    return prob_chosen, prob_nochosen

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_value = prompt.invoke({"question": state["question"], "context": docs_content})
    if logprobs_enabled:
        messages = prompt_value.to_messages()
        openai_messages = []
        for m in messages:
            role = getattr(m, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            openai_messages.append({"role": role, "content": m.content})
        response = openai_client.responses.create(
            model=logprobs_model,
            input=openai_messages,
            temperature=logprobs_temperature,
            max_output_tokens=logprobs_max_output_tokens,
            top_logprobs=logprobs_top_logprobs,
            include=logprobs_include,
        )
        output_text = response.output_text
        prob_chosen, prob_nochosen = extract_logprobs(response, output_text)
        return {
            "answer": output_text,
            "prob_chosen": prob_chosen,
            "prob_nochosen": prob_nochosen,
        }
    response = llm.invoke(prompt_value)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))

result = graph.invoke({"question": "What is the next generation of iPhone lineup? And what is its release date?"})

print(f"Context: {result['context']}\n\n")
print(f"Answer: {result['answer']}")

# ========== Main ==========
import ast
import csv

profiles_df = pd.read_csv(get_data_path(cfg.get("collection", "profiles_file")))
sample_ids = resample_profile_ids(
    profiles_df,
    n_makeup=n_makeup,
    sample_limit=sample_limit,
    seed=seed,
    output_file=f"sample{n_makeup}_profile_ids.npy",
    use_existing=True,
)

df = pd.read_csv(f"../output/{real_profile_id}_fixreal{n_makeup}.csv")
df_sample = df[df["profile_id"].isin(sample_ids)].reset_index(drop=True)
assert len(df_sample) == len(sample_ids)

csv_path = f"../output/rag_langchain_{real_profile_id}_fixreal{n_makeup}.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    extra_cols = ['rag_prompt_response', 'rag_chosen_id']
    if logprobs_enabled:
        extra_cols += ['prob_chosen', 'prob_nochosen']
    writer.writerow(df_sample.columns.tolist() + extra_cols)

for i in df_sample.index:
    lst = ast.literal_eval(df_sample.loc[i, 'prompt'])
    rag_prompt = ''.join([j['content'] for j in lst])
    rag_prompt = 'Assume today is March 17, 2025. ' + rag_prompt
    
    result = graph.invoke({"question": rag_prompt})
    rag_prompt_response = result['answer']
    prob_chosen = result.get('prob_chosen')
    prob_nochosen = result.get('prob_nochosen')

    labels = [real_profile_id, str(df_sample.loc[i, 'profile_id'])]
    if result['answer'] == df_sample.loc[i, 'prompt_response']:
        rag_chosen_id = df_sample.loc[i, 'chosen_id']
    else:
        rag_chosen_id = [x for x in labels if x != df_sample.loc[i, 'chosen_id']][0]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = df_sample.loc[i].tolist() + [rag_prompt_response, rag_chosen_id]
        if logprobs_enabled:
            row += [prob_chosen, prob_nochosen]
        writer.writerow(row)
