import os
import ast
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import random
random.seed(2025)
os.chdir(
    os.path.dirname(os.path.abspath(__file__))
)
import numpy as np
import argparse
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

    args = parser.parse_args()
    if args.start is not None and args.end is None:
        parser.error("--start requires --end.")

    return args

args = read_arguments()
real_profile_id, account = (
    args.real_profile_id,
    args.account,
)

llm = init_chat_model("gpt-5-nano", temperature = 1, reasoning_effort = 'low', model_provider="openai")

# llm = init_chat_model("gpt-5-nano",api_key=os.environ.get(account), temperature = 1, reasoning_effort = 'low', model_provider="openai")
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

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
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

resample_500index = np.load("../data/rag_langchain_profile_ids500.npy", allow_pickle=True)
df = pd.read_csv(f"../output/{real_profile_id}_fixreal5000.csv")

df_sample = df[df['profile_id'].isin(resample_500index)].reset_index(drop=True)
assert len(df_sample) == 500

csv_path = f"../output/rag_langchain_{real_profile_id}_fixreal500.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(df_sample.columns.tolist() + ['rag_prompt_response', 'rag_chosen_id'])

for i in df_sample.index:
    lst = ast.literal_eval(df_sample.loc[i, 'prompt'])
    rag_prompt = ''.join([j['content'] for j in lst])
    rag_prompt = 'Assume today is March 17, 2025. ' + rag_prompt
    
    result = graph.invoke({"question": rag_prompt})
    rag_prompt_response = result['answer']

    labels = [real_profile_id, str(df_sample.loc[i, 'profile_id'])]
    if result['answer'] == df_sample.loc[i, 'prompt_response']:
        rag_chosen_id = df_sample.loc[i, 'chosen_id']
    else:
        rag_chosen_id = [x for x in labels if x != df_sample.loc[i, 'chosen_id']][0]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(df_sample.loc[i].tolist() + [rag_prompt_response, rag_chosen_id])
