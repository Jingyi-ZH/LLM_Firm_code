#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RAW_DIR="${RAG_RAW_DIR:-$PROJECT_ROOT/RAG/raw_material/iPhone16lineup}"
COLLECTED_JSONL="${RAG_COLLECTED_JSONL:-$PROJECT_ROOT/RAG/retrieved_material/iPhone16lineup/collected.jsonl}"
INDEX_FAISS="${RAG_INDEX_FAISS:-$PROJECT_ROOT/RAG/retrieved_material/iPhone16lineup/index.faiss}"
INDEX_META="${RAG_INDEX_META:-$PROJECT_ROOT/RAG/retrieved_material/iPhone16lineup/records.jsonl}"
QUESTIONS_FILE="${RAG_QUESTIONS:-$PROJECT_ROOT/RAG/validation/iPhone16lineup/qs.txt}"
CONTEXTS_OUT="${RAG_CONTEXTS_OUT:-$PROJECT_ROOT/RAG/validation/iPhone16lineup/contexts.jsonl}"

echo "Project root: $PROJECT_ROOT"
echo "Collecting original text corpus..."
args_ingest=(
  --root "$RAW_DIR"
  --out "$COLLECTED_JSONL"
)
python "$PROJECT_ROOT/RAG/rag_build.py" ingest "${args_ingest[@]}"

echo "Chunking and embedding..."
args_index=(
  --input "$COLLECTED_JSONL"
  --faiss "$INDEX_FAISS"
  --meta "$INDEX_META"
)
python "$PROJECT_ROOT/RAG/rag_build.py" index "${args_index[@]}"

echo "Retrieving contexts for questions..."
args_retrieve=(
  --faiss "$INDEX_FAISS"
  --meta "$INDEX_META"
  --questions "$QUESTIONS_FILE"
  --out "$CONTEXTS_OUT"
  --k 3
)
python "$PROJECT_ROOT/RAG/rag_build.py" retrieve "${args_retrieve[@]}"

echo "SUCCESS! All steps completed successfully."
