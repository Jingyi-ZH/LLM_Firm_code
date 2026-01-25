#!/usr/bin/env bash
# ./wayback_fetch.sh -i urllist.txt -t 20250315 [--single-file | --mirror]
set -euo pipefail

# ============ Default config ============
INPUT_FILE=""
OUT_DIR="wayback_downloads"
TS="20250315"            # Target timestamp (YYYYMMDD or YYYYMMDDhhmmss). Empty = let Wayback choose.
MIRROR=false             # true: full-page mirror (with assets); false: HTML only
SINGLE_FILE=false        # true: use monolith to build a single self-contained HTML
DELAY=1                  # Delay between requests (seconds)
PARALLEL=1               # Parallelism for snapshot lookup
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

# Download retry settings
WGET_TRIES=10
WGET_WAITRETRY=2
WGET_TIMEOUT=30
WGET_READ_TIMEOUT=30

usage() {
  cat <<EOF
Usage: $0 -i urllist.txt [-o output_dir] [-t timestamp] [--single-file | --mirror] [--delay seconds] [--parallel N]

Required:
  -i, --input         Text file with URLs (one per line)

Optional:
  -o, --outdir        Output root directory (default: ${OUT_DIR})
  -t, --timestamp     Target timestamp (YYYYMMDD or YYYYMMDDhhmmss), default ${TS}
  -m, --mirror        Full-page mirror (download page and assets to a directory)
  -s, --single-file   Build a single self-contained HTML (requires monolith)
      --delay         Delay between requests (seconds), default ${DELAY}
      --parallel      Parallel snapshot lookups, default ${PARALLEL} (suggest <= 4)
  -h, --help          Show help

Modes (choose one):
  default (no flag): HTML only
  --mirror          : Full-page mirror (directory)
  --single-file     : Single-file HTML via monolith

Dependencies: curl, jq, wget; monolith required for --single-file
Examples:
  $0 -i urllist.txt -t 20250315                    # HTML only
  $0 -i urllist.txt -t 20250315 --mirror           # Full-page mirror
  $0 -i urllist.txt -t 20250315 --single-file      # Single-file HTML
EOF
}

# ============ Argument parsing ============
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) INPUT_FILE="$2"; shift 2;;
    -o|--outdir) OUT_DIR="$2"; shift 2;;
    -t|--timestamp) TS="$2"; shift 2;;
    -m|--mirror) MIRROR=true; shift 1;;
    -s|--single-file) SINGLE_FILE=true; shift 1;;
    --delay) DEPLAY="$2"; shift 2;;        # Accept common typo
    --parallel) PARALLEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done
if [[ -n "${DEPLAY:-}" ]]; then DELAY="$DEPLAY"; fi

# Mutual exclusivity checks
if $MIRROR && $SINGLE_FILE; then
  echo "Error: --mirror and --single-file cannot be used together"; exit 1
fi

if [[ -z "${INPUT_FILE}" || ! -f "${INPUT_FILE}" ]]; then
  echo "Error: -i/--input must point to an existing file"; exit 1
fi

mkdir -p "${OUT_DIR}"
: > "${OUT_DIR}/_not_found.txt"
: > "${OUT_DIR}/_failed.txt"

# ============ Dependency checks ============
for cmd in curl jq wget; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing dependency: $cmd not installed"; exit 1
  fi
done
if $SINGLE_FILE && ! command -v monolith >/dev/null 2>&1; then
  echo "Missing dependency: monolith required for --single-file"; exit 1
fi

# ============ Timestamp normalization ============
# If YYYYMMDD is provided, expand to 23:59:59 for that day.
norm_ts="$TS"
if [[ -n "$norm_ts" && "$norm_ts" =~ ^[0-9]{8}$ ]]; then
  norm_ts="${norm_ts}235959"
fi

# ============ URL trailing slash toggle ============
toggle_trailing_slash() {
  local u="$1"
  local core="${u%%[\?#]*}"
  if [[ "$core" == */ ]]; then
    echo "${core%/}"
  else
    echo "${core}/"
  fi
}

# ============ Wayback lookup with fallback ============
# try_lookup <url> <ts_or_empty> -> output "url \t snapshot_url \t ts" or "url \t \t"
try_lookup() {
  local url="$1"
  local ts="$2"
  local q="https://archive.org/wayback/available?url=$(printf '%s' "$url" | sed 's/ /%20/g')"
  if [[ -n "$ts" ]]; then q="${q}&timestamp=${ts}"; fi

  local json available snap_url snap_ts
  json=$(curl -sS -A "$UA" "$q" || true)
  available=$(printf '%s' "$json" | jq -r '.archived_snapshots.closest.available // false' 2>/dev/null || echo "false")
  if [[ "${available}" != "true" ]]; then
    echo -e "${url}\t\t"; return
  fi
  snap_url=$(printf '%s' "$json" | jq -r '.archived_snapshots.closest.url // empty')
  snap_ts=$(printf '%s' "$json" | jq -r '.archived_snapshots.closest.timestamp // empty')
  echo -e "${url}\t${snap_url}\t${snap_ts}"
}

# lookup_snapshot <origin_url> -> fallback order: ts -> toggle-slash -> no-ts -> no-ts+toggle
lookup_snapshot() {
  local url="$1"

  # 1) Lookup with normalized timestamp
  local r
  r=$(try_lookup "$url" "$norm_ts")
  if [[ -n "$(printf '%s' "$r" | awk -F'\t' '{print $2}')" ]]; then
    echo "$r"; return
  fi

  # 2) Toggle trailing slash and retry
  local toggled
  toggled="$(toggle_trailing_slash "$url")"
  if [[ "$toggled" != "$url" ]]; then
    r=$(try_lookup "$toggled" "$norm_ts")
    if [[ -n "$(printf '%s' "$r" | awk -F'\t' '{print $2}')" ]]; then
      echo "$r"; return
    fi
  fi

  # 3) Lookup without timestamp
  r=$(try_lookup "$url" "")
  if [[ -n "$(printf '%s' "$r" | awk -F'\t' '{print $2}')" ]]; then
    echo "$r"; return
  fi

  # 4) No timestamp + toggled slash
  if [[ "$toggled" != "$url" ]]; then
    r=$(try_lookup "$toggled" "")
    if [[ -n "$(printf '%s' "$r" | awk -F'\t' '{print $2}')" ]]; then
      echo "$r"; return
    fi
  fi

  # No match found
  echo -e "${url}\t\t"
}

export -f toggle_trailing_slash try_lookup lookup_snapshot
export TS norm_ts UA

# ============ Step 1: Resolve snapshots ============
TMP_MAP="$(mktemp)"
cat "${INPUT_FILE}" \
  | sed -e 's/\r$//' -e '/^[[:space:]]*$/d' \
  | xargs -I{} -P "${PARALLEL}" bash -c 'lookup_snapshot "$@"' _ {} \
  > "${TMP_MAP}"

TOTAL=$(wc -l < "${TMP_MAP}" | tr -d ' ')
FOUND=$(awk -F'\t' 'NF>=2 && $2!="" {c++} END{print c+0}' "${TMP_MAP}")
MISS=$((TOTAL-FOUND))
echo "TOTAL: ${TOTAL} URLs; Found snapshots: ${FOUND}; Not found: ${MISS}"
awk -F'\t' 'NF<2 || $2=="" {print $1}' "${TMP_MAP}" >> "${OUT_DIR}/_not_found.txt"

# ============ Step 2: Download ============
download_one() {
  local origin_url="$1"
  local snapshot_url="$2"
  local snapshot_ts="$3"

  if [[ -z "${snapshot_url}" ]]; then
    echo "[NOT FOUND] ${origin_url}"; return
  fi

  # Force https to avoid http rejection
  snapshot_url="${snapshot_url/http:\/\//https://}"

  local domain d
  domain=$(printf '%s' "$origin_url" | awk -F/ '{print $3}')
  d="${OUT_DIR}/${domain}/${snapshot_ts:-unknown}"
  mkdir -p "$d"

  local base
  base="$(printf '%s' "$origin_url" | sed 's/[?#].*$//' | awk -F'/' '{print $NF}')"
  if [[ -z "$base" ]]; then base="index"; fi
  base="$(printf '%s' "$base" | sed 's/[^A-Za-z0-9._-]/_/g')"
  if [[ ! "$base" =~ \.html?$ ]]; then base="${base}.html"; fi

  echo "[Downloading] ${origin_url}"
  echo "      Snapshot: ${snapshot_url}"
  echo "      Saving to: ${d}/${base}"

  if $MIRROR; then
    wget \
      --user-agent="$UA" \
      --page-requisites \
      --convert-links \
      --adjust-extension \
      --timestamping \
      --no-parent \
      --directory-prefix="$d" \
      --domains web.archive.org \
      --tries="${WGET_TRIES}" \
      --retry-connrefused \
      --wait="${DELAY}" \
      --waitretry="${WGET_WAITRETRY}" \
      --timeout="${WGET_TIMEOUT}" \
      --read-timeout="${WGET_READ_TIMEOUT}" \
      --random-wait=off \
      --no-verbose \
      "$snapshot_url" \
    || { echo "$origin_url" >> "${OUT_DIR}/_failed.txt"; echo "      [FAILED: recorded in _failed.txt]"; }

  elif $SINGLE_FILE; then
    monolith "$snapshot_url" -o "${d}/${base}" \
    || { echo "$origin_url" >> "${OUT_DIR}/_failed.txt"; echo "      [FAILED: recorded in _failed.txt]"; }

  else
    wget \
      --user-agent="$UA" \
      --tries="${WGET_TRIES}" \
      --retry-connrefused \
      --wait="${DELAY}" \
      --waitretry="${WGET_WAITRETRY}" \
      --timeout="${WGET_TIMEOUT}" \
      --read-timeout="${WGET_READ_TIMEOUT}" \
      --random-wait=off \
      --no-verbose \
      --output-document="${d}/${base}" \
      "$snapshot_url" \
    || { echo "$origin_url" >> "${OUT_DIR}/_failed.txt"; echo "      [FAILED: recorded in _failed.txt]"; }
  fi
}
export -f download_one
export OUT_DIR DELAY MIRROR SINGLE_FILE UA WGET_TRIES WGET_WAITRETRY WGET_TIMEOUT WGET_READ_TIMEOUT

while IFS=$'\t' read -r ORI SURL STS; do
  download_one "${ORI}" "${SURL}" "${STS}"
done < "${TMP_MAP}"

echo "COMPLETED! Output directory: ${OUT_DIR}"
echo "Not found URLs saved to: ${OUT_DIR}/_not_found.txt"
echo "Failed downloads saved to: ${OUT_DIR}/_failed.txt"

rm -f "${TMP_MAP}"
