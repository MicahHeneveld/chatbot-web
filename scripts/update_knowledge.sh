#!/usr/bin/env bash
# Pull the latest vendored knowledge index from the knowledge repo (bash).
# One-way sync: the knowledge repo is the source of truth; this script only copies.
# Usage: ./scripts/update_knowledge.sh
set -euo pipefail

# --- Placeholder: point this at the knowledge repo checkout ---
# Set the KNOWLEDGE_REPO env var, or edit the path below.
KNOWLEDGE_REPO="${KNOWLEDGE_REPO:-/path/to/longitudinal_ecg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/../knowledge"

if [ ! -d "${KNOWLEDGE_REPO}/knowledge" ]; then
  echo "ERROR: knowledge source not found at ${KNOWLEDGE_REPO}/knowledge" >&2
  echo "Set KNOWLEDGE_REPO or edit this script." >&2
  exit 1
fi

for f in index.sqlite chunks.jsonl manifest.json; do
  if [ ! -f "${KNOWLEDGE_REPO}/knowledge/${f}" ]; then
    echo "ERROR: missing ${f} in the knowledge repo" >&2
    exit 1
  fi
  cp "${KNOWLEDGE_REPO}/knowledge/${f}" "${DEST_DIR}/${f}"
  echo "Copied ${f}"
done

echo "Knowledge index refreshed from ${KNOWLEDGE_REPO}"
