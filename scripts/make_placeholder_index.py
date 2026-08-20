"""Generate a small placeholder knowledge index so the app runs end-to-end.

Produces ``knowledge/{index.sqlite, chunks.jsonl, manifest.json}`` using the
same schema ``app/retrieval.py`` expects. The SQLite file is the primary
artifact the app reads; ``chunks.jsonl`` is a human-readable copy and also the
load fallback for a fresh clone before the indexer has run.

Run from the repo root:

    python scripts/make_placeholder_index.py

To use the real knowledge repo instead, see ``scripts/update_knowledge.*`` and
HANDOFF.md §9.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import retrieval  # noqa: E402

# A few realistic ECG-doc chunks. The real index from the knowledge repo will
# be larger; this is enough to demo keyword + semantic retrieval and citations.
CHUNKS = [
    {
        "path": "Heart-Analysis-Approach/01-signal-pipeline.md",
        "heading_path": "01 \u2014 Signal Pipeline > 1. Overview",
        "section": "1",
        "kind": "paragraph",
        "start_line": 8,
        "end_line": 10,
        "content": (
            "The signal pipeline ingests raw wearable ECG recordings and produces "
            "cleaned, quality-scored segments suitable for downstream feature "
            "extraction and modeling. All processing steps operate on 30-second "
            "windows with 50% overlap."
        ),
        "tags": ["signal", "overview"],
    },
    {
        "path": "Heart-Analysis-Approach/01-signal-pipeline.md",
        "heading_path": "01 \u2014 Signal Pipeline > 2. Signal Quality Index > 2.1 SQI components",
        "section": "2.1",
        "kind": "table",
        "start_line": 65,
        "end_line": 71,
        "content": (
            "Motion artifact: h10_acc_rms_mg > 180 mg -> poor. Flatline: "
            "qrs_amp < 0.15 mV for more than 5 s -> poor. High-frequency noise: "
            "hf_noise_ratio > 0.4 -> poor. Baseline wander: bw_amp > 0.6 mV -> poor."
        ),
        "tags": ["sqi", "quality", "threshold"],
    },
    {
        "path": "Heart-Analysis-Approach/01-signal-pipeline.md",
        "heading_path": "01 \u2014 Signal Pipeline > 2. Signal Quality Index > 2.2 SQI aggregation",
        "section": "2.2",
        "kind": "paragraph",
        "start_line": 78,
        "end_line": 82,
        "content": (
            "Per-lead SQI scores are aggregated with a lead-weighted median. Any "
            "lead flagged poor is excluded from downstream fusion unless at least "
            "two leads remain; if fewer than two leads pass the quality gates the "
            "window is rejected."
        ),
        "tags": ["sqi", "aggregation"],
    },
    {
        "path": "Heart-Analysis-Approach/01-signal-pipeline.md",
        "heading_path": "01 \u2014 Signal Pipeline > 3. Preprocessing filters",
        "section": "3",
        "kind": "paragraph",
        "start_line": 95,
        "end_line": 98,
        "content": (
            "Raw signals are band-pass filtered between 0.5 Hz and 40 Hz, "
            "powerline interference is removed with a 50/60 Hz notch filter, and "
            "R-peaks are detected with a Pan-Tompkins style detector tuned for "
            "wearable devices."
        ),
        "tags": ["filtering", "preprocessing"],
    },
    {
        "path": "Heart-Analysis-Approach/02-feature-extraction.md",
        "heading_path": "02 \u2014 Feature Extraction > 1. HRV features",
        "section": "1",
        "kind": "paragraph",
        "start_line": 12,
        "end_line": 16,
        "content": (
            "HRV features include time-domain metrics (SDNN, RMSSD, pNN50) and "
            "frequency-domain metrics (LF/HF ratio, total power) computed from NN "
            "intervals within each validated window."
        ),
        "tags": ["hrv", "features"],
    },
    {
        "path": "Heart-Analysis-Approach/02-feature-extraction.md",
        "heading_path": "02 \u2014 Feature Extraction > 2. Morphology features",
        "section": "2",
        "kind": "paragraph",
        "start_line": 30,
        "end_line": 34,
        "content": (
            "Morphology features capture the QT interval, QRS duration, and "
            "ST-segment deviation per beat, summarized as medians across the "
            "window. QT is corrected with Bazett's formula."
        ),
        "tags": ["morphology", "qt", "qrs"],
    },
    {
        "path": "Models/03-classification.md",
        "heading_path": "03 \u2014 Classification > 1. Model zoo",
        "section": "1",
        "kind": "paragraph",
        "start_line": 15,
        "end_line": 19,
        "content": (
            "The classification stack compares a random forest baseline against a "
            "gradient-boosted model over the extracted features. The "
            "gradient-boosted model is the production default."
        ),
        "tags": ["model", "random-forest", "gradient-boosting"],
    },
    {
        "path": "Models/03-classification.md",
        "heading_path": "03 \u2014 Classification > 3. Validation",
        "section": "3",
        "kind": "paragraph",
        "start_line": 42,
        "end_line": 46,
        "content": (
            "Models are validated with stratified 5-fold cross-validation. "
            "Reported metrics are AUROC and F1; the decision threshold is chosen "
            "to maximize Youden's J index on the validation folds."
        ),
        "tags": ["validation", "auroc", "threshold"],
    },
]


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - git may be unavailable
        return "unknown"


def build_chunks() -> list[dict]:
    part_counter: dict[tuple[str, str], int] = {}
    chunks = []
    for raw in CHUNKS:
        key = (raw["path"], raw["section"])
        part = part_counter.get(key, 0)
        part_counter[key] = part + 1
        chunk = dict(raw)
        chunk["id"] = f"{raw['path']}::{raw['section']}::part{part}"
        chunks.append(chunk)
    return chunks


def main() -> None:
    out_dir = ROOT / "knowledge"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build embeddings with the same backend the app will use at query time
    # (EMBED_BACKEND env or auto -> fastembed when installed, else local), so
    # the dimension guard matches and semantic search actually runs.
    embedder = retrieval.get_embedder()
    embed_backend = "fastembed" if isinstance(embedder, retrieval.FastEmbedEmbedder) else "local"
    embed_dim = embedder.DIM
    chunks = build_chunks()

    # JSONL — portable copy; embeddings are recomputed at load time.
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # SQLite — the primary artifact.
    db_path = out_dir / "index.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, path TEXT, heading_path TEXT, "
        "section TEXT, kind TEXT, start_line INTEGER, end_line INTEGER, content "
        "TEXT, tags TEXT, embedding BLOB)"
    )
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id, content)")
    for chunk in chunks:
        vec = np.asarray(embedder.embed_query(chunk["content"]), dtype=np.float32)
        conn.execute(
            "INSERT INTO chunks (id, path, heading_path, section, kind, "
            "start_line, end_line, content, tags, embedding) VALUES "
            "(?,?,?,?,?,?,?,?,?,?)",
            (
                chunk["id"], chunk["path"], chunk["heading_path"],
                chunk["section"], chunk["kind"], chunk["start_line"],
                chunk["end_line"], chunk["content"], ",".join(chunk["tags"]),
                vec.tobytes(),
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts (id, content) VALUES (?,?)",
            (chunk["id"], chunk["content"]),
        )
    conn.commit()
    conn.close()

    manifest = {
        "schema_version": retrieval.SCHEMA_VERSION,
        "index_version": retrieval.INDEX_VERSION,
        "embed_backend": embed_backend,
        "embed_dim": embed_dim,
        "git_sha": git_sha(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "note": "Placeholder index generated by scripts/make_placeholder_index.py",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(chunks)} chunks to {db_path} (+ chunks.jsonl, manifest.json)")


if __name__ == "__main__":
    main()
