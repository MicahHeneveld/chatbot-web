"""Vendored hybrid retrieval over the ECG knowledge index.

Self-contained copy of the knowledge repo's ``tools/indexer`` retrieval
modules (search/embed/store/citations). It has no runtime dependency on the
knowledge repo.

Primary artifact: ``knowledge/index.sqlite`` with a ``chunks`` table plus an
FTS5 ``chunks_fts`` virtual table. When the SQLite file is absent but
``knowledge/chunks.jsonl`` exists (e.g. a fresh clone before the indexer has
run), the index is rebuilt in-memory from the JSONL so the app still runs
end-to-end. Retrieval never writes upstream.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
INDEX_VERSION = 1

RRF_K = 60
DEFAULT_K = 8
MAX_K = 50
DEFAULT_DB = "knowledge/index.sqlite"

DEFAULT_GITHUB_BRANCH = "main"

_EXCERPT_CHARS = 320


class EmbeddingBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested embedding backend is not installed."""


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; keeps identifiers like ``h10_acc_rms_mg`` intact."""
    return re.findall(r"[a-z0-9_]+", text.lower())


class LocalHashEmbedder:
    """Deterministic feature-hashing embedder (512-dim). No network, no deps."""

    DIM = 512

    @staticmethod
    def _hash(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for token in _tokenize(text):
                h = self._hash(token)
                vec[h % self.DIM] += 1.0 if (h >> 8) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FastEmbedEmbedder:
    """fastembed backend: BAAI/bge-small-en-v1.5, 384-dim. Downloaded on first use."""

    DIM = 384
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingBackendUnavailable(
                "fastembed is not installed; run `pip install fastembed` "
                "or set EMBED_BACKEND=local"
            ) from exc
        self._model = TextEmbedding(model_name=self.MODEL_NAME)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embedder(backend: str | None = None):
    """Instantiate the configured embedding backend.

    ``auto``/``fastembed`` use fastembed when installed and fall back to the
    local hash embedder otherwise; ``local`` always uses the hash embedder.
    """
    backend = (backend or os.environ.get("EMBED_BACKEND") or "auto").lower()
    if backend in ("auto", "fastembed"):
        try:
            return FastEmbedEmbedder()
        except EmbeddingBackendUnavailable:
            if backend == "fastembed":
                raise
            return LocalHashEmbedder()
    if backend == "local":
        return LocalHashEmbedder()
    raise ValueError(f"unknown EMBED_BACKEND: {backend!r}")


class Index:
    """In-memory view of the knowledge index with hybrid search."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or os.environ.get("KNOWLEDGE_DB", DEFAULT_DB))
        self.conn: sqlite3.Connection | None = None
        self.chunks: list[dict] = []
        self.chunks_by_id: dict[str, dict] = {}
        self.embeddings: np.ndarray | None = None
        self.dim: int | None = None
        self.manifest: dict = {}
        self.embedder = get_embedder()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, db_path: str | Path | None = None) -> "Index":
        index = cls(db_path)
        index._load_manifest()
        sqlite_file = index.db_path
        if sqlite_file.exists():
            index._load_sqlite(sqlite_file)
        elif (sqlite_file.parent / "chunks.jsonl").exists():
            index._load_jsonl(sqlite_file.parent / "chunks.jsonl")
        else:
            raise FileNotFoundError(
                f"No knowledge index found at {sqlite_file}. Run "
                "`python scripts/make_placeholder_index.py` to generate one, "
                "or see HANDOFF.md §9 to refresh from the knowledge repo."
            )
        return index

    def _load_manifest(self) -> None:
        manifest_file = self.db_path.parent / "manifest.json"
        if manifest_file.exists():
            try:
                self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.manifest = {}

    def _load_sqlite(self, sqlite_file: Path) -> None:
        uri = Path(sqlite_file).resolve().as_uri() + "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)
        try:
            rows = self.conn.execute(
                "SELECT id, path, heading_path, section, kind, start_line, "
                "end_line, content, tags, embedding FROM chunks"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ValueError(
                f"{sqlite_file} is not a valid knowledge index: {exc}"
            ) from exc
        self._ingest(rows)

    def _load_jsonl(self, jsonl_file: Path) -> None:
        # Rebuild the SQLite layout in-memory so FTS5 keyword search works the
        # same way as on the file-backed index.
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE chunks (id TEXT PRIMARY KEY, path TEXT, heading_path "
            "TEXT, section TEXT, kind TEXT, start_line INTEGER, end_line "
            "INTEGER, content TEXT, tags TEXT, embedding BLOB)"
        )
        self.conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id, content)")
        local = LocalHashEmbedder()
        rows = []
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["tags"] = ",".join(rec.get("tags") or [])
            embedding = rec.get("embedding")
            if embedding:
                vec = np.asarray(embedding, dtype=np.float32)
            else:
                vec = np.asarray(local.embed_query(rec["content"]), dtype=np.float32)
            blob = vec.tobytes()
            self.conn.execute(
                "INSERT INTO chunks (id, path, heading_path, section, kind, "
                "start_line, end_line, content, tags, embedding) VALUES "
                "(?,?,?,?,?,?,?,?,?,?)",
                (
                    rec["id"], rec["path"], rec.get("heading_path", ""),
                    rec.get("section", ""), rec.get("kind", ""),
                    rec.get("start_line"), rec.get("end_line"),
                    rec["content"], rec["tags"], blob,
                ),
            )
            self.conn.execute(
                "INSERT INTO chunks_fts (id, content) VALUES (?,?)",
                (rec["id"], rec["content"]),
            )
            rows.append(
                (rec["id"], rec["path"], rec.get("heading_path", ""),
                 rec.get("section", ""), rec.get("kind", ""),
                 rec.get("start_line"), rec.get("end_line"),
                 rec["content"], rec["tags"], blob)
            )
        self.conn.commit()
        self._ingest(rows)

    def _ingest(self, rows: list[tuple]) -> None:
        self.chunks = []
        self.chunks_by_id = {}
        blobs: list[bytes | None] = []
        for row in rows:
            cid, path, heading_path, section, kind, start_line, end_line, content, tags, emb = row
            chunk = {
                "id": cid,
                "path": path,
                "heading_path": heading_path or "",
                "section": section or "",
                "kind": kind or "",
                "start_line": start_line,
                "end_line": end_line,
                "content": content or "",
                "tags": tags or "",
            }
            self.chunks.append(chunk)
            self.chunks_by_id[cid] = chunk
            blobs.append(emb)
        if blobs and all(b is not None for b in blobs):
            d = len(blobs[0]) // 4
            if d > 0:
                mat = np.empty((len(blobs), d), dtype=np.float32)
                for i, b in enumerate(blobs):
                    mat[i] = np.frombuffer(b, dtype=np.float32)
                self.embeddings = mat
                self.dim = d
            else:
                self.embeddings = None
                self.dim = None

    # ------------------------------------------------------------- properties

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def embedding_dim(self) -> int | None:
        return self.dim

    @property
    def embedder_name(self) -> str:
        return "fastembed" if isinstance(self.embedder, FastEmbedEmbedder) else "local"

    # -------------------------------------------------------------- search

    def _keyword_terms(self, query: str) -> list[str]:
        terms = []
        for token in _tokenize(query):
            if token not in terms:
                terms.append(token)
        return terms

    @staticmethod
    def _fts_match_expr(terms: list[str]) -> str:
        return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def _keyword_search(self, terms: list[str], limit: int) -> list[str]:
        """FTS5 bm25 ranking, with a substring fallback for rare terms."""
        if not terms or self.conn is None:
            return []
        results: list[str] = []
        try:
            expr = self._fts_match_expr(terms)
            rows = self.conn.execute(
                "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                (expr, limit),
            ).fetchall()
            results = [r[0] for r in rows]
        except sqlite3.Error:
            results = []
        if len(results) < limit:
            extra = self._substring_search(terms, limit - len(results))
            for cid in extra:
                if cid not in results:
                    results.append(cid)
        return results[:limit]

    def _substring_search(self, terms: list[str], limit: int) -> list[str]:
        if not terms or self.conn is None:
            return []

        def escape_like(t: str) -> str:
            return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        conds = " OR ".join(["content LIKE ? ESCAPE '\\'"] * len(terms))
        params = [f"%{escape_like(t)}%" for t in terms]
        rows = self.conn.execute(
            f"SELECT id, content FROM chunks WHERE {conds}", params
        ).fetchall()
        scored = []
        for cid, content in rows:
            hits = sum(1 for t in terms if t in content)
            if hits > 0:
                scored.append((hits, cid))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [cid for _, cid in scored[:limit]]

    def _semantic_search(self, query_vec: list[float], limit: int) -> list[tuple[str, float]]:
        """Cosine similarity. Empty when the query-vector dim mismatches the index."""
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []
        if self.dim is None or len(query_vec) != self.dim:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []
        q = q / q_norm
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:limit]
        return [(self.chunks[i]["id"], float(scores[i])) for i in order]

    @staticmethod
    def _rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda pair: -pair[1])

    def hybrid_search(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        """Run keyword + semantic channels, fuse with reciprocal rank fusion."""
        k = max(1, min(int(k), MAX_K))
        terms = self._keyword_terms(query)
        keyword_ids = self._keyword_search(terms, k)

        query_vec = self.embedder.embed_query(query)
        semantic_ids = [doc_id for doc_id, _ in self._semantic_search(query_vec, k)]

        if semantic_ids:
            fused = self._rrf([keyword_ids, semantic_ids])
        else:
            fused = [
                (doc_id, 1.0 / (RRF_K + rank + 1))
                for rank, doc_id in enumerate(keyword_ids)
            ]
        return [
            self.to_citation(doc_id, score) for doc_id, score in fused[:k]
        ]

    # ------------------------------------------------------------ citations

    def to_citation(self, chunk_id: str, score: float) -> dict:
        chunk = self.chunks_by_id[chunk_id]
        return {
            "id": chunk["id"],
            "path": chunk["path"],
            "section": chunk["section"],
            "heading_path": chunk["heading_path"],
            "kind": chunk["kind"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "excerpt": self._excerpt(chunk["content"]),
            "url": self._citation_url(chunk),
            "score": round(score, 4),
        }

    @staticmethod
    def _excerpt(content: str, max_chars: int = _EXCERPT_CHARS) -> str:
        content = " ".join(content.split())
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 3].rstrip() + "..."

    def _citation_url(self, chunk: dict) -> str:
        repo = os.environ.get("GITHUB_REPO", "").strip()
        if not repo:
            return ""
        branch = os.environ.get("GITHUB_BRANCH", DEFAULT_GITHUB_BRANCH) or DEFAULT_GITHUB_BRANCH
        return (
            f"https://github.com/{repo}/blob/{branch}/{chunk['path']}"
            f"#L{chunk['start_line']}-L{chunk['end_line']}"
        )

    # -------------------------------------------------------------- helpers

    def doc_sections(self, path: str) -> list[dict]:
        """Chunks of one document, ordered by line number."""
        return sorted(
            (c for c in self.chunks if c["path"] == path),
            key=lambda c: c["start_line"] or 0,
        )

    def doc_markdown(self, path: str) -> str:
        """Reconstruct a readable markdown document from its chunks."""
        chunks = self.doc_sections(path)
        if not chunks:
            raise KeyError(path)
        parts: list[str] = []
        last_heading: str | None = None
        for chunk in chunks:
            heading = (chunk.get("heading_path") or "").strip()
            if heading and heading != last_heading:
                parts.append(f"## {heading}")
                last_heading = heading
            if chunk.get("content"):
                parts.append(chunk["content"].strip())
        return "\n\n".join(parts)

    def doc_map(self) -> dict:
        """Navigation manifest: doc graph derived from chunk provenance."""
        docs: dict[str, dict] = {}
        for chunk in self.chunks:
            path = chunk["path"]
            doc = docs.setdefault(path, {"path": path, "headings": [], "sections": [], "chunks": 0})
            heading = chunk.get("heading_path")
            if heading and heading not in doc["headings"]:
                doc["headings"].append(heading)
            section = chunk.get("section")
            if section and section not in doc["sections"]:
                doc["sections"].append(section)
            doc["chunks"] += 1
        for doc in docs.values():
            doc["headings"].sort()
            doc["sections"].sort()
        return {"docs": sorted(docs.values(), key=lambda d: d["path"])}
