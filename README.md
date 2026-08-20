# chatbot-web

A deployable FastAPI web app that answers questions about the **Longitudinal ECG project** with grounded, cited answers. A user asks a question, the server runs **hybrid RAG** over a vendored SQLite index of the project docs, calls an **OpenAI-compatible LLM** (OpenAI or DeepSeek) server-side, and returns an answer with **citations** to specific files, sections, and line ranges.

The full operating guide lives in [HANDOFF.md](HANDOFF.md) — read it before extending or deploying.

## Quick start

```powershell
pip install -r requirements.txt
$env:LLM_API_KEY = "..."            # required for /chat; /search etc. work without it
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

This repo ships a generated placeholder index (`knowledge/chunks.jsonl` + `manifest.json`), so the app runs immediately. To also generate the SQLite index:

```powershell
python scripts/make_placeholder_index.py
```

To pull the real index from the knowledge repo instead, run `scripts/update_knowledge.ps1` (Windows) or `./scripts/update_knowledge.sh` (bash) — set the `KNOWLEDGE_REPO` env var or use the script's default (`C:\longitudinal_ecg`).

**Keeping the index fresh is automated**: `.github/workflows/sync-knowledge.yml` pulls the latest index from the `MicahHeneveld/longitudinal_ecg` repo daily (and on demand via the workflow's "Run workflow" button) and pushes it back to `main` when it changes — Render redeploys automatically. See [HANDOFF.md](HANDOFF.md) §9 for details.

If `fastembed` is installed (`pip install fastembed`), retrieval uses `BAAI/bge-small-en-v1.5` (384-dim) embeddings; otherwise a deterministic local hash embedder (512-dim) is used, with a dimension guard that degrades to keyword-only search when the index was built with a different backend.

## Configuration (env vars)

| Var | Required | Default | Notes |
|---|---|---|---|
| `LLM_API_KEY` | yes | — | Never ship to the browser; set as a Render secret. |
| `LLM_BASE_URL` | no | `https://api.openai.com/v1` | DeepSeek: `https://api.deepseek.com`. |
| `LLM_MODEL` | no | `gpt-4o-mini` | e.g. `deepseek-chat`, `gpt-4o-mini`. |
| `GITHUB_REPO` | no | `""` | e.g. `you/longitudinal_ecg` enables GitHub blob citation URLs. |
| `GITHUB_BRANCH` | no | `main` | Branch used for citation URLs. |
| `CORS_ORIGINS` | no | `*` | Comma-separated allowed origins. |
| `KNOWLEDGE_DB` | no | `knowledge/index.sqlite` | Override index location. |
| `EMBED_BACKEND` | no | `auto` | `auto`/`fastembed` use fastembed; `local` forces the hash embedder. |

Copy `.env.example` to `.env` as a reference (the app reads real environment variables, not `.env`).

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness: `{ok, chunks, dim, embedder, llm_configured, llm_model}` |
| GET | `/search?q=&k=` | raw hybrid retrieval, no LLM |
| GET | `/doc/{path}` | full markdown of one document |
| GET | `/chunks?path=&section=` | chunk provenance for a doc |
| GET | `/map` | navigation manifest (doc graph) |
| POST | `/chat` | `{query, history, k}` -> `{answer, citations, query}` |
| GET | `/` | the chat UI |

## Deploy on Render

1. Push this folder as its own git repo.
2. Render → **New → Web Service** (not Static Site) → connect the repo.
3. `render.yaml` describes the service: build `pip install -r requirements.txt`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Set `LLM_API_KEY` as a secret; `LLM_BASE_URL`, `LLM_MODEL`, `GITHUB_REPO`, `CORS_ORIGINS` as needed.

## Layout

```
app/
├── main.py          FastAPI app (endpoints, lifespan index load, CORS)
├── chat.py          LLM proxy + prompt assembly (OpenAI-compatible)
├── retrieval.py     vendored hybrid retrieval (self-contained)
└── static/          chat UI: index.html, styles.css, app.js
knowledge/           vendored index: index.sqlite, chunks.jsonl, manifest.json
scripts/
├── make_placeholder_index.py   generate the demo placeholder index
├── update_knowledge.ps1        pull latest index from the knowledge repo (Windows)
└── update_knowledge.sh         same (bash)
render.yaml          Render Blueprint (Web Service)
requirements.txt     fastapi, uvicorn, openai, numpy
```
