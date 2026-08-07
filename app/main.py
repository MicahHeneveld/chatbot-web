"""ECG Knowledge Chatbot — FastAPI app (endpoints, lifespan index load, CORS)."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import chat, retrieval

logger = logging.getLogger("ecg-chatbot")

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    history: list[dict[str, str]] = []
    k: int = Field(default=retrieval.DEFAULT_K, ge=1, le=retrieval.MAX_K)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = None
    app.state.index_error = None
    try:
        app.state.index = retrieval.Index.load()
        logger.info(
            "knowledge index loaded: %d chunks, dim=%s, embedder=%s",
            app.state.index.chunk_count,
            app.state.index.embedding_dim,
            app.state.index.embedder_name,
        )
    except Exception as exc:  # noqa: BLE001 - surface any load failure in /health
        app.state.index_error = str(exc)
        logger.error("failed to load knowledge index: %s", exc)
    yield


app = FastAPI(title="ECG Knowledge Chatbot", lifespan=lifespan)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _require_index(request: Request) -> retrieval.Index:
    index = request.app.state.index
    if index is None:
        detail = request.app.state.index_error or "unknown error"
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge index not loaded: {detail}",
        )
    return index


@app.get("/health")
async def health(request: Request):
    index = request.app.state.index
    llm = chat.llm_config()
    return {
        "ok": index is not None,
        "chunks": index.chunk_count if index else 0,
        "dim": index.embedding_dim if index else None,
        "embedder": index.embedder_name if index else None,
        "llm_configured": bool(llm["api_key"]),
        "llm_model": llm["model"],
        "index_error": request.app.state.index_error,
    }


@app.get("/search")
async def search(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(default=retrieval.DEFAULT_K, ge=1, le=retrieval.MAX_K),
):
    index = _require_index(request)
    citations = index.hybrid_search(q, k=k)
    return {"query": q, "citations": citations}


@app.get("/doc/{path:path}")
async def get_doc(request: Request, path: str):
    index = _require_index(request)
    try:
        markdown = index.doc_markdown(path)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"document not found: {path}")
    return JSONResponse({"path": path, "markdown": markdown})


@app.get("/chunks")
async def get_chunks(
    request: Request,
    path: str = Query(...),
    section: str | None = Query(default=None),
):
    index = _require_index(request)
    items = index.doc_sections(path)
    if section is not None:
        items = [c for c in items if c.get("section") == section]
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"no chunks for path={path!r} section={section!r}",
        )
    return {"path": path, "chunks": items}


@app.get("/map")
async def doc_map(request: Request):
    index = _require_index(request)
    return index.doc_map()


@app.post("/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    index = _require_index(request)
    citations = index.hybrid_search(body.query, k=body.k)
    messages = chat.compose_messages(body.query, body.history, citations)
    try:
        answer = chat.llm_chat(messages)
    except chat.LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - provider errors are surfaced to the client
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}")
    return {"answer": answer, "citations": citations, "query": body.query}


@app.get("/")
async def index_page():
    return FileResponse(STATIC_DIR / "index.html")
