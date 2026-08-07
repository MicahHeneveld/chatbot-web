"""LLM proxy + prompt assembly (OpenAI-compatible).

DeepSeek and OpenAI both expose OpenAI-compatible APIs, so this module uses a
single thin client (``openai`` package pointed at ``LLM_BASE_URL``). Switching
providers is a config change, not a code change.
"""

from __future__ import annotations

import os
from typing import Any

import openai

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_HISTORY = 10
TIMEOUT_SECONDS = 120.0

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the Longitudinal "
    "ECG project documentation.\n"
    "Answer ONLY from the numbered sources provided in the user message. Cite "
    "the source for every factual claim using inline markers like [1] or [2] "
    "that match the source numbers.\n"
    "Never invent thresholds, file names, section numbers, or any other detail "
    "that is not in the provided context. If the context does not contain the "
    "answer, say so explicitly rather than guessing.\n"
    "Keep the answer concise and technical, matching the tone of the source docs."
)


class LLMNotConfigured(Exception):
    """Raised when LLM_API_KEY is missing from the server environment."""


def llm_config() -> dict:
    return {
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "api_key": os.environ.get("LLM_API_KEY", "").strip(),
    }


def compose_messages(
    query: str,
    history: list[dict[str, str]],
    citations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """System prompt + last N history turns + the query with numbered sources."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (history or [])[-MAX_HISTORY:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    user_block = query.strip()
    sources = _format_sources(citations)
    if sources:
        user_block = f"{user_block}\n\nSources:\n{sources}"
    messages.append({"role": "user", "content": user_block})
    return messages


def _format_sources(citations: list[dict[str, Any]]) -> str:
    lines = []
    for i, citation in enumerate(citations, start=1):
        loc = citation.get("section") or ""
        section = f" §{loc}" if loc else ""
        lines.append(
            f"[{i}] {citation.get('path', '?')}{section} "
            f"(L{citation.get('start_line')}-L{citation.get('end_line')}): "
            f"{citation.get('excerpt', '')}"
        )
    return "\n".join(lines)


def llm_chat(messages: list[dict[str, str]]) -> str:
    """Call the configured OpenAI-compatible endpoint and return the answer text."""
    config = llm_config()
    if not config["api_key"]:
        raise LLMNotConfigured(
            "LLM_API_KEY is not configured. Set it in the server environment "
            "(Render secret or local env)."
        )
    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=TIMEOUT_SECONDS,
        max_retries=2,
    )
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return content.strip()
