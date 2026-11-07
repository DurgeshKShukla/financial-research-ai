"""Hosted Groq chat inference, with secrets handled server-side only.

The app retains its existing Ollama path when no GROQ_API_KEY is configured.
Do not log or show the secret. Groq receives the selected chat context.
"""
from __future__ import annotations
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _secret(name: str) -> str:
    env = os.environ.get(name, "").strip()
    if env:
        return env
    try:
        import streamlit as st
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def get_groq_key() -> str:
    return _secret("GROQ_API_KEY")


def groq_key_configured() -> bool:
    return bool(get_groq_key())


def groq_model() -> str:
    return _secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL


def groq_chat(messages: list[dict], model: str | None = None,
              *, max_tokens: int = 1500, temperature: float = 0.05,
              timeout: int = 90) -> str:
    key = get_groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured in Streamlit Cloud secrets.")
    payload = json.dumps({
        "model": model or groq_model(),
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    request = Request(GROQ_URL, data=payload, headers={
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "FinancialResearchAI/1.0",
}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except HTTPError as exc:
        # Do not echo provider response bodies, which might disclose sensitive details.
        if exc.code == 401:
            raise RuntimeError("Groq rejected the API key. Check cloud secrets.") from exc
        if exc.code == 429:
            raise RuntimeError("Groq rate limit reached. Try later or review the provider limits.") from exc
        raise RuntimeError(f"Hosted AI request failed (HTTP {exc.code}). Verify the model and account.") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("Could not connect to the hosted AI provider.") from exc
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Hosted AI response format was not recognised.") from exc
