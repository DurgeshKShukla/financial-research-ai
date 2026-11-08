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
    selected_model = model or groq_model()
    is_reasoning_model = selected_model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer.")
    # GPT-OSS uses completion tokens for reasoning as well as its final answer.
    # A tiny caller override must not consume the entire budget before visible output.
    budget = max(max_tokens, 1500) if is_reasoning_model else max_tokens
    body = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": budget,
        "stream": False,
    }
    if is_reasoning_model:
        body["reasoning_effort"] = "low"
    data = _groq_request(body, key, timeout)
    answer, finish_reason = _answer(data)
    if answer:
        return answer
    # Retry once only when the provider explicitly reports budget exhaustion.
    # Do not retry rate limits or authentication/network errors automatically.
    if is_reasoning_model and finish_reason == "length" and budget < 3000:
        body["max_completion_tokens"] = min(budget * 2, 3000)
        data = _groq_request(body, key, timeout)
        answer, finish_reason = _answer(data)
        if answer:
            return answer
    if finish_reason == "length":
        raise RuntimeError("The model exhausted its response token budget before producing an answer. Try a shorter, specific question.")
    raise RuntimeError("The hosted model returned no answer text. Try again or ask a specific financial question.")


def _groq_request(body: dict, key: str, timeout: int) -> dict:
    payload = json.dumps(body).encode("utf-8")
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
    return data


def _answer(data: dict) -> tuple[str, str | None]:
    """Read only final answer content; never display the model's reasoning field."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if content is not None and not isinstance(content, str):
            raise TypeError("Unexpected content type")
        return (content or "").strip(), choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError("Hosted AI response format was not recognised.") from exc
