"""Conversation-aware PDF retrieval helpers (no Streamlit or model imports).

These rules improve diversity and avoid obvious repeat responses; they do not
certify factual correctness. The LLM must still cite retrieved PDF pages.
"""
from __future__ import annotations

import hashlib
import re

TOPIC_QUERIES = (
    "financial highlights revenue net income operating income fee revenue fiscal year compared previous year",
    "assets under management net inflows outflows institutional retail ETFs asset classes year end",
    "cash flows from operations capital expenditures dividends share repurchases debt liquidity",
    "acquisitions completed purchase consideration integration private markets infrastructure technology",
    "operating expenses compensation margins profitability cost performance",
    "investment management strategy client segments geography product distribution competitive position",
    "risk factors regulatory compliance cybersecurity market risks legal matters",
)

# Conversational requests which should NOT be interpreted as a new standalone topic.
_NEW_RE = re.compile(
    r"\b(?:something more|some more|anything more|anything else|what else|"
    r"other|another|additional|different|new findings?|more findings?|"
    r"more (?:important )?(?:points|details|insights|information)|apart from|besides|"
    r"except (?:this|that|these)|not (?:the )?same|don't repeat|do not repeat)\b",
    re.I,
)
_BROAD_RE = re.compile(r"\b(?:findings?|highlights?|important|summary|key points?|takeaways?|overview)\b", re.I)
_WORD_RE = re.compile(r"[a-z0-9]+", re.I)
_STOP = {"the", "and", "for", "with", "from", "that", "this", "were", "have", "into", "their", "about", "which", "these", "other", "more", "blackrock", "report", "company", "annual"}


def wants_new_findings(question: str) -> bool:
    return bool(_NEW_RE.search(question or ""))


def broad_findings_question(question: str) -> bool:
    return bool(_BROAD_RE.search(question or "")) or wants_new_findings(question)


def passage_key(p: dict) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", p.get("text", "")).strip().lower().encode()).hexdigest()


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 3 and w not in _STOP}


def lexical_overlap(a: str, b: str) -> float:
    """Conservative overlap relative to shorter passage, for near-duplicate detection."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def prior_sources(history: list[dict]) -> list[dict]:
    return [src for msg in history if msg.get("role") == "assistant" for src in msg.get("sources", [])]


def topic_queries(question: str) -> list[str]:
    """A summary needs coverage of multiple sections, not one vague vector query."""
    if broad_findings_question(question):
        return list(TOPIC_QUERIES)
    return [question]


def select_diverse_passages(candidates: list[dict], history: list[dict], k: int = 5, novelty: bool = False) -> list[dict]:
    """Select relevant but non-duplicate evidence with diversity across pages.

    Candidates are expected to have a 'score' from semantic search; scores are
    not comparable across arbitrary prompts, so each query's rank is also used.
    """
    previous = prior_sources(history) if novelty else []
    old_ids = {passage_key(p) for p in previous}
    pool = []
    seen_ids = set()
    for candidate in candidates:
        key = passage_key(candidate)
        if key in seen_ids or (novelty and key in old_ids):
            continue
        seen_ids.add(key)
        if novelty and any(lexical_overlap(candidate["text"], old["text"]) > 0.77 for old in previous):
            continue
        pool.append(candidate)

    selected = []
    while len(selected) < k and pool:
        def value(p):
            text = p.get("text", "")
            # Modest boost for concrete information; discourage repetitive boilerplate.
            has_figures = bool(re.search(r"(?:\$\s?\d|\d{1,3}(?:,\d{3})+|\b\d+(?:\.\d+)?%)", text))
            numeric_bonus = 0.06 if has_figures else 0
            page_penalty = 0.16 if p.get("page") in {x.get("page") for x in selected} else 0
            topic_penalty = 0.19 if p.get("topic_id") is not None and p.get("topic_id") in {x.get("topic_id") for x in selected} else 0
            similarity_penalty = max((lexical_overlap(text, x["text"]) for x in selected), default=0) * 0.18
            return p.get("score", 0) + numeric_bonus - page_penalty - topic_penalty - similarity_penalty
        next_item = max(pool, key=value)
        pool.remove(next_item)
        if any(lexical_overlap(next_item["text"], x["text"]) > 0.87 for x in selected):
            continue
        selected.append(next_item)
    return selected


def prior_answer_summary(history: list[dict], limit=1000) -> str:
    """Give the LLM a negative-content memory, but not entire older chat turns."""
    assistant_text = [m.get("content", "") for m in history if m.get("role") == "assistant"]
    return "\n---\n".join(x[:limit] for x in assistant_text[-2:])


def repeated_answer(answer: str, history: list[dict]) -> bool:
    """Catch grossly repeated answers (not a semantic proof of novelty)."""
    previous = [m.get("content", "") for m in history if m.get("role") == "assistant"]
    if not previous:
        return False
    current = _words(answer)
    if len(current) < 10:
        return False
    for old in previous[-3:]:
        old_words = _words(old)
        if len(old_words) >= 10 and len(current & old_words) / min(len(current), len(old_words)) > 0.78:
            return True
    return False
