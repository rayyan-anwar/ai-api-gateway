"""Dependency-free local text analysis used by the built-in `local` provider.

These are classic algorithms (frequency-based extractive summarisation, lexicon sentiment with negation,
frequency keywords). They are intentionally simple and are NOT comparable to large language models.

Author: Rayyan Anwar
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

STOP = frozenset("""a an the and or but if then of to in on for with at by from is are was were be been it this that as
not no we you they he she i our your their its will would can could should may might have has had do does did so than
too very just also into about over after before there here what which who when where how all any each more most some
such only own same""".split())
POSITIVE = frozenset("""good great excellent amazing awesome love loved loves like liked fantastic wonderful happy pleased
delighted perfect best brilliant impressive reliable fast smooth easy helpful friendly enjoyable superb nice beautiful
recommend recommended useful satisfied outstanding solid""".split())
NEGATIVE = frozenset("""bad terrible awful horrible hate hated hates poor worst disappointing disappointed slow broken
buggy useless annoying frustrating difficult painful unreliable ugly boring waste mediocre sad angry fail failed failure
crash crashed crashes expensive rude confusing unhappy defective""".split())
NEGATORS = frozenset({"not", "never", "no", "cannot", "without", "hardly"})
INTENSIFIERS = frozenset({"very", "extremely", "really", "incredibly", "absolutely"})
_WORD = re.compile(r"[a-z']+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def _content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP and len(t) > 2]


def keywords(text: str, k: int = 8) -> list[tuple[str, int]]:
    words = _content_words(tokenize(text))
    counts = Counter(words)
    first = {}
    for i, w in enumerate(words):
        first.setdefault(w, i)
    return sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]]))[:k]


def summarize(text: str, n_sentences: int = 3) -> str:
    sents = split_sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    freq = Counter(_content_words(tokenize(text)))
    scored = []
    for i, s in enumerate(sents):
        words = _content_words(tokenize(s))
        score = sum(freq[w] for w in words) / math.sqrt(len(words) + 1)
        scored.append((score, i))
    top = sorted(sorted(scored, key=lambda t: (-t[0], t[1]))[:n_sentences], key=lambda t: t[1])
    return " ".join(sents[i] for _, i in top)


def sentiment(text: str) -> dict:
    toks = tokenize(text)
    pos = neg = 0.0
    for i, t in enumerate(toks):
        val = 1.0 if t in POSITIVE else -1.0 if t in NEGATIVE else 0.0
        if not val:
            continue
        if i > 0 and toks[i - 1] in INTENSIFIERS:
            val *= 1.5
        if any(p in NEGATORS or p.endswith("n't") for p in toks[max(0, i - 2):i]):
            val = -val
        if val > 0:
            pos += val
        else:
            neg -= val
    total = pos + neg
    score = (pos - neg) / total if total else 0.0
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return {"label": label, "score": round(score, 3), "positive_weight": pos, "negative_weight": neg}


def run_task(upstream_model: str, text: str, max_tokens: int) -> str:
    if upstream_model == "local/summarizer":
        return summarize(text, max(1, min(5, max_tokens // 64)))
    if upstream_model == "local/sentiment":
        return json.dumps(sentiment(text))
    if upstream_model == "local/keywords":
        return ", ".join(f"{w} ({c})" for w, c in keywords(text))
    raise ValueError(f"unknown local model '{upstream_model}'")
