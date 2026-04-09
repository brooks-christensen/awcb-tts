from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SpeechRewriteConfig


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
PAREN_RE = re.compile(r"\([^)]*\)")
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text.replace("\u2014", ", ")
    text = text.replace("\u2013", ", ")
    text = text.replace(";", ",")
    text = PAREN_RE.sub("", text)
    text = text.replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    return parts


def _trim_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = sentence.replace("  ", " ")
    return sentence


def rewrite_for_speech(text: str, config: SpeechRewriteConfig) -> str:
    """
    Cheap local rewrite so we do not pay for an extra model call.
    Goal: shorter, smoother, more speakable audio input.
    """
    if not config.enabled:
        return text.strip()

    normalized = _normalize_text(text)
    if not normalized:
        return ""

    sentences = [_trim_sentence(s) for s in _split_sentences(normalized)]

    chosen: list[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        projected = current_len + len(sentence) + (1 if chosen else 0)
        if chosen and (len(chosen) >= config.max_sentences or projected > config.target_max_chars):
            break
        chosen.append(sentence)
        current_len = projected

    if not chosen:
        chosen = [normalized[: config.target_max_chars].rstrip()]

    out = " ".join(chosen).strip()

    # Try to avoid abrupt ending if we clipped mid-thought.
    if len(out) < len(normalized) and out[-1] not in ".!?":
        out = out.rstrip(",;: ") + "."

    return out
