from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from loguru import logger


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def chunk_paragraphs(
    paragraphs: List[str],
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
            continue

        if len(current) + 2 + len(para) <= target_chars:
            current += "\n\n" + para
        else:
            chunks.append(current)
            if overlap_chars > 0:
                overlap_text = current[-overlap_chars:]
                split_idx = overlap_text.find(" ")
                if split_idx != -1:
                    overlap_text = overlap_text[split_idx + 1 :]
                current = overlap_text + "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def write_jsonl(chunks: List[str], path: Path, source_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            record = {"chunk_id": idx, "text": chunk, "source": source_name}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ingest_corpus(input_path: Path, output_path: Path) -> Path:
    raw_text = load_text(input_path)
    clean_text = normalize_text(raw_text)
    paragraphs = split_paragraphs(clean_text)
    chunks = chunk_paragraphs(paragraphs)
    write_jsonl(chunks, output_path, source_name=input_path.name)

    logger.info("Ingested corpus: input={}, paragraphs={}, chunks={}, output={}", input_path, len(paragraphs), len(chunks), output_path)
    return output_path
