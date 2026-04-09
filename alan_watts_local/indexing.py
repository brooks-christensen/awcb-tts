from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


INDEX_FILENAME = "rag_faiss.index"
METADATA_FILENAME = "rag_metadata.jsonl"
MANIFEST_FILENAME = "rag_index_manifest.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    return rows


def validate_chunk_record(record: Dict[str, Any]) -> None:
    required = {"chunk_id", "text", "source"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Chunk record missing fields: {sorted(missing)}")
    if not isinstance(record["chunk_id"], int):
        raise ValueError("chunk_id must be int")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError("source must be a non-empty string")


def split_text_for_embedding(text: str, max_chars: int = 24000, overlap_chars: int = 1000) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    pieces: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= text_len:
            break
        start = max(end - overlap_chars, 0)

    return pieces


def embed_single_text(client: OpenAI, text: str, model: str, max_retries: int) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(model=model, input=text)
            vector = np.asarray(response.data[0].embedding, dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError(f"Expected 1D embedding vector, got shape {vector.shape}")
            return vector
        except Exception as exc:
            last_error = exc
            logger.warning("Embedding failed on attempt {}/{}: {}", attempt, max_retries, exc)
            time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError(f"Embedding failed after {max_retries} attempts: {last_error}")


def embed_record_text(client: OpenAI, text: str, model: str, max_retries: int) -> np.ndarray:
    pieces = split_text_for_embedding(text)
    if len(pieces) == 1:
        return embed_single_text(client, pieces[0], model, max_retries)

    logger.info("Large chunk detected; embedding as {} sub-pieces and pooling.", len(pieces))
    piece_vectors = [embed_single_text(client, piece, model, max_retries) for piece in pieces]
    return np.vstack(piece_vectors).astype(np.float32).mean(axis=0)


def chunked(seq: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [list(seq[i : i + batch_size]) for i in range(0, len(seq), batch_size)]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def write_metadata_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_index_manifest(
    path: Path,
    *,
    input_path: Path,
    index_path: Path,
    metadata_path: Path,
    embedding_model: str,
    vector_dim: int,
    num_vectors: int,
    batch_size: int,
) -> None:
    manifest = {
        "input_path": str(input_path),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embedding_model": embedding_model,
        "vector_dim": vector_dim,
        "num_vectors": num_vectors,
        "normalized": True,
        "batch_size": batch_size,
        "faiss_index_type": "IndexFlatIP",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_index(
    *,
    client: OpenAI,
    rag_chunks_path: Path,
    index_dir: Path,
    embedding_model: str,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    overwrite: bool,
) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / INDEX_FILENAME
    metadata_path = index_dir / METADATA_FILENAME
    manifest_path = index_dir / MANIFEST_FILENAME

    if not overwrite and index_path.exists() and metadata_path.exists() and manifest_path.exists():
        logger.info("Existing index found at {}. Skipping rebuild.", index_dir)
        return index_dir

    records = load_jsonl(rag_chunks_path)
    for record in records:
        validate_chunk_record(record)

    logger.info("Building embeddings for {} chunks using model={}", len(records), embedding_model)

    vectors: List[np.ndarray] = []
    cleaned_records: List[Dict[str, Any]] = []

    for batch in tqdm(chunked(records, batch_size), desc="Embedding batches"):
        for record in batch:
            vector = embed_record_text(
                client=client,
                text=record["text"],
                model=embedding_model,
                max_retries=max_retries,
            )
            vectors.append(vector)
            cleaned_records.append(
                {
                    "chunk_id": record["chunk_id"],
                    "source": record["source"],
                    "text": record["text"],
                }
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    matrix = np.vstack(vectors).astype(np.float32)
    matrix = l2_normalize(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_path))
    write_metadata_jsonl(metadata_path, cleaned_records)
    write_index_manifest(
        manifest_path,
        input_path=rag_chunks_path,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
        vector_dim=matrix.shape[1],
        num_vectors=matrix.shape[0],
        batch_size=batch_size,
    )

    logger.info("Wrote index artifacts to {}", index_dir)
    return index_dir
