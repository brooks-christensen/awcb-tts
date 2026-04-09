from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from openai import OpenAI

from .config import LocalConfig
from .indexing import INDEX_FILENAME, MANIFEST_FILENAME, METADATA_FILENAME


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: int
    source: str
    text: str


def load_metadata(path: Path) -> List[Dict[str, Any]]:
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


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_index_alignment(index: faiss.Index, metadata: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    if index.ntotal != len(metadata):
        raise ValueError(f"Index/metadata mismatch: {index.ntotal} vs {len(metadata)}")
    manifest_vectors = manifest.get("num_vectors")
    if manifest_vectors is not None and manifest_vectors != len(metadata):
        raise ValueError(f"Manifest/metadata mismatch: {manifest_vectors} vs {len(metadata)}")
    manifest_dim = manifest.get("vector_dim")
    if manifest_dim is not None and manifest_dim != index.d:
        raise ValueError(f"Manifest/index dimension mismatch: {manifest_dim} vs {index.d}")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Query embedding norm is too small to normalize.")
    return vector / norm


def embed_query(client: OpenAI, query: str, model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=query)
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D embedding vector, got shape {vector.shape}")
    return np.ascontiguousarray(l2_normalize(vector).reshape(1, -1), dtype=np.float32)


def retrieve_top_k(
    query_vector: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[RetrievedChunk]:
    scores, indices = index.search(query_vector, top_k)
    out: List[RetrievedChunk] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        row = metadata[int(idx)]
        out.append(
            RetrievedChunk(
                rank=rank,
                score=float(score),
                chunk_id=int(row["chunk_id"]),
                source=str(row["source"]),
                text=str(row["text"]),
            )
        )
    return out


def truncate_context_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_context_block(retrieved_chunks: List[RetrievedChunk], max_chars_per_chunk: int) -> str:
    parts: List[str] = []
    for chunk in retrieved_chunks:
        parts.append(
            "\n".join(
                [
                    f"[Source Rank {chunk.rank}]",
                    f"chunk_id: {chunk.chunk_id}",
                    f"source: {chunk.source}",
                    f"score: {chunk.score:.4f}",
                    "excerpt:",
                    truncate_context_text(chunk.text, max_chars_per_chunk),
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def build_user_prompt(query: str, retrieved_chunks: List[RetrievedChunk], max_chars_per_chunk: int) -> str:
    context_block = build_context_block(retrieved_chunks, max_chars_per_chunk=max_chars_per_chunk)
    return f"""Answer the user's question using the retrieved Alan Watts source excerpts below.

User question:
{query}

Retrieved source excerpts:
{context_block}

Instructions:
- Use the excerpts as the primary grounding for your answer.
- Synthesize naturally rather than quoting excessively.
- Do not mention retrieval, chunks, transcripts, or source excerpts explicitly.
- Avoid unsupported outside claims.
- Keep the answer conversational, reflective, and clear.
"""


def generate_answer(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if not getattr(response, "output_text", None):
        raise ValueError("Generation model returned no output_text")
    return response.output_text.strip()


def answer_query(client: OpenAI, config: LocalConfig, query: str, top_k: int | None = None) -> Dict[str, Any]:
    index_path = config.paths.index_dir / INDEX_FILENAME
    metadata_path = config.paths.index_dir / METADATA_FILENAME
    manifest_path = config.paths.index_dir / MANIFEST_FILENAME

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)
    manifest = load_manifest(manifest_path)
    validate_index_alignment(index, metadata, manifest)

    effective_top_k = top_k if top_k is not None else config.retrieve.top_k
    query_vector = embed_query(client, query, config.retrieve.embedding_model)
    retrieved = retrieve_top_k(query_vector, index, metadata, effective_top_k)
    user_prompt = build_user_prompt(query, retrieved, config.generate.max_context_chars_per_chunk)
    answer = generate_answer(
        client,
        model=config.generate.model,
        system_prompt=config.generate.system_prompt,
        user_prompt=user_prompt,
        temperature=config.generate.temperature,
        max_output_tokens=config.generate.max_output_tokens,
    )

    return {
        "query": query,
        "answer": answer,
        "generation_model": config.generate.model,
        "embedding_model": config.retrieve.embedding_model,
        "top_k": effective_top_k,
        "retrieved_context": [
            {
                "rank": c.rank,
                "score": c.score,
                "chunk_id": c.chunk_id,
                "source": c.source,
                "text": c.text,
            }
            for c in retrieved
        ],
    }
