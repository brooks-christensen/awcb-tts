from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path

from loguru import logger
from openai import OpenAI

from ..config import TTSConfig


WAV_EXT = ".wav"


def _chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break

        split_at = remaining.rfind(". ", 0, max_chars)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars

        piece = remaining[:split_at].strip()
        if not piece:
            piece = remaining[:max_chars].strip()
            split_at = max_chars

        chunks.append(piece)
        remaining = remaining[split_at:].strip()

    return [c for c in chunks if c]


def _cache_key(text: str, config: TTSConfig) -> str:
    payload = {
        "text": text,
        "backend": config.backend,
        "model": config.model,
        "voice": config.voice,
        "instructions": config.instructions,
        "response_format": config.response_format,
        "speed": config.speed,
        "max_input_chars": config.max_input_chars,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stream_chunk_to_file(client: OpenAI, text: str, config: TTSConfig, out_path: Path) -> None:
    with client.audio.speech.with_streaming_response.create(
        model=config.model,
        voice=config.voice,
        input=text,
        instructions=config.instructions,
        response_format=config.response_format,
        speed=config.speed,
    ) as response:
        response.stream_to_file(out_path)


def _concat_wav_files(paths: list[Path], out_path: Path) -> None:
    if not paths:
        raise ValueError("No WAV files provided for concatenation.")

    params = None
    frames: list[bytes] = []
    for path in paths:
        with wave.open(str(path), "rb") as wf:
            current = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getcomptype(), wf.getcompname())
            if params is None:
                params = current
            elif params != current:
                raise ValueError(f"Cannot concatenate WAV files with mismatched params: {paths[0]} vs {path}")
            frames.append(wf.readframes(wf.getnframes()))

    nchannels, sampwidth, framerate, comptype, compname = params
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.setcomptype(comptype, compname)
        for blob in frames:
            wf.writeframes(blob)


class OpenAITTSBackend:
    def __init__(self, client: OpenAI, config: TTSConfig):
        if config.response_format.lower() != "wav":
            raise ValueError("This starter backend currently expects response_format='wav' for reliable local post-processing.")
        self.client = client
        self.config = config

    def synthesize(self, text: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path = self.config.cache_dir / f"{_cache_key(text, self.config)}{WAV_EXT}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.use_cache and cache_path.exists():
            logger.info("Using cached TTS audio at {}", cache_path)
            shutil.copy2(cache_path, out_path)
            return out_path

        chunks = _chunk_text(text, self.config.max_input_chars)
        logger.info("Synthesizing {} TTS chunk(s) with model={} voice={}", len(chunks), self.config.model, self.config.voice)

        if len(chunks) == 1:
            _stream_chunk_to_file(self.client, chunks[0], self.config, out_path)
        else:
            temp_paths: list[Path] = []
            for idx, chunk in enumerate(chunks, start=1):
                temp_path = out_path.parent / f"{out_path.stem}.part{idx:02d}{out_path.suffix}"
                _stream_chunk_to_file(self.client, chunk, self.config, temp_path)
                temp_paths.append(temp_path)
            _concat_wav_files(temp_paths, out_path)
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        if self.config.use_cache:
            shutil.copy2(out_path, cache_path)

        return out_path


def synthesize_speech(client: OpenAI, text: str, config: TTSConfig, out_path: Path) -> Path:
    backend = OpenAITTSBackend(client=client, config=config)
    return backend.synthesize(text=text, out_path=out_path)
