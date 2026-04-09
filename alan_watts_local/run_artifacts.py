from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import LocalConfig


SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 48) -> str:
    text = text.lower().strip()
    text = SLUG_RE.sub("-", text)
    text = text.strip("-")
    if not text:
        return "query"
    return text[:max_len].rstrip("-")


def make_run_dir(config: LocalConfig, query: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(query)
    run_dir = config.paths.runs_dir / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_history_log(config: LocalConfig, record: dict[str, Any]) -> Path:
    history_path = config.paths.logs_dir / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return history_path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_config_snapshot(config: LocalConfig) -> dict[str, Any]:
    return {
        "preset_name": config.preset_name,
        "retrieve": {
            "embedding_model": config.retrieve.embedding_model,
            "top_k": config.retrieve.top_k,
        },
        "generate": {
            "model": config.generate.model,
            "temperature": config.generate.temperature,
            "max_output_tokens": config.generate.max_output_tokens,
            "max_context_chars_per_chunk": config.generate.max_context_chars_per_chunk,
        },
        "speech_rewrite": {
            "enabled": config.speech_rewrite.enabled,
            "target_max_chars": config.speech_rewrite.target_max_chars,
            "max_sentences": config.speech_rewrite.max_sentences,
        },
        "tts": {
            "backend": config.tts.backend,
            "model": config.tts.model,
            "voice": config.tts.voice,
            "instructions": config.tts.instructions,
            "response_format": config.tts.response_format,
            "speed": config.tts.speed,
            "max_input_chars": config.tts.max_input_chars,
            "use_cache": config.tts.use_cache,
        },
        "audio_postprocess": {
            "enabled": config.audio_postprocess.enabled,
            "vintage_sample_rate": config.audio_postprocess.vintage_sample_rate,
            "highpass_hz": config.audio_postprocess.highpass_hz,
            "soft_clip_drive": config.audio_postprocess.soft_clip_drive,
            "hiss_level": config.audio_postprocess.hiss_level,
            "normalize_peak": config.audio_postprocess.normalize_peak,
        },
    }


def save_run_bundle(
    *,
    config: LocalConfig,
    query: str,
    result_payload: dict[str, Any],
    raw_audio_path: Path | None = None,
    final_audio_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_dir = make_run_dir(config=config, query=query)

    answer_text = str(result_payload.get("answer", ""))
    speech_text = str(result_payload.get("speech_text", ""))

    _write_text(run_dir / "query.txt", query)
    _write_text(run_dir / "answer.txt", answer_text)
    if speech_text:
        _write_text(run_dir / "speech_text.txt", speech_text)

    serializable_payload = dict(result_payload)
    if not config.run_artifacts.include_context_json:
        serializable_payload.pop("retrieved_context", None)
    _write_json(run_dir / "response.json", serializable_payload)
    _write_yaml(run_dir / "config_snapshot.yaml", make_config_snapshot(config))

    copied_raw = None
    copied_final = None
    if raw_audio_path and raw_audio_path.exists():
        copied_raw = run_dir / "raw.wav"
        shutil.copy2(raw_audio_path, copied_raw)
    if final_audio_path and final_audio_path.exists():
        copied_final = run_dir / "vintage.wav"
        shutil.copy2(final_audio_path, copied_final)

    history_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "preset_name": config.preset_name,
        "query": query,
        "answer_chars": len(answer_text),
        "speech_chars": len(speech_text),
        "generation_model": result_payload.get("generation_model"),
        "embedding_model": result_payload.get("embedding_model"),
        "tts_model": result_payload.get("tts_model"),
        "tts_voice": result_payload.get("tts_voice"),
        "run_dir": str(run_dir),
        "raw_audio_path": str(copied_raw) if copied_raw else None,
        "final_audio_path": str(copied_final) if copied_final else None,
    }

    history_path = None
    if config.run_artifacts.append_history_log:
        history_path = append_history_log(config=config, record=history_record)
        history_record["history_log_path"] = str(history_path)

    return run_dir, history_record
