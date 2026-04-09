from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    project_root: Path
    corpus_path: Path
    rag_chunks_path: Path
    index_dir: Path
    audio_dir: Path
    tts_cache_dir: Path
    runs_dir: Path
    logs_dir: Path


@dataclass
class RetrieveConfig:
    embedding_model: str
    top_k: int


@dataclass
class BuildIndexConfig:
    embedding_model: str
    embedding_batch_size: int
    embedding_sleep_seconds: float
    embedding_max_retries: int
    index_overwrite: bool


@dataclass
class GenerateConfig:
    model: str
    temperature: float
    max_output_tokens: int
    max_context_chars_per_chunk: int
    system_prompt: str


@dataclass
class SpeechRewriteConfig:
    enabled: bool
    target_max_chars: int
    max_sentences: int


@dataclass
class TTSConfig:
    backend: str
    model: str
    voice: str
    instructions: str
    response_format: str
    speed: float
    max_input_chars: int
    use_cache: bool
    cache_dir: Path


@dataclass
class AudioPostprocessConfig:
    enabled: bool
    vintage_sample_rate: int
    highpass_hz: float
    soft_clip_drive: float
    hiss_level: float
    normalize_peak: float


@dataclass
class RunArtifactsConfig:
    enabled: bool
    save_latest_audio: bool
    append_history_log: bool
    include_context_json: bool


@dataclass
class LocalConfig:
    preset_name: str
    paths: PathsConfig
    retrieve: RetrieveConfig
    build_index: BuildIndexConfig
    generate: GenerateConfig
    speech_rewrite: SpeechRewriteConfig
    tts: TTSConfig
    audio_postprocess: AudioPostprocessConfig
    run_artifacts: RunArtifactsConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_raw_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config_path, raw


def available_presets(config_path: str | Path) -> list[str]:
    _, raw = load_raw_config(config_path)
    presets = raw.get("presets", {}) or {}
    return sorted(str(name) for name in presets.keys())


def load_config(config_path: str | Path, preset_name: str | None = None) -> LocalConfig:
    config_path, raw = load_raw_config(config_path)
    project_root = config_path.parents[1]

    if preset_name:
        presets = raw.get("presets", {}) or {}
        if preset_name not in presets:
            available = ", ".join(sorted(presets.keys())) or "<none>"
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        raw = _deep_merge(raw, presets[preset_name])

    paths_raw = raw.get("paths", {})
    retrieve_raw = raw.get("retrieve", {})
    build_raw = raw.get("build_index", {})
    generate_raw = raw.get("generate", {})
    rewrite_raw = raw.get("speech_rewrite", {})
    tts_raw = raw.get("tts", {})
    post_raw = raw.get("audio_postprocess", {})
    artifacts_raw = raw.get("run_artifacts", {})

    paths = PathsConfig(
        project_root=project_root,
        corpus_path=project_root / paths_raw.get("corpus_path", "data/watts_rag_corpus_clean.txt"),
        rag_chunks_path=project_root / paths_raw.get("rag_chunks_path", "data/processed/rag_chunks.jsonl"),
        index_dir=project_root / paths_raw.get("index_dir", "data/indexes"),
        audio_dir=project_root / paths_raw.get("audio_dir", "outputs/audio"),
        tts_cache_dir=project_root / paths_raw.get("tts_cache_dir", "outputs/audio/cache"),
        runs_dir=project_root / paths_raw.get("runs_dir", "outputs/runs"),
        logs_dir=project_root / paths_raw.get("logs_dir", "outputs/logs"),
    )

    retrieve = RetrieveConfig(
        embedding_model=str(retrieve_raw.get("embedding_model", "text-embedding-3-small")),
        top_k=int(retrieve_raw.get("top_k", 3)),
    )

    build_index = BuildIndexConfig(
        embedding_model=str(build_raw.get("embedding_model", retrieve.embedding_model)),
        embedding_batch_size=int(build_raw.get("embedding_batch_size", 32)),
        embedding_sleep_seconds=float(build_raw.get("embedding_sleep_seconds", 0.1)),
        embedding_max_retries=int(build_raw.get("embedding_max_retries", 3)),
        index_overwrite=bool(build_raw.get("index_overwrite", False)),
    )

    generate = GenerateConfig(
        model=str(generate_raw.get("model", "gpt-5.4-mini")),
        temperature=float(generate_raw.get("temperature", 0.2)),
        max_output_tokens=int(generate_raw.get("max_output_tokens", 400)),
        max_context_chars_per_chunk=int(generate_raw.get("max_context_chars_per_chunk", 2500)),
        system_prompt=str(generate_raw.get("system_prompt", "Answer clearly and reflectively.")),
    )

    speech_rewrite = SpeechRewriteConfig(
        enabled=bool(rewrite_raw.get("enabled", True)),
        target_max_chars=int(rewrite_raw.get("target_max_chars", 900)),
        max_sentences=int(rewrite_raw.get("max_sentences", 6)),
    )

    tts = TTSConfig(
        backend=str(tts_raw.get("backend", "openai")).lower(),
        model=str(tts_raw.get("model", "gpt-4o-mini-tts")),
        voice=str(tts_raw.get("voice", "onyx")),
        instructions=str(
            tts_raw.get(
                "instructions",
                "Speak calmly, warmly, and reflectively, with measured pacing and clear enunciation. Avoid theatrical imitation. Keep the tone intimate and lecture-like.",
            )
        ),
        response_format=str(tts_raw.get("response_format", "wav")).lower(),
        speed=float(tts_raw.get("speed", 0.92)),
        max_input_chars=int(tts_raw.get("max_input_chars", 3500)),
        use_cache=bool(tts_raw.get("use_cache", True)),
        cache_dir=project_root / paths_raw.get("tts_cache_dir", "outputs/audio/cache"),
    )

    audio_postprocess = AudioPostprocessConfig(
        enabled=bool(post_raw.get("enabled", True)),
        vintage_sample_rate=int(post_raw.get("vintage_sample_rate", 16000)),
        highpass_hz=float(post_raw.get("highpass_hz", 120.0)),
        soft_clip_drive=float(post_raw.get("soft_clip_drive", 1.15)),
        hiss_level=float(post_raw.get("hiss_level", 0.0025)),
        normalize_peak=float(post_raw.get("normalize_peak", 0.92)),
    )

    run_artifacts = RunArtifactsConfig(
        enabled=bool(artifacts_raw.get("enabled", True)),
        save_latest_audio=bool(artifacts_raw.get("save_latest_audio", True)),
        append_history_log=bool(artifacts_raw.get("append_history_log", True)),
        include_context_json=bool(artifacts_raw.get("include_context_json", True)),
    )

    return LocalConfig(
        preset_name=preset_name or "default",
        paths=paths,
        retrieve=retrieve,
        build_index=build_index,
        generate=generate,
        speech_rewrite=speech_rewrite,
        tts=tts,
        audio_postprocess=audio_postprocess,
        run_artifacts=run_artifacts,
    )
