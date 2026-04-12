from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from .config import available_presets, load_config
from .run_artifacts import save_run_bundle


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "local_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Alan Watts RAG runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config file")
    parser.add_argument("--preset", default=None, help="Optional preset name from config/local_config.yaml")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create chunks and FAISS index if missing")
    prepare_parser.add_argument("--force-rebuild-index", action="store_true")

    subparsers.add_parser("list-presets", help="List available presets")

    ask_parser = subparsers.add_parser("ask", help="Run retrieval + generation")
    ask_parser.add_argument("--query", required=True)
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.add_argument("--json", action="store_true", help="Emit full JSON instead of plain answer")
    ask_parser.add_argument("--no-save-run", action="store_true", help="Do not save a reproducible run bundle")

    speak_parser = subparsers.add_parser("ask-and-speak", help="Run retrieval + generation + TTS")
    speak_parser.add_argument("--query", required=True)
    speak_parser.add_argument("--top-k", type=int, default=None)
    speak_parser.add_argument("--play", action="store_true", help="Attempt local playback after synthesis")
    speak_parser.add_argument("--json", action="store_true", help="Emit full JSON instead of plain answer")
    speak_parser.add_argument("--no-save-run", action="store_true", help="Do not save a reproducible run bundle")
    speak_parser.add_argument("--raw", action="store_true", help="Skip vintage post-processing")
    speak_parser.add_argument("--vintage", action="store_true", help="Force vintage post-processing")

    return parser.parse_args()


def build_client() -> Any:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def prepare(config_path: str | Path, force_rebuild_index: bool, preset_name: str | None = None) -> None:
    from .indexing import ensure_index
    from .ingest import ingest_corpus

    config = load_config(config_path, preset_name=preset_name)
    client = build_client()

    if not config.paths.corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {config.paths.corpus_path}\n"
            "Copy data/watts_rag_corpus_clean.txt from the existing repo into this package's data/ directory."
        )

    if not config.paths.rag_chunks_path.exists():
        ingest_corpus(config.paths.corpus_path, config.paths.rag_chunks_path)
    else:
        logger.info("Chunks already exist at {}", config.paths.rag_chunks_path)

    ensure_index(
        client=client,
        rag_chunks_path=config.paths.rag_chunks_path,
        index_dir=config.paths.index_dir,
        embedding_model=config.build_index.embedding_model,
        batch_size=config.build_index.embedding_batch_size,
        sleep_seconds=config.build_index.embedding_sleep_seconds,
        max_retries=config.build_index.embedding_max_retries,
        overwrite=force_rebuild_index or config.build_index.index_overwrite,
    )


def list_presets(config_path: str | Path) -> None:
    names = available_presets(config_path)
    if not names:
        print("No presets defined.")
        return
    for name in names:
        print(name)


def ask(
    config_path: str | Path,
    query: str,
    top_k: int | None,
    emit_json: bool,
    save_run: bool,
    preset_name: str | None = None,
) -> None:
    from .pipeline import answer_query

    config = load_config(config_path, preset_name=preset_name)
    client = build_client()

    if not config.paths.rag_chunks_path.exists() or not (config.paths.index_dir / "rag_faiss.index").exists():
        logger.info("Required local artifacts are missing. Preparing them now.")
        prepare(config_path=config_path, force_rebuild_index=False, preset_name=preset_name)

    result = answer_query(client=client, config=config, query=query, top_k=top_k)

    run_dir = None
    if save_run and config.run_artifacts.enabled:
        run_dir, history_record = save_run_bundle(config=config, query=query, result_payload=result)
        result["run_dir"] = str(run_dir)
        result["history_log_path"] = history_record.get("history_log_path")

    if emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        if run_dir is not None:
            print()
            print(f"Run bundle saved to: {run_dir}")


def ask_and_speak(
    config_path: str | Path,
    query: str,
    top_k: int | None,
    play: bool,
    emit_json: bool,
    save_run: bool,
    raw: bool = False,
    vintage: bool = False,
    preset_name: str | None = None,
) -> None:
    from .audio.playback import play_audio_file
    from .audio.postprocess import apply_vintage_postprocess
    from .pipeline import answer_query
    from .speech_rewrite import rewrite_for_speech
    from .tts import synthesize_speech

    config = load_config(config_path, preset_name=preset_name)
    client = build_client()

    if not config.paths.rag_chunks_path.exists() or not (config.paths.index_dir / "rag_faiss.index").exists():
        logger.info("Required local artifacts are missing. Preparing them now.")
        prepare(config_path=config_path, force_rebuild_index=False, preset_name=preset_name)

    result = answer_query(client=client, config=config, query=query, top_k=top_k)
    answer_text = result["answer"]
    speech_text = rewrite_for_speech(answer_text, config.speech_rewrite)

    raw_audio_path = config.paths.audio_dir / "latest_raw.wav"
    vintage_audio_path = config.paths.audio_dir / "latest_vintage.wav"
    raw_audio_path.parent.mkdir(parents=True, exist_ok=True)

    use_vintage = config.audio_postprocess.enabled
    if raw:
        use_vintage = False
    elif vintage:
        use_vintage = True

    synthesize_speech(client=client, text=speech_text, config=config.tts, out_path=raw_audio_path)

    if use_vintage:
        apply_vintage_postprocess(
            in_path=raw_audio_path,
            out_path=vintage_audio_path,
            config=config.audio_postprocess,
        )
        final_audio_path = vintage_audio_path
    else:
        final_audio_path = raw_audio_path

    payload = {
        **result,
        "preset_name": config.preset_name,
        "speech_text": speech_text,
        "tts_model": getattr(config.tts, "model", getattr(config.tts, "model_path", "xtts_local")),
        "tts_voice": getattr(config.tts, "voice", "xtts_local"),
        "audio_raw_path": str(raw_audio_path),
        "audio_final_path": str(final_audio_path),
    }

    run_dir = None
    if save_run and config.run_artifacts.enabled:
        run_dir, history_record = save_run_bundle(
            config=config,
            query=query,
            result_payload=payload,
            raw_audio_path=raw_audio_path,
            final_audio_path=final_audio_path,
        )
        payload["run_dir"] = str(run_dir)
        payload["history_log_path"] = history_record.get("history_log_path")

    if emit_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["answer"])
        print()
        print(f"Audio saved to: {final_audio_path}")
        if run_dir is not None:
            print(f"Run bundle saved to: {run_dir}")

    if play:
        play_audio_file(final_audio_path)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(config_path=args.config, force_rebuild_index=args.force_rebuild_index, preset_name=args.preset)
    elif args.command == "list-presets":
        list_presets(config_path=args.config)
    elif args.command == "ask":
        ask(
            config_path=args.config,
            query=args.query,
            top_k=args.top_k,
            emit_json=args.json,
            save_run=not args.no_save_run,
            preset_name=args.preset,
        )
    elif args.command == "ask-and-speak":
        ask_and_speak(
            config_path=args.config,
            query=args.query,
            top_k=args.top_k,
            play=args.play,
            emit_json=args.json,
            save_run=not args.no_save_run,
            raw=args.raw,
            vintage=args.vintage,
            preset_name=args.preset,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
