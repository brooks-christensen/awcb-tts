from .xtts_tts import XTTSTTS


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_tts_engine(tts_cfg):
    backend = _get(tts_cfg, "backend")

    if backend == "openai":
        from .openai_tts import OpenAITTS
        return OpenAITTS(tts_cfg)

    if backend == "xtts_local":
        return XTTSTTS(tts_cfg)

    raise ValueError(f"Unsupported TTS backend: {backend}")


def synthesize_speech(*, client, text, config, out_path):
    backend = _get(config, "backend")

    if backend == "openai":
        from .openai_tts import synthesize_speech as synthesize_openai_speech
        return synthesize_openai_speech(
            client=client,
            text=text,
            config=config,
            out_path=out_path,
        )

    if backend == "xtts_local":
        engine = XTTSTTS(config)
        return engine.synthesize(text=text, out_path=out_path)

    raise ValueError(f"Unsupported TTS backend: {backend}")


__all__ = ["build_tts_engine", "synthesize_speech", "XTTSTTS"]