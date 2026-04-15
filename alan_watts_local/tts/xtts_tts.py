from pathlib import Path
from TTS.api import TTS


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class XTTSTTS:
    def __init__(self, cfg):
        self.cfg = cfg

        model_dir = _get(cfg, "model_dir")
        model_path = _get(cfg, "model_path")
        config_path = _get(cfg, "config_path")
        device = _get(cfg, "device", "cuda")

        # Recover model_dir if schema dropped it but model_path/config_path survived
        if model_dir is None and model_path:
            model_dir = str(Path(model_path).expanduser().parent)
        if model_dir is None and config_path:
            model_dir = str(Path(config_path).expanduser().parent)

        # Last-resort explicit fallback so the pipeline keeps moving
        if model_dir is None:
            model_dir = "/home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot"

        if config_path is None:
            config_path = str(Path(model_dir) / "config.json")

        self.tts = TTS(
            model_path=str(Path(model_dir).expanduser()),
            config_path=str(Path(config_path).expanduser()),
        ).to(device)

    def synthesize(self, text: str, out_path: str) -> str:
        out_path = Path(out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        speaker_wavs = _get(self.cfg, "speaker_wavs")
        language = _get(self.cfg, "language", "en")

        if not speaker_wavs:
            speaker_wavs = [
                "/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_a1.wav",
                "/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_b1.wav",
                "/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_d1.wav",
            ]

        self.tts.tts_to_file(
            text=text,
            file_path=str(out_path),
            speaker_wav=speaker_wavs,
            language=language,
        )
        return str(out_path)
