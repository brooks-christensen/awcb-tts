from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from loguru import logger

from ..config import AudioPostprocessConfig


INT16_MAX = 32767.0


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported in this starter postprocess chain. Got sample width={sampwidth}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / INT16_MAX
    if nchannels > 1:
        audio = audio.reshape(-1, nchannels).mean(axis=1)
    return audio, framerate


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * INT16_MAX).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _resample_linear(audio: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    if old_sr == new_sr or len(audio) == 0:
        return audio
    duration = len(audio) / old_sr
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_len = max(1, int(round(duration * new_sr)))
    new_times = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def _one_pole_highpass(audio: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sr
    alpha = rc / (rc + dt)
    out = np.empty_like(audio)
    out[0] = audio[0]
    for i in range(1, len(audio)):
        out[i] = alpha * (out[i - 1] + audio[i] - audio[i - 1])
    return out


def _soft_clip(audio: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 0:
        return audio
    return np.tanh(drive * audio) / np.tanh(drive)


def _add_hiss(audio: np.ndarray, level: float, seed: int = 7) -> np.ndarray:
    if level <= 0:
        return audio
    rng = np.random.default_rng(seed)
    hiss = rng.normal(0.0, level, size=audio.shape).astype(np.float32)
    return audio + hiss


def _normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak < 1e-8:
        return audio
    gain = min(1.0, target_peak / peak)
    return audio * gain


def apply_vintage_postprocess(in_path: Path, out_path: Path, config: AudioPostprocessConfig) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.enabled:
        if in_path != out_path:
            out_path.write_bytes(in_path.read_bytes())
        return out_path

    audio, sr = _read_wav(in_path)
    logger.info("Post-processing WAV: sr={} samples={}", sr, len(audio))

    # Downsample for a more vintage, bandwidth-limited sound.
    reduced = _resample_linear(audio, sr, config.vintage_sample_rate)
    processed = _one_pole_highpass(reduced, config.highpass_hz, config.vintage_sample_rate)
    processed = _soft_clip(processed, config.soft_clip_drive)
    processed = _add_hiss(processed, config.hiss_level)
    processed = _normalize_peak(processed, config.normalize_peak)

    _write_wav(out_path, processed, config.vintage_sample_rate)
    return out_path
