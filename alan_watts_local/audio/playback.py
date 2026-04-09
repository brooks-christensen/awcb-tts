from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from loguru import logger


def play_audio_file(path: Path) -> bool:
    system = platform.system().lower()
    candidates: list[list[str]] = []

    if system == "darwin":
        candidates.append(["afplay", str(path)])
    elif system == "linux":
        candidates.extend([
            ["aplay", str(path)],
            ["ffplay", "-nodisp", "-autoexit", str(path)],
            ["paplay", str(path)],
        ])
    elif system == "windows":
        candidates.append([
            "powershell",
            "-c",
            f'(New-Object Media.SoundPlayer "{str(path)}").PlaySync();',
        ])

    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning("Playback command failed: {} ({})", cmd, exc)

    logger.warning("No working local playback command found. Audio saved at {}", path)
    return False
