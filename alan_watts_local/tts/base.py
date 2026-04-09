from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TTSBackend(Protocol):
    def synthesize(self, text: str, out_path: Path) -> Path:
        ...
