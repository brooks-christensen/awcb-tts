import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
LABELS_PATH = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/labels_1.txt")
SOURCE_DIR = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/section_a")
OUTPUT_DIR = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/section_a_clips_trimmed")
MANIFEST_PATH = OUTPUT_DIR / "section_a_trimmed_manifest.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Parameters to tune
# -----------------------------
SEARCH_PAD_SEC = 0.25      # how far around each label to search for better boundaries
KEEP_HEAD_SEC = 0.05       # how much silence to keep before detected speech onset
KEEP_TAIL_SEC = 0.10       # how much silence to keep after detected speech offset

FRAME_MS = 20              # RMS frame length
HOP_MS = 10                # RMS hop length

THRESH_MULTIPLIER = 3.0    # threshold above local noise floor
MIN_THRESHOLD = 1e-4       # absolute minimum RMS threshold
MAX_THRESHOLD_FRACTION = 0.20  # do not let threshold exceed 20% of local max RMS


# -----------------------------
# Helpers
# -----------------------------
def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def frame_rms(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        padded = np.pad(x, (0, max(0, frame_len - len(x))))
        return np.array([np.sqrt(np.mean(padded ** 2) + 1e-12)])

    rms_vals = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start:start + frame_len]
        rms_vals.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
    return np.array(rms_vals)


def auto_trim_region(
    audio: np.ndarray,
    sr: int,
    coarse_start_sec: float,
    coarse_end_sec: float,
    search_pad_sec: float = SEARCH_PAD_SEC,
    keep_head_sec: float = KEEP_HEAD_SEC,
    keep_tail_sec: float = KEEP_TAIL_SEC,
):
    n_samples = len(audio)

    search_start_sec = max(0.0, coarse_start_sec - search_pad_sec)
    search_end_sec = min(n_samples / sr, coarse_end_sec + search_pad_sec)

    s0 = int(round(search_start_sec * sr))
    s1 = int(round(search_end_sec * sr))

    region = audio[s0:s1]
    if len(region) == 0:
        return None

    frame_len = max(1, int(round(FRAME_MS / 1000 * sr)))
    hop_len = max(1, int(round(HOP_MS / 1000 * sr)))
    rms = frame_rms(region, frame_len, hop_len)

    if len(rms) == 0:
        return None

    # Estimate local noise floor from quiet percentiles
    noise_floor = np.percentile(rms, 20)
    peak_rms = np.max(rms)

    threshold = max(MIN_THRESHOLD, noise_floor * THRESH_MULTIPLIER)
    threshold = min(threshold, peak_rms * MAX_THRESHOLD_FRACTION)

    active = np.where(rms >= threshold)[0]

    # Fallback: if nothing crosses threshold, use the original coarse bounds
    if len(active) == 0:
        trimmed_start_sec = max(0.0, coarse_start_sec - keep_head_sec)
        trimmed_end_sec = min(n_samples / sr, coarse_end_sec + keep_tail_sec)
        return {
            "trimmed_start_sec": trimmed_start_sec,
            "trimmed_end_sec": trimmed_end_sec,
            "threshold": threshold,
            "noise_floor": noise_floor,
            "peak_rms": peak_rms,
            "used_fallback": True,
        }

    first_frame = int(active[0])
    last_frame = int(active[-1])

    detected_start_sample = s0 + first_frame * hop_len
    detected_end_sample = min(s1, s0 + last_frame * hop_len + frame_len)

    trimmed_start_sample = max(0, int(round(detected_start_sample - keep_head_sec * sr)))
    trimmed_end_sample = min(n_samples, int(round(detected_end_sample + keep_tail_sec * sr)))

    return {
        "trimmed_start_sec": trimmed_start_sample / sr,
        "trimmed_end_sec": trimmed_end_sample / sr,
        "threshold": threshold,
        "noise_floor": noise_floor,
        "peak_rms": peak_rms,
        "used_fallback": False,
    }


# -----------------------------
# Load labels
# -----------------------------
labels = pd.read_csv(
    LABELS_PATH,
    sep="\t",
    header=None,
    names=["start", "end", "label"],
)

labels["start"] = pd.to_numeric(labels["start"], errors="coerce")
labels["end"] = pd.to_numeric(labels["end"], errors="coerce")
labels["label_num"] = labels["label"].str.extract(r"(\d+)$").astype(int)
labels = labels.sort_values("label_num").reset_index(drop=True)

assert len(labels) == 180, f"Expected 180 labels, found {len(labels)}"

# 10 labels per source file
labels["source_idx"] = ((labels["label_num"] - 1) // 10) + 1

manifest_rows = []

for source_idx in range(1, 19):
    source_path = SOURCE_DIR / f"src{source_idx}.wav"
    assert source_path.exists(), f"Missing source file: {source_path}"

    audio, sr = sf.read(source_path)
    audio = to_mono(np.asarray(audio, dtype=np.float32))

    group = labels[labels["source_idx"] == source_idx].copy()
    assert len(group) == 10, f"Expected 10 labels for source {source_idx}, found {len(group)}"

    for _, row in group.iterrows():
        result = auto_trim_region(
            audio=audio,
            sr=sr,
            coarse_start_sec=float(row["start"]),
            coarse_end_sec=float(row["end"]),
        )

        if result is None:
            continue

        start_sec = result["trimmed_start_sec"]
        end_sec = result["trimmed_end_sec"]

        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))

        clip = audio[start_sample:end_sample]
        out_path = OUTPUT_DIR / f"{row['label']}.wav"
        sf.write(out_path, clip, sr, subtype="PCM_16")

        manifest_rows.append({
            "filename": out_path.name,
            "label": row["label"],
            "source_file": source_path.name,
            "source_idx": int(source_idx),
            "orig_start_sec": float(row["start"]),
            "orig_end_sec": float(row["end"]),
            "orig_duration_sec": float(row["end"] - row["start"]),
            "trimmed_start_sec": float(start_sec),
            "trimmed_end_sec": float(end_sec),
            "trimmed_duration_sec": float(end_sec - start_sec),
            "noise_floor": float(result["noise_floor"]),
            "peak_rms": float(result["peak_rms"]),
            "threshold": float(result["threshold"]),
            "used_fallback": bool(result["used_fallback"]),
        })

manifest = pd.DataFrame(manifest_rows).sort_values("label")
manifest.to_csv(MANIFEST_PATH, index=False)

print(f"Wrote {len(manifest)} trimmed clips to {OUTPUT_DIR}")
print(f"Wrote manifest to {MANIFEST_PATH}")
print(manifest.head())