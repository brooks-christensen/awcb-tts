from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import soundfile as sf


BASE_DIR = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts")
LABELS_PATH = BASE_DIR / "labels_d.txt"
SOURCE_DIR = BASE_DIR / "section_d"
LINES_PATH = BASE_DIR / "section_d_lines.txt"

OUT_DIR = BASE_DIR / "section_d_xtts_dataset"
WAVS_DIR = OUT_DIR / "wavs"

RANDOM_SEED = 42
EVAL_FRACTION = 0.10

SEARCH_PAD_SEC = 0.20
KEEP_HEAD_SEC = 0.04
KEEP_TAIL_SEC = 0.08

FRAME_MS = 20
HOP_MS = 10
THRESH_MULTIPLIER = 3.0
MIN_THRESHOLD = 1e-4
MAX_THRESHOLD_FRACTION = 0.20


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def frame_rms(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        padded = np.pad(x, (0, max(0, frame_len - len(x))))
        return np.array([np.sqrt(np.mean(padded ** 2) + 1e-12)], dtype=np.float32)

    vals = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start:start + frame_len]
        vals.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
    return np.array(vals, dtype=np.float32)


def auto_trim_region(
    audio: np.ndarray,
    sr: int,
    coarse_start_sec: float,
    coarse_end_sec: float,
):
    n_samples = len(audio)

    search_start_sec = max(0.0, coarse_start_sec - SEARCH_PAD_SEC)
    search_end_sec = min(n_samples / sr, coarse_end_sec + SEARCH_PAD_SEC)

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

    noise_floor = np.percentile(rms, 20)
    peak_rms = np.max(rms)

    threshold = max(MIN_THRESHOLD, noise_floor * THRESH_MULTIPLIER)
    threshold = min(threshold, peak_rms * MAX_THRESHOLD_FRACTION)

    active = np.where(rms >= threshold)[0]

    if len(active) == 0:
        trimmed_start_sec = max(0.0, coarse_start_sec - KEEP_HEAD_SEC)
        trimmed_end_sec = min(n_samples / sr, coarse_end_sec + KEEP_TAIL_SEC)
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

    trimmed_start_sample = max(0, int(round(detected_start_sample - KEEP_HEAD_SEC * sr)))
    trimmed_end_sample = min(n_samples, int(round(detected_end_sample + KEEP_TAIL_SEC * sr)))

    return {
        "trimmed_start_sec": trimmed_start_sample / sr,
        "trimmed_end_sec": trimmed_end_sample / sr,
        "threshold": float(threshold),
        "noise_floor": float(noise_floor),
        "peak_rms": float(peak_rms),
        "used_fallback": False,
    }


def normalize_text(text: str) -> str:
    return (
        text.replace("—", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
            .strip()
    )


def write_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f'{row["filename"]}|{row["text"]}|{row["normalized_text"]}\n')


def split_train_eval(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    idxs = list(range(len(rows)))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(idxs)
    eval_size = max(1, int(len(rows) * EVAL_FRACTION))
    eval_idxs = set(idxs[:eval_size])

    train_rows = []
    eval_rows = []
    for i, row in enumerate(rows):
        if i in eval_idxs:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(
        LABELS_PATH,
        sep="\t",
        header=None,
        names=["start", "end", "label"],
    )

    labels["start"] = pd.to_numeric(labels["start"], errors="coerce")
    labels["end"] = pd.to_numeric(labels["end"], errors="coerce")
    labels["label"] = labels["label"].astype(str).str.strip()
    labels["label_num"] = labels["label"].str.extract(r"(\d+)$").astype(int)
    labels = labels.sort_values("label_num").reset_index(drop=True)

    assert len(labels) == 20, f"Expected 20 labels, found {len(labels)}"
    assert labels["label_num"].min() == 1, "Expected labels to start at 1"
    assert labels["label_num"].max() == 20, "Expected labels to end at 20"

    labels["source_idx"] = ((labels["label_num"] - 1) // 10) + 1

    source_files = sorted(SOURCE_DIR.glob("*.wav"))
    assert len(source_files) == 2, f"Expected 2 source wav files, found {len(source_files)}"

    lines = [
        line.strip()
        for line in LINES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 20, f"Expected 20 section D lines, found {len(lines)}"

    manifest_rows = []

    for source_idx in range(1, 3):
        source_path = source_files[source_idx - 1]
        audio, sr = sf.read(source_path)
        audio = to_mono(np.asarray(audio, dtype=np.float32))

        group = labels[labels["source_idx"] == source_idx].copy()
        assert len(group) == 10, f"Expected 10 labels for source {source_idx}, found {len(group)}"

        for _, row in group.iterrows():
            trim = auto_trim_region(
                audio=audio,
                sr=sr,
                coarse_start_sec=float(row["start"]),
                coarse_end_sec=float(row["end"]),
            )
            if trim is None:
                raise ValueError(f"Failed to trim {row['label']} from {source_path.name}")

            start_sec = trim["trimmed_start_sec"]
            end_sec = trim["trimmed_end_sec"]

            s0 = int(round(start_sec * sr))
            s1 = int(round(end_sec * sr))
            clip = audio[s0:s1]

            filename = f"{row['label']}.wav"
            out_path = WAVS_DIR / filename
            sf.write(out_path, clip, sr, subtype="PCM_16")

            line_idx = int(row["label_num"]) - 1
            text = lines[line_idx]

            manifest_rows.append({
                "filename": filename,
                "label": row["label"],
                "label_num": int(row["label_num"]),
                "source_file": source_path.name,
                "source_idx": int(source_idx),
                "orig_start_sec": float(row["start"]),
                "orig_end_sec": float(row["end"]),
                "orig_duration_sec": float(row["end"] - row["start"]),
                "trimmed_start_sec": float(start_sec),
                "trimmed_end_sec": float(end_sec),
                "trimmed_duration_sec": float(end_sec - start_sec),
                "text": text,
                "normalized_text": normalize_text(text),
                "noise_floor": trim["noise_floor"],
                "peak_rms": trim["peak_rms"],
                "threshold": trim["threshold"],
                "used_fallback": trim["used_fallback"],
            })

    df = pd.DataFrame(manifest_rows).sort_values("label_num").reset_index(drop=True)
    df.to_csv(OUT_DIR / "dataset_manifest.csv", index=False)

    metadata_rows = df[["filename", "text", "normalized_text"]].to_dict("records")
    write_metadata(OUT_DIR / "metadata_all.csv", metadata_rows)

    train_rows, eval_rows = split_train_eval(metadata_rows)
    write_metadata(OUT_DIR / "metadata_train.csv", train_rows)
    write_metadata(OUT_DIR / "metadata_eval.csv", eval_rows)

    summary = {
        "dataset_dir": str(OUT_DIR),
        "wavs_dir": str(WAVS_DIR),
        "total_clips": len(df),
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
        "fallback_count": int(df["used_fallback"].sum()),
        "metadata_all": str(OUT_DIR / "metadata_all.csv"),
        "metadata_train": str(OUT_DIR / "metadata_train.csv"),
        "metadata_eval": str(OUT_DIR / "metadata_eval.csv"),
        "dataset_manifest": str(OUT_DIR / "dataset_manifest.csv"),
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nFirst 3 metadata lines:")
    for row in metadata_rows[:3]:
        print(f'{row["filename"]}|{row["text"]}|{row["normalized_text"]}')


if __name__ == "__main__":
    main()
