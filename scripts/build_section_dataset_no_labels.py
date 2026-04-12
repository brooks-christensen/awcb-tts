from __future__ import annotations

from pathlib import Path
import json
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf


# -----------------------------
# USER CONFIG
# -----------------------------
BASE_DIR = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts")

SECTION_CONFIG = {
    "D": {
        "source_dir": BASE_DIR / "section_d",
        "lines_path": BASE_DIR / "section_d_lines.txt",
        "dataset_dir": BASE_DIR / "section_d_xtts_dataset",
        "raw_expected_segments_per_source": [10, 10],
        "final_lines_per_source": [10, 10],
        "duplicate_rules": {},
    },
    "B": {
        "source_dir": BASE_DIR / "section_b_sources",
        "lines_path": BASE_DIR / "section_b_lines.txt",
        "dataset_dir": BASE_DIR / "section_b_xtts_dataset",
        "raw_expected_segments_per_source": [10, 10, 11, 11, 10, 10],
        "final_lines_per_source": [10, 10, 10, 10, 10, 10],
        # Assumptions:
        # - In source 3, line 21 was read twice, immediately at the start of that source.
        # - In source 4, line 36 was read twice, immediately after lines 31-35.
        # keep_local_segment is the one we keep as canonical.
        "duplicate_rules": {
            3: {
                "global_line_num": 21,
                "duplicate_local_segments": [1, 2],
                "keep_local_segment": 2,  # usually keep the second take
            },
            4: {
                "global_line_num": 36,
                "duplicate_local_segments": [6, 7],
                "keep_local_segment": 7,  # usually keep the second take
            },
        },
    },
}

RANDOM_SEED = 42
EVAL_FRACTION = 0.10

FRAME_MS = 20
HOP_MS = 10
THRESH_MULTIPLIER = 3.0
MIN_THRESHOLD = 1e-4
MAX_THRESHOLD_FRAC = 0.20
MERGE_GAP_SEC = 0.35
MIN_REGION_SEC = 0.50
KEEP_HEAD_SEC = 0.04
KEEP_TAIL_SEC = 0.08


# -----------------------------
# Helpers
# -----------------------------
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


def detect_regions(audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    frame_len = max(1, int(round(FRAME_MS / 1000 * sr)))
    hop_len = max(1, int(round(HOP_MS / 1000 * sr)))

    rms = frame_rms(audio, frame_len, hop_len)
    noise_floor = np.percentile(rms, 20)
    peak_rms = float(np.max(rms))

    threshold = max(MIN_THRESHOLD, noise_floor * THRESH_MULTIPLIER)
    threshold = min(threshold, peak_rms * MAX_THRESHOLD_FRAC)

    active = rms >= threshold
    if not np.any(active):
        return []

    regions = []
    start_idx = None
    for i, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = i
        elif not is_active and start_idx is not None:
            end_idx = i - 1
            start_sec = (start_idx * hop_len) / sr
            end_sec = ((end_idx * hop_len) + frame_len) / sr
            regions.append((start_sec, end_sec))
            start_idx = None

    if start_idx is not None:
        end_idx = len(active) - 1
        start_sec = (start_idx * hop_len) / sr
        end_sec = ((end_idx * hop_len) + frame_len) / sr
        regions.append((start_sec, end_sec))

    # Merge small gaps
    merged = []
    for start_sec, end_sec in regions:
        if not merged:
            merged.append([start_sec, end_sec])
            continue
        prev_start, prev_end = merged[-1]
        if start_sec - prev_end <= MERGE_GAP_SEC:
            merged[-1][1] = end_sec
        else:
            merged.append([start_sec, end_sec])

    # Filter short regions, add head/tail padding
    padded = []
    total_sec = len(audio) / sr
    for start_sec, end_sec in merged:
        if (end_sec - start_sec) < MIN_REGION_SEC:
            continue
        s = max(0.0, start_sec - KEEP_HEAD_SEC)
        e = min(total_sec, end_sec + KEEP_TAIL_SEC)
        padded.append((s, e))

    return padded


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return (
        text.replace("—", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
            .strip()
    )


def write_metadata_lines(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f'{row["filename"]}|{row["text"]}|{row["normalized_text"]}\n')


def split_train_eval(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    idxs = list(range(len(rows)))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(idxs)

    eval_size = max(1, int(len(rows) * EVAL_FRACTION))
    eval_idxs = set(idxs[:eval_size])

    train_rows, eval_rows = [], []
    for i, row in enumerate(rows):
        if i in eval_idxs:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows

def coerce_regions_to_expected_count(
    regions,
    expected_count,
    short_region_sec=1.20,
    merge_gap_sec=0.60,
):
    """
    Try to reduce over-segmentation when we know how many utterances we expect.

    Strategy:
    1. Merge very short regions into the nearest neighbor when reasonable.
    2. If still too many, merge across the smallest gap until expected_count is reached.
    """
    regions = [list(r) for r in regions]

    def region_duration(r):
        return r[1] - r[0]

    def gap(left, right):
        return right[0] - left[1]

    changed = True
    while len(regions) > expected_count and changed:
        changed = False

        # Pass 1: merge suspiciously short regions into neighbors
        for i in range(len(regions)):
            dur = region_duration(regions[i])
            if dur >= short_region_sec:
                continue

            left_gap = gap(regions[i - 1], regions[i]) if i > 0 else None
            right_gap = gap(regions[i], regions[i + 1]) if i < len(regions) - 1 else None

            # Choose nearest merge candidate
            candidates = []
            if left_gap is not None and left_gap <= merge_gap_sec:
                candidates.append(("left", left_gap))
            if right_gap is not None and right_gap <= merge_gap_sec:
                candidates.append(("right", right_gap))

            if not candidates:
                continue

            side = min(candidates, key=lambda x: x[1])[0]

            if side == "left":
                regions[i - 1][1] = regions[i][1]
                del regions[i]
            else:
                regions[i + 1][0] = regions[i][0]
                del regions[i]

            changed = True
            break

    # Pass 2: if still too many, merge across the smallest gap
    while len(regions) > expected_count:
        gaps = []
        for i in range(len(regions) - 1):
            gaps.append((i, regions[i + 1][0] - regions[i][1]))

        if not gaps:
            break

        i_min, _ = min(gaps, key=lambda x: x[1])
        regions[i_min][1] = regions[i_min + 1][1]
        del regions[i_min + 1]

    return [tuple(r) for r in regions]


# -----------------------------
# Main builder
# -----------------------------
def build_section(section: str) -> None:
    cfg = SECTION_CONFIG[section]
    source_dir: Path = cfg["source_dir"]
    lines_path: Path = cfg["lines_path"]
    dataset_dir: Path = cfg["dataset_dir"]
    wavs_dir = dataset_dir / "wavs"
    rejected_dir = dataset_dir / "rejected_takes"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    lines = read_lines(lines_path)
    final_total_expected = sum(cfg["final_lines_per_source"])
    assert len(lines) == final_total_expected, (
        f"{section}: expected {final_total_expected} script lines, found {len(lines)}"
    )

    source_files = sorted(source_dir.glob("*.wav"))
    assert len(source_files) == len(cfg["raw_expected_segments_per_source"]), (
        f"{section}: expected {len(cfg['raw_expected_segments_per_source'])} source wavs, found {len(source_files)}"
    )

    rows = []
    line_cursor = 0

    for source_idx, source_path in enumerate(source_files, start=1):
        audio, sr = sf.read(source_path)
        audio = to_mono(np.asarray(audio))

        regions = detect_regions(audio, sr)
        expected_raw = cfg["raw_expected_segments_per_source"][source_idx - 1]

        if len(regions) > expected_raw:
            regions = coerce_regions_to_expected_count(
                regions,
                expected_count=expected_raw,
                short_region_sec=1.20,
                merge_gap_sec=0.60,
            )

        if len(regions) != expected_raw:
            raise ValueError(
                f"{section} source {source_idx}: detected {len(regions)} regions, expected {expected_raw}. "
                f"Manual review needed for {source_path.name}."
            )

        duplicate_rule = cfg["duplicate_rules"].get(source_idx)
        keep_mask = [True] * len(regions)

        if duplicate_rule:
            dup_local = duplicate_rule["duplicate_local_segments"]
            keep_local = duplicate_rule["keep_local_segment"]
            for seg_idx_1based in dup_local:
                if seg_idx_1based != keep_local:
                    keep_mask[seg_idx_1based - 1] = False

        kept_regions = [r for keep, r in zip(keep_mask, regions) if keep]
        expected_final = cfg["final_lines_per_source"][source_idx - 1]

        if len(kept_regions) != expected_final:
            raise ValueError(
                f"{section} source {source_idx}: kept {len(kept_regions)} regions, expected {expected_final} "
                f"after duplicate handling."
            )

        # Save rejected duplicate takes for audit
        if duplicate_rule:
            global_line_num = duplicate_rule["global_line_num"]
            for seg_idx_1based in duplicate_rule["duplicate_local_segments"]:
                if seg_idx_1based == duplicate_rule["keep_local_segment"]:
                    continue
                start_sec, end_sec = regions[seg_idx_1based - 1]
                s0, s1 = int(round(start_sec * sr)), int(round(end_sec * sr))
                clip = audio[s0:s1]
                rejected_name = f"ls_{section.lower()}{global_line_num}_take_rejected_{seg_idx_1based}.wav"
                sf.write(rejected_dir / rejected_name, clip, sr, subtype="PCM_16")

        # Export kept clips and pair to script lines
        for local_idx, (start_sec, end_sec) in enumerate(kept_regions, start=1):
            global_line_num = line_cursor + 1
            text = lines[line_cursor]
            normalized_text = normalize_text(text)
            filename = f"ls_{section.lower()}{global_line_num}.wav"

            s0, s1 = int(round(start_sec * sr)), int(round(end_sec * sr))
            clip = audio[s0:s1]
            sf.write(wavs_dir / filename, clip, sr, subtype="PCM_16")

            rows.append({
                "filename": filename,
                "label": f"ls_{section.lower()}{global_line_num}",
                "label_num": global_line_num,
                "section": section,
                "source_file": source_path.name,
                "source_idx": source_idx,
                "local_idx_within_source": local_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "text": text,
                "normalized_text": normalized_text,
            })
            line_cursor += 1

    assert line_cursor == len(lines), f"{section}: consumed {line_cursor} lines but had {len(lines)}"

    df = pd.DataFrame(rows).sort_values("label_num").reset_index(drop=True)
    df.to_csv(dataset_dir / "dataset_manifest.csv", index=False)

    metadata_rows = df[["filename", "text", "normalized_text"]].to_dict("records")
    train_rows, eval_rows = split_train_eval(metadata_rows)

    write_metadata_lines(dataset_dir / "metadata_all.csv", metadata_rows)
    write_metadata_lines(dataset_dir / "metadata_train.csv", train_rows)
    write_metadata_lines(dataset_dir / "metadata_eval.csv", eval_rows)

    summary = {
        "section": section,
        "dataset_dir": str(dataset_dir),
        "wavs_dir": str(wavs_dir),
        "rejected_dir": str(rejected_dir),
        "total_clips": len(df),
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nFirst 3 metadata lines:")
    for row in metadata_rows[:3]:
        print(f'{row["filename"]}|{row["text"]}|{row["normalized_text"]}')


if __name__ == "__main__":
    # Change to "B" when ready
    build_section("D")