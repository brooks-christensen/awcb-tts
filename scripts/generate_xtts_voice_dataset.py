from pathlib import Path
import pandas as pd
import shutil
import random
import json

# -----------------------------
# EDIT THESE PATHS
# -----------------------------
BASE_DIR = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts")
CLIPS_DIR = BASE_DIR / "section_a_clips_trimmed"
TRIMMED_MANIFEST_PATH = CLIPS_DIR / "section_a_trimmed_manifest.csv"
LINES_PATH = BASE_DIR / "section_a_lines.txt"

# Final XTTS dataset output
DATASET_DIR = BASE_DIR / "section_a_xtts_dataset"
WAVS_DIR = DATASET_DIR / "wavs"

# Split settings
EVAL_FRACTION = 0.10
RANDOM_SEED = 42

# -----------------------------
# SETUP
# -----------------------------
DATASET_DIR.mkdir(parents=True, exist_ok=True)
WAVS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD TRIMMED MANIFEST
# -----------------------------
manifest_df = pd.read_csv(TRIMMED_MANIFEST_PATH)

# Sort labels numerically: ls_a1, ls_a2, ..., ls_a180
manifest_df["label_num"] = (
    manifest_df["label"].astype(str).str.extract(r"(\d+)$").astype(int)
)
manifest_df = manifest_df.sort_values("label_num").reset_index(drop=True)

# -----------------------------
# LOAD SCRIPT LINES
# -----------------------------
lines = [
    line.strip()
    for line in LINES_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

assert len(manifest_df) == len(lines), (
    f"Clip count ({len(manifest_df)}) does not match line count ({len(lines)})"
)

# -----------------------------
# ATTACH TEXT
# -----------------------------
manifest_df["text"] = lines

# Keep normalization simple for now
manifest_df["normalized_text"] = (
    manifest_df["text"]
    .str.replace("—", "-", regex=False)
    .str.replace("“", '"', regex=False)
    .str.replace("”", '"', regex=False)
    .str.replace("’", "'", regex=False)
    .str.replace("‘", "'", regex=False)
    .str.strip()
)

# -----------------------------
# VERIFY WAVS EXIST + COPY INTO FINAL DATASET
# -----------------------------
missing = []
for fn in manifest_df["filename"]:
    src = CLIPS_DIR / fn
    dst = WAVS_DIR / fn
    if not src.exists():
        missing.append(str(src))
        continue
    shutil.copy2(src, dst)

if missing:
    raise FileNotFoundError(f"Missing WAV files, first few: {missing[:5]}")

# -----------------------------
# BUILD METADATA LINES
# -----------------------------
metadata_lines = [
    f'{row.filename}|{row.text}|{row.normalized_text}'
    for row in manifest_df.itertuples(index=False)
]

# Write all samples
(DATASET_DIR / "metadata_all.csv").write_text(
    "\n".join(metadata_lines) + "\n",
    encoding="utf-8",
)

# -----------------------------
# TRAIN / EVAL SPLIT
# -----------------------------
indices = list(range(len(manifest_df)))
rng = random.Random(RANDOM_SEED)
rng.shuffle(indices)

eval_size = max(1, int(len(indices) * EVAL_FRACTION))
eval_idx = set(indices[:eval_size])

train_lines = []
eval_lines = []

for i, line in enumerate(metadata_lines):
    if i in eval_idx:
        eval_lines.append(line)
    else:
        train_lines.append(line)

(DATASET_DIR / "metadata_train.csv").write_text(
    "\n".join(train_lines) + "\n",
    encoding="utf-8",
)

(DATASET_DIR / "metadata_eval.csv").write_text(
    "\n".join(eval_lines) + "\n",
    encoding="utf-8",
)

# -----------------------------
# OPTIONAL HUMAN-READABLE MANIFEST
# -----------------------------
human_df = manifest_df[[
    "filename",
    "label",
    "label_num",
    "source_file",
    "source_idx",
    "orig_start_sec",
    "orig_end_sec",
    "orig_duration_sec",
    "trimmed_start_sec",
    "trimmed_end_sec",
    "trimmed_duration_sec",
    "text",
    "normalized_text",
]].copy()

human_df.to_csv(DATASET_DIR / "dataset_manifest.csv", index=False)

# -----------------------------
# SUMMARY
# -----------------------------
summary = {
    "dataset_dir": str(DATASET_DIR),
    "wavs_dir": str(WAVS_DIR),
    "total_clips": len(manifest_df),
    "train_clips": len(train_lines),
    "eval_clips": len(eval_lines),
    "metadata_all": str(DATASET_DIR / "metadata_all.csv"),
    "metadata_train": str(DATASET_DIR / "metadata_train.csv"),
    "metadata_eval": str(DATASET_DIR / "metadata_eval.csv"),
    "dataset_manifest": str(DATASET_DIR / "dataset_manifest.csv"),
}

(DATASET_DIR / "dataset_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, indent=2))
print("\nFirst 3 metadata lines:")
for line in metadata_lines[:3]:
    print(line)