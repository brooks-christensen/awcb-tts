from pathlib import Path
import shutil
import pandas as pd
import json
import random

BASE_DIR = Path("/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts")

DATASET_DIRS = [
    BASE_DIR / "section_a_xtts_dataset",
    BASE_DIR / "section_d_xtts_dataset",
]

OUT_DIR = BASE_DIR / "combined_ad_xtts_dataset"
WAVS_DIR = OUT_DIR / "wavs"

RANDOM_SEED = 42
EVAL_FRACTION = 0.10

OUT_DIR.mkdir(parents=True, exist_ok=True)
WAVS_DIR.mkdir(parents=True, exist_ok=True)

all_rows = []

for ds in DATASET_DIRS:
    manifest = pd.read_csv(ds / "dataset_manifest.csv")
    for _, row in manifest.iterrows():
        src = ds / "wavs" / row["filename"]
        dst = WAVS_DIR / row["filename"]
        shutil.copy2(src, dst)
        all_rows.append({
            "filename": row["filename"],
            "text": row["text"],
            "normalized_text": row["normalized_text"],
        })

df = pd.DataFrame(all_rows).sort_values("filename").reset_index(drop=True)

with (OUT_DIR / "metadata_all.csv").open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f'{row["filename"]}|{row["text"]}|{row["normalized_text"]}\n')

idxs = list(range(len(df)))
rng = random.Random(RANDOM_SEED)
rng.shuffle(idxs)
eval_size = max(1, int(len(df) * EVAL_FRACTION))
eval_idxs = set(idxs[:eval_size])

with (OUT_DIR / "metadata_train.csv").open("w", encoding="utf-8") as f_train, \
     (OUT_DIR / "metadata_eval.csv").open("w", encoding="utf-8") as f_eval:
    for i, row in df.iterrows():
        line = f'{row["filename"]}|{row["text"]}|{row["normalized_text"]}\n'
        if i in eval_idxs:
            f_eval.write(line)
        else:
            f_train.write(line)

df.to_csv(OUT_DIR / "dataset_manifest.csv", index=False)
(OUT_DIR / "dataset_summary.json").write_text(
    json.dumps({"total_clips": len(df), "out_dir": str(OUT_DIR)}, indent=2),
    encoding="utf-8",
)

print(f"Combined dataset written to: {OUT_DIR}")
print(f"Total clips: {len(df)}")
