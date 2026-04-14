# Alan Watts Local RAG + TTS

A local Alan Watts-inspired question-answering pipeline that:

1. retrieves relevant Alan Watts source material from a local FAISS index,
2. generates a grounded answer with an OpenAI text model,
3. rewrites the answer slightly for spoken delivery,
4. synthesizes speech using either:
   - the OpenAI TTS pipeline, or
   - a locally fine-tuned XTTS model,
5. optionally applies vintage-style post-processing.

This README is written for the current practical workflow of this repo: a stable OpenAI-backed path, plus a local XTTS path for private use and experimentation.

---

## What this repo can do

There are two main speech backends.

### 1. OpenAI speech backend
Use this when you want the easiest and most stable path.

- text retrieval is local
- text generation uses OpenAI
- speech synthesis uses OpenAI TTS
- easiest setup
- best choice for a clean fallback path

### 2. Local XTTS backend
Use this when you want the answer spoken in your own fine-tuned voice.

- text retrieval is local
- text generation uses OpenAI
- speech synthesis is local via Coqui XTTS
- requires a dedicated XTTS-compatible Python environment
- best for private/local experiments

---

## Current architecture

At a high level:

- `alan_watts_local/cli.py` orchestrates the workflow
- `alan_watts_local/pipeline.py` handles retrieval + generation
- `alan_watts_local/tts/` contains backend-specific speech synthesis code
- `config/local_config.*.yaml` controls behavior
- `scripts/ask_and_speak_openai.sh` runs the OpenAI TTS path
- `scripts/ask_and_speak_xtts.sh` runs the XTTS path

The XTTS backend should be treated as a local/private feature unless you have already thought through hosting cost, latency, abuse controls, and model-license constraints.

---

## Repo layout

A typical layout looks like this:

```text
awcb-tts/
├── alan_watts_local/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── indexing.py
│   ├── ingest.py
│   ├── speech_rewrite.py
│   ├── run_artifacts.py
│   ├── audio/
│   │   ├── playback.py
│   │   └── postprocess.py
│   └── tts/
│       ├── __init__.py
│       ├── openai_tts.py
│       └── xtts_tts.py
├── config/
│   ├── local_config.yaml
│   ├── local_config.openai.yaml
│   └── local_config.xtts.yaml
├── data/
│   ├── watts_rag_corpus_clean.txt
│   ├── processed/
│   └── indexes/
├── models/
│   └── xtts/
│       └── brooks_abd_pilot/
│           ├── model.pth
│           ├── config.json
│           ├── vocab.json
│           ├── dvae.pth
│           └── mel_stats.pth
├── outputs/
│   ├── audio/
│   ├── logs/
│   └── runs/
└── scripts/
    ├── ask_and_speak_openai.sh
    └── ask_and_speak_xtts.sh
```

---

## Environments

This project is easiest to manage with **two separate Python environments**.

### A. Main app environment
Use this for the normal local RAG / OpenAI path.

Typical dependencies include:

- `openai`
- `faiss-cpu`
- `PyYAML`
- `loguru`
- `numpy`
- whatever else your local app already uses

### B. XTTS environment
Use this for the local XTTS backend.

This environment needs:

- Coqui TTS / XTTS-compatible dependencies
- the local app runtime dependencies too
- a NumPy version that remains compatible with the TTS stack

In practice, the XTTS environment often needs more careful pinning than the main app environment.

---

## Core local workflow

### 1. Build or rebuild the local index

```bash
python -m alan_watts_local.cli --config config/local_config.openai.yaml prepare
```

If the index already exists, the CLI usually detects that and skips unnecessary rebuild work.

### 2. Ask a text-only question

```bash
python -m alan_watts_local.cli --config config/local_config.openai.yaml ask --query "What is the self?"
```

### 3. Ask and synthesize speech with OpenAI

```bash
bash scripts/ask_and_speak_openai.sh "What is the self?"
```

### 4. Ask and synthesize speech with XTTS

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?"
```

---

## Configuration strategy

A good stable setup is to keep separate config files:

- `config/local_config.openai.yaml`
- `config/local_config.xtts.yaml`

That way you do not have to constantly overwrite one working configuration with another.

### OpenAI config
Use the OpenAI config when:

- you want the fallback path
- you want easier debugging
- you want a known-stable synthesis path

### XTTS config
Use the XTTS config when:

- you want local speech synthesis
- you want your fine-tuned voice
- you are doing local/private experiments

---

## Suggested XTTS config shape

A practical XTTS config section looks like this:

```yaml
tts:
  backend: xtts_local
  device: cuda
  model_dir: /home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot
  model_path: /home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot/model.pth
  config_path: /home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot/config.json
  vocab_path: /home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot/vocab.json
  dvae_path: /home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot/dvae.pth
  mel_norm_path: /home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot/mel_stats.pth
  language: en
  speaker_wavs:
    - /mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_a1.wav
    - /mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_b1.wav
    - /mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_d1.wav
  speed: 0.92
  use_cache: true
```

For a clean personal-voice output, keep the post-processing disabled by default:

```yaml
audio_postprocess:
  enabled: false
```

Then only enable the vintage effect deliberately when you want it.

---

## Running the OpenAI pipeline

### OpenAI shell script

```bash
bash scripts/ask_and_speak_openai.sh "What is the self?"
```

### With a preset

```bash
bash scripts/ask_and_speak_openai.sh "What is the self?" --preset lecture_warm
```

### Text-only command

```bash
python -m alan_watts_local.cli \
  --config config/local_config.openai.yaml \
  ask \
  --query "What is the self?"
```

Use the OpenAI path when:

- the XTTS environment is being debugged
- you want a stable comparison path
- you want a simple way to verify retrieval/generation independent of local voice synthesis

---

## Running the XTTS pipeline

### XTTS shell script

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?"
```

### XTTS with a preset

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?" --preset lecture_warm
```

The XTTS shell script should:

- activate the XTTS environment,
- point to `config/local_config.xtts.yaml`,
- run `python -m alan_watts_local.cli ...`.

---

## Toggling vintage post-processing

The recommended default for your fine-tuned personal voice is:

- **clean by default**
- **vintage only when requested**

A good CLI interface is:

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?" --raw
bash scripts/ask_and_speak_xtts.sh "What is the self?" --vintage
```

Recommended behavior:

- `--raw` forces clean output
- `--vintage` forces post-processing on
- if neither flag is supplied, the config file default is used

### Recommended default
For your own voice model, I recommend:

- `audio_postprocess.enabled: false` in the XTTS config
- only use `--vintage` for comparison or stylistic experiments

This is especially sensible because the trained voice is already more interesting now, and heavy post-processing can mask whether the model itself is improving.

---

## XTTS smoke test

Before trying the full pipeline, verify the model itself loads cleanly.

```bash
source ~/xtts-venv/bin/activate
python - <<'PY'
from TTS.api import TTS

model_dir = "/home/peacelovephysics/awcb-tts/models/xtts/brooks_abd_pilot"
speaker_wavs = [
    "/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_a1.wav",
    "/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_b1.wav",
    "/mnt/c/Users/Brooks/Documents/Audacity/brooks_tts/combined_abd_xtts_dataset/wavs/ls_d1.wav",
]

tts = TTS(
    model_path=model_dir,
    config_path=f"{model_dir}/config.json",
).to("cuda")

tts.tts_to_file(
    text="This is a direct XTTS smoke test using the Brooks fine tuned checkpoint.",
    speaker_wav=speaker_wavs,
    language="en",
    file_path="tmp_xtts_smoke_test.wav",
)

print("Wrote tmp_xtts_smoke_test.wav")
PY
```

If this works, then the checkpoint and support files are probably wired correctly.

---

## Training a new XTTS model

The training workflow that worked here was:

1. build a clean aligned dataset of short WAV clips plus metadata
2. merge sections into one training dataset
3. fine-tune XTTS on a local GPU
4. choose the best checkpoint based on validation loss, not just latest checkpoint number
5. copy the best checkpoint and support files into the main repo

### 1. Prepare a dataset

The current project used sectioned recordings:

- Section A
- Section B
- Section D

Each dataset ultimately became a folder with:

```text
section_x_xtts_dataset/
├── wavs/
├── metadata_all.csv
├── metadata_train.csv
├── metadata_eval.csv
└── dataset_manifest.csv
```

Metadata format should be compatible with the TTS recipe you are using.

### 2. Merge datasets

The merged dataset ended up as something like:

```text
combined_abd_xtts_dataset/
├── wavs/
├── metadata_all.csv
├── metadata_train.csv
├── metadata_eval.csv
└── dataset_manifest.csv
```

### 3. Fine-tune XTTS

Run the fine-tuning recipe from the XTTS-compatible environment.

Keep the pilot run conservative:

- batch size small
- gradient accumulation large enough for memory limits
- save checkpoints regularly
- watch eval loss, not just training loss

### 4. Choose the best model

Use the **best validated checkpoint**, not just the latest checkpoint.

Good rule:

- prefer `best_model_XXXXX.pth`
- do **not** assume `checkpoint_XXXXX.pth` is better just because it is later

### 5. Copy the model into the repo

Copy the winning checkpoint plus auxiliary files into:

```text
models/xtts/brooks_abd_pilot/
```

Recommended final layout:

```text
models/xtts/brooks_abd_pilot/
├── model.pth
├── config.json
├── vocab.json
├── dvae.pth
└── mel_stats.pth
```

`model.pth` can simply be a renamed copy of your selected best checkpoint.

---

## Wiring a newly trained XTTS model into the pipeline

### 1. Copy the selected best checkpoint

Example:

```bash
cp best_model_24570.pth ~/awcb-tts/models/xtts/brooks_abd_pilot/model.pth
```

### 2. Copy the support files

Also copy:

- `config.json`
- `vocab.json`
- `dvae.pth`
- `mel_stats.pth`

### 3. Update the XTTS config

Point `config/local_config.xtts.yaml` at the new model directory.

### 4. Verify the XTTS loader

The XTTS backend should load from the **model directory**, not from `.../model.pth/model.pth`.

A working pattern is:

```python
TTS(
    model_path=model_dir,
    config_path=config_path,
).to(device)
```

### 5. Test the direct smoke path first

Always test the model alone before debugging the whole chatbot pipeline.

---

## Suggested config architecture

A good long-term cleanup is:

- one unified `TTSConfig` dataclass that supports both OpenAI and XTTS fields
- deep preset merge instead of shallow overwrite
- lazy imports in `alan_watts_local/tts/__init__.py`
- no hardcoded XTTS rescue paths once config loading is fixed

This avoids the common failure mode where:

- `backend` survives,
- but XTTS-specific fields like `model_dir` are silently dropped.

---

## Recommended preset strategy

Use presets for:

- speech rewrite length
- speaking speed
- audio post-processing style

Do **not** let presets accidentally wipe out the entire XTTS config block.

That means your config loader should apply presets with a **deep merge**.

---

## Does Section C help?

Probably yes, but only if you add it carefully.

### Section C is likely worth adding if:

- it contains longer, smoother lecture-style phrasing
- it is cleanly segmented
- the text/audio alignment is trustworthy
- it introduces useful cadence variation without adding noise

### Section C may help with:

- longer sentence carry-through
- more natural phrase transitions
- lecture-like pacing
- reducing the “assembled from short clips” feeling

### Section C may **not** help if:

- the clips are noisy or poorly segmented
- the alignment is inconsistent
- the dataset starts mixing too many recording conditions

### Practical recommendation

Do **not** dump all of Section C in blindly.

Instead:

1. curate a small clean subset first,
2. fine-tune or continue-training with that subset added,
3. compare raw output before and after.

Because your current model is already working well, the right question is no longer “can we make it work at all?” but “does Section C improve transitions without degrading clarity?”

That should be tested empirically.

---

## About sentence-boundary artifacts

If you hear odd artifacts between sentences, the cause may be one of several things:

- the XTTS model itself
- the way the speech rewrite breaks or structures sentences
- silence or boundary behavior in the synthesizer output
- the post-processing stage
- concatenation or normalization choices downstream

### Best debugging order

1. compare `latest_raw.wav` vs processed output
2. test with post-processing disabled
3. test shorter answers
4. inspect whether the speech rewrite creates abrupt sentence boundaries
5. only then decide whether new training data is needed

Because you now prefer your own voice clean and clear, keep post-processing off by default while evaluating core model quality.

---

## Run artifacts

If run artifacts are enabled, the pipeline can save:

- raw audio
- final audio
- generation context / metadata
- history logs
- run bundles

That is useful for comparing outputs across:

- presets
- TTS backends
- vintage vs raw
- different XTTS checkpoints

---

## Practical recommendations going forward

### Best current default

- use XTTS locally
- keep post-processing off by default
- use `--vintage` only for experiments
- treat `best_model_24570.pth` or whichever best checkpoint won validation as the canonical model until a better one is proven

### Best next experiments

1. compare raw vs vintage on the same prompt
2. inspect sentence-boundary artifacts on raw audio
3. add a small clean subset of Section C
4. retrain and compare A/B
5. keep the OpenAI path as a reliability fallback

---

## Example command summary

### Prepare index

```bash
python -m alan_watts_local.cli --config config/local_config.openai.yaml prepare
```

### Ask only

```bash
python -m alan_watts_local.cli --config config/local_config.openai.yaml ask --query "What is the self?"
```

### OpenAI speech

```bash
bash scripts/ask_and_speak_openai.sh "What is the self?"
```

### XTTS speech

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?"
```

### XTTS speech with preset

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?" --preset lecture_warm
```

### Force raw / clean output

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?" --raw
```

### Force vintage output

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?" --vintage
```

---

## Final note

At this point, the project has crossed the line from “interesting experiment” into “working local system.”

The main remaining work is no longer basic viability. It is refinement:

- cleaner config handling
- smoother sentence transitions
- thoughtful use of Section C
- and clearer separation between raw voice quality and stylistic post-processing

That is a good place to be.
