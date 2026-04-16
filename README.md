# Alan Watts Local RAG + TTS

A local Alan Watts-inspired question-answering and speech-generation pipeline that can:

1. retrieve relevant Alan Watts source material from a local FAISS index,
2. generate a grounded answer with an OpenAI text model,
3. optionally rewrite the answer for spoken delivery,
4. synthesize speech using either:
   - an OpenAI TTS backend, or
   - a local fine-tuned XTTS backend,
5. optionally apply vintage-style post-processing,
6. save reproducible run artifacts for later comparison.

This README reflects the current practical workflow of the repo:

- a stable OpenAI-backed path,
- a local XTTS path for personal/private use,
- and a standalone **speak-text** workflow for generating audio directly from user-provided text without running retrieval or answer generation.

---

## What this repo can do

There are now **three** main usage patterns.

### 1. Ask a question and get text only
Use the local RAG + OpenAI generation path without speech.

### 2. Ask a question and get synthesized speech
Use the full pipeline:

- retrieval is local,
- answer generation uses OpenAI,
- speech synthesis uses either OpenAI TTS or local XTTS,
- run artifacts can be saved automatically.

### 3. Generate speech directly from supplied text
Use the standalone speech path when you already have text and only want audio output.

This is especially useful for:

- smoke testing the XTTS model,
- reading arbitrary text aloud,
- comparing presets,
- generating short narration clips,
- testing post-processing independently of the RAG pipeline.

---

## Current architecture

At a high level:

- `alan_watts_local/cli.py` orchestrates the workflow
- `alan_watts_local/pipeline.py` handles retrieval + generation
- `alan_watts_local/speech_rewrite.py` adapts answers for spoken delivery
- `alan_watts_local/run_artifacts.py` saves reproducible outputs
- `alan_watts_local/tts/` contains backend-specific speech synthesis code
- `config/local_config.*.yaml` controls behavior
- `scripts/ask_and_speak_openai.sh` runs the OpenAI TTS path
- `scripts/ask_and_speak_xtts.sh` runs the XTTS question-answer + speech path
- `scripts/speak_text_xtts.sh` runs the standalone XTTS text-to-speech path

The XTTS backend should be treated as a local/private feature unless you have already thought through hosting cost, latency, abuse controls, and license constraints.

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
    ├── ask_and_speak_xtts.sh
    └── speak_text_xtts.sh
```

Depending on your config, output artifacts may be written inside the repo or redirected to a Windows-mounted folder such as `/mnt/c/Users/<name>/Documents/...`.

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
- any other app-specific dependencies already used by the local pipeline

### B. XTTS environment
Use this for the local XTTS backend.

This environment needs:

- Coqui TTS / XTTS-compatible dependencies
- the local app runtime dependencies too
- a NumPy version compatible with the TTS stack

In practice, the XTTS environment often needs tighter pinning than the main app environment.

---

## Core workflows

### 1. Build or rebuild the local index

```bash
python -m alan_watts_local.cli --config config/local_config.openai.yaml prepare
```

If the index already exists, the CLI usually detects that and skips unnecessary rebuild work.

### 2. Ask a text-only question

```bash
python -m alan_watts_local.cli \
  --config config/local_config.openai.yaml \
  ask \
  --query "What is the self?"
```

### 3. Ask and synthesize speech with OpenAI

```bash
bash scripts/ask_and_speak_openai.sh "What is the self?"
```

### 4. Ask and synthesize speech with XTTS

```bash
bash scripts/ask_and_speak_xtts.sh "What is the self?"
```

### 5. Generate speech directly from input text with XTTS

```bash
bash scripts/speak_text_xtts.sh --text "Hello from Brooks."
```

### 6. Generate speech from a text file with XTTS

```bash
bash scripts/speak_text_xtts.sh --text-file /path/to/input.txt
```

---

## Standalone speak-text feature

The standalone **speak-text** feature is the simplest way to use the fine-tuned XTTS model directly.

It skips:

- retrieval,
- OpenAI answer generation,
- speech rewriting tied to the RAG flow.

It only does:

- environment activation,
- config loading,
- XTTS synthesis,
- optional preset selection,
- organized audio output.

### Typical uses

- quick voice smoke tests,
- reading arbitrary text aloud,
- checking pacing and tone,
- comparing presets like `clean_modern` vs `lecture_warm`,
- generating local narration clips.

### Example commands

#### Inline text

```bash
bash scripts/speak_text_xtts.sh --text "Hello from Brooks." --preset clean_modern
```

#### Text file input

```bash
bash scripts/speak_text_xtts.sh \
  --text-file /mnt/c/Users/Brooks/Documents/test_tts.txt \
  --preset lecture_warm
```

#### Explicit CLI form

```bash
python -m alan_watts_local.cli \
  --config config/local_config.xtts.yaml \
  --preset clean_modern \
  speak-text \
  --text "Hello from Brooks."
```

### Output behavior

The standalone speak path is intended to write timestamped WAV files rather than reusing a single `latest_*.wav` file.

A practical output pattern is:

```text
/mnt/c/Users/Brooks/Documents/awcb_tts/outputs/audio/YYYY-MM-DD/
  20260414_183505_hello-from-brooks_raw.wav
```

This keeps generated audio:

- organized by day,
- easy to browse from Windows,
- outside the WSL disk image when desired,
- safe from accidental overwrite.

---

## Output storage strategy

For this repo, the most practical setup is usually:

- keep code and models in WSL,
- keep generated artifacts on the Windows side.

That means configuring paths such as:

```yaml
paths:
  audio_dir: /mnt/c/Users/Brooks/Documents/awcb_tts/outputs/audio
  tts_cache_dir: /mnt/c/Users/Brooks/Documents/awcb_tts/outputs/audio/cache
  runs_dir: /mnt/c/Users/Brooks/Documents/awcb_tts/outputs/runs
  logs_dir: /mnt/c/Users/Brooks/Documents/awcb_tts/outputs/logs
```

This helps prevent unnecessary growth of the WSL virtual disk while keeping outputs easy to access from the Windows desktop environment.

---

## Configuration strategy

A stable setup is to keep separate config files:

- `config/local_config.openai.yaml`
- `config/local_config.xtts.yaml`

That way you do not have to constantly overwrite one working configuration with another.

### OpenAI config
Use the OpenAI config when:

- you want the fallback path,
- you want easier debugging,
- you want a known-stable synthesis path.

### XTTS config
Use the XTTS config when:

- you want local speech synthesis,
- you want your fine-tuned voice,
- you are doing local/private experiments,
- you want to use the standalone speak-text path.

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

- the XTTS environment is being debugged,
- you want a stable comparison path,
- you want a simple way to verify retrieval/generation independent of local voice synthesis.

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

### XTTS standalone speak-text shell script

```bash
bash scripts/speak_text_xtts.sh --text "Hello from Brooks." --preset clean_modern
```

This is the best first test when:

- you only want to verify XTTS loading,
- you want to compare presets,
- you do not want retrieval or generation in the loop.

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

## Recommended preset strategy

Use presets for:

- speech rewrite length
- speaking speed
- audio post-processing style

Do **not** let presets accidentally wipe out the entire XTTS config block.

That means your config loader should apply presets with a **deep merge**.

---

## About sentence-boundary artifacts

If you hear odd artifacts between sentences, the cause may be one of several things:

- the XTTS model itself
- the way the speech rewrite breaks or structures sentences
- silence or boundary behavior in the synthesizer output
- the post-processing stage
- concatenation or normalization choices downstream

### Best debugging order

1. compare raw output vs processed output
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

A typical run bundle contains files such as:

- `query.txt`
- `answer.txt`
- `speech_text.txt`
- `response.json`
- `config_snapshot.yaml`
- copied audio artifacts

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

### XTTS standalone speak with inline text

```bash
bash scripts/speak_text_xtts.sh --text "Hello from Brooks." --preset clean_modern
```

### XTTS standalone speak from a text file

```bash
bash scripts/speak_text_xtts.sh --text-file /path/to/input.txt --preset lecture_warm
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

## Handoff notes for another coder

This repo is now in good shape as a **technical handoff document**, but it is **not fully clone-and-run portable** without local edits.

Another coder should be able to understand the architecture and workflow from this README, but they will still need to adapt:

- absolute paths in `config/local_config*.yaml`
- the XTTS virtual environment name and activation path
- local model and speaker-reference file locations
- any private or unpublished corpus assets
- OpenAI credentials and environment setup

If you want a truly portable handoff, the next cleanup step is to replace machine-specific absolute paths with environment variables or a checked-in example config.

### Recommended portability cleanup

1. add `config/local_config.example.yaml`
2. move machine-specific paths into environment variables or ignored local config files
3. document exact dependency install commands for both environments
4. document required local assets explicitly
5. rename any misleading artifact filenames if needed

---

## Final note

At this point, the project has crossed the line from “interesting experiment” into “working local system.”

The remaining work is mostly refinement:

- cleaner config portability
- smoother sentence transitions
- clearer environment setup instructions
- better separation between personal local paths and repo defaults
- continued evaluation of raw voice quality vs stylistic post-processing

That is a good place to be.
