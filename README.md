# Alan Watts Local TTS Runner - Stage 2

This package extends the local RAG + TTS runner with the next practical layer:

- reproducible run bundles
- preset-based tuning
- persistent local run logging
- one-command local text or text+speech generation

## What is new in Stage 2

Each run can now save a bundle under `outputs/runs/...` containing:

- `query.txt`
- `answer.txt`
- `speech_text.txt` (when TTS is used)
- `response.json`
- `config_snapshot.yaml`
- `raw.wav` (when TTS is used)
- `vintage.wav` (when TTS is used)

A persistent history file is also appended at:

- `outputs/logs/history.jsonl`

## Quick start

```bash
bash scripts/setup_local.sh
export OPENAI_API_KEY="your-key-here"
cp /path/to/watts_rag_corpus_clean.txt data/
```

Text only:

```bash
bash scripts/ask_local.sh "What is the self?"
```

Text + speech:

```bash
bash scripts/ask_and_speak.sh "What is the self?"
```

With a preset:

```bash
python -m alan_watts_local.cli --config config/local_config.yaml list-presets
bash scripts/ask_and_speak.sh "What is the self?" --preset lecture_warm
bash scripts/ask_and_speak.sh "What is the self?" --preset clean_modern
```

JSON output:

```bash
bash scripts/ask_and_speak.sh "What is the self?" --preset lecture_warm --json
```

Skip saving a run bundle:

```bash
bash scripts/ask_and_speak.sh "What is the self?" --no-save-run
```

## Notes

- Retrieval still uses the saved local FAISS index.
- Query embeddings still cost API money.
- Answer generation still costs API money.
- TTS adds one more API call.
- Speech rewrite is local and cheap.

## Recommended first experiments

- Compare `lecture_warm` vs `deeper_slower`
- Compare `lecture_warm` vs `clean_modern`
- Inspect `speech_text.txt` for each run
- Listen to `raw.wav` vs `vintage.wav`
- Review `outputs/logs/history.jsonl` to track what settings you liked
