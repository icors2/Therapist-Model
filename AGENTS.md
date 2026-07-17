# AGENTS.md

## Cursor Cloud specific instructions

This repo is an **ML fine-tuning pipeline** for an "AI therapist" QLoRA LoRA adapter. It has
**no web app, backend, database, or long-running service** — it is a batch/notebook pipeline.
See `colab_training_guide.md`, `Models and datasets.md`, and `ai_fine_tune.md` for the full
strategy.

### Two stages
1. **Data prep** (`scripts/prepare_datasets.py`, also `prepare_training_data.ipynb`) — downloads
   Hugging Face psychology datasets, filters crisis/empty rows, formats with the Qwen chat
   template, and writes JSON. **Runs on CPU.** This is the only stage runnable in the Cloud VM.
2. **QLoRA training** (`ai_therapist_qlora_training.ipynb`) — 4-bit training with
   `bitsandbytes`. **Requires a CUDA GPU**, which the Cloud VM does not have, so this stage
   cannot be run/verified here. Training deps are installed inline in the notebook, not in a
   requirements file.

### Running the data-prep stage
The update script installs `scripts/requirements-prepare.txt`. Then:
```bash
python3 scripts/prepare_datasets.py --project-dir ./data/processed --skip-articles
```
Gotchas:
- `--skip-articles` avoids ~923 slow calls to the `psychologieetserenite.com` article API
  (~10–15 min); use it for quick runs. Omit it only when you specifically need that data source.
- `--max-empathetic N` caps the Empathetic dataset rows for a fast smoke run.
- The `[transformers] PyTorch was not found` warning is **expected and harmless** — data prep
  only uses the tokenizer (`AutoTokenizer`), not torch. Do not install torch just for this stage.
- Downloads require Hugging Face Hub network access. The datasets used are public, so no HF token
  is needed for data prep (a token is only needed for the GPU training notebook).
- Output goes to `data/processed/` (git-ignored): `clinical_synthetic_data.json`,
  `intima_eval.json`, `dataset_stats.json`.

### Lint / test
There is no linter config, no test suite, and no CI in this repo. Use
`python3 -m py_compile scripts/prepare_datasets.py` as a basic syntax check.
