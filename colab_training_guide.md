# Google Colab Training Guide — AI Therapist LoRA Fine-Tune

**Notebooks (run in order):**

1. [`prepare_training_data.ipynb`](./prepare_training_data.ipynb) — download HF datasets, format to Qwen chat JSON
2. [`ai_therapist_qlora_training.ipynb`](./ai_therapist_qlora_training.ipynb) — QLoRA fine-tune on merged data

See [`Models and datasets.md`](./Models%20and%20datasets.md) for source details.

This guide implements the strategy in [`ai_fine_tune.md`](./ai_fine_tune.md): PEFT/LoRA on a psychology-tuned base model, with safety architecture kept *outside* the model weights.

> **Important:** Fine-tuning teaches communication style, not clinical competence. Crisis routing, PII scrubbing, and RAG must live in your application layer.

---

## What You Will Train

| Item | Choice |
|------|--------|
| Base model | `Qwen/Qwen2.5-0.5B` |
| Starting adapter | `phxdev/psychology-qwen-0.5b` (merged before new LoRA) |
| Method | QLoRA (4-bit + new LoRA adapters) |
| Trainer | Hugging Face `SFTTrainer` |
| Output | LoRA adapter weights (~10–50 MB) |

---

## Prerequisites

1. **Google account** — [Google Colab](https://colab.research.google.com/)
2. **Hugging Face account** — [huggingface.co/join](https://huggingface.co/join)
3. **HF access token** — [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. **Repo files** — push this repo to GitHub; set `REPO_URL` in `prepare_training_data.ipynb` (or Colab Secret `THERAPIST_REPO_URL`)

### Colab GPU

| Tier | GPU | 0.5B QLoRA? |
|------|-----|-------------|
| Free | T4 (16 GB) | Yes, comfortably |
| Colab Pro | L4 / A100 | Faster |

**Data prep:** CPU runtime is fine. **Training:** GPU required.

---

## Quick Start Checklist

- [ ] Run `prepare_training_data.ipynb` (CPU) → writes `clinical_synthetic_data.json` to Drive
- [ ] Switch to GPU runtime for training notebook
- [ ] Log in to Hugging Face
- [ ] Run `ai_therapist_qlora_training.ipynb` with `USE_SAMPLE_DATA = False`
- [ ] Smoke-test inference
- [ ] Evaluate on `intima_eval.json` before any deployment

---

## Phase 1 — Prepare Data (`prepare_training_data.ipynb`)

### What it does

1. Downloads three **SFT** sources from Hugging Face:
   - `LuangMV97/Empathetic_counseling_Dataset` (input/label pairs)
   - `UKPLab/Graph2Counsel` (multi-turn dialog → turn pairs)
   - `psychologie-et-serenite/articles-metadata` + article bodies via public API
2. Filters crisis keywords and empty rows
3. Formats all rows with Qwen `apply_chat_template`
4. Writes **eval-only** `intima_eval.json` from `AI-companionship/INTIMA`

### Output (Drive: `MyDrive/ai-therapist/`)

```
clinical_synthetic_data.json   # [{"text": "..."}]  ~10k+ rows
intima_eval.json               # [{"prompt", "code", "model"}]
dataset_stats.json             # counts per source
```

### Local alternative

```bash
pip install datasets transformers requests
python scripts/prepare_datasets.py --project-dir ./data/processed
```

Flags: `--max-empathetic 10000`, `--skip-articles`, `--article-rate-limit 0.5`

Article fetching takes ~10–15 min (923 API calls @ 0.5s each). Set `SKIP_ARTICLES = True` in the notebook to skip.

### Auto-clone repo in Colab

In `prepare_training_data.ipynb`, set:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/Therapist-model.git"
```

Or add a Colab Secret named `THERAPIST_REPO_URL`. The notebook runs `git clone` into `/content/therapist-model` and imports `scripts/prepare_datasets.py`.

For **private repos**, use a personal access token in the URL or secret:

```
https://<GITHUB_TOKEN>@github.com/you/Therapist-model.git
```

If the repo is already on Drive, the notebook will use that copy automatically without cloning.

---

## Phase 2 — Train (`ai_therapist_qlora_training.ipynb`)

### Model loading strategy

1. Load `Qwen/Qwen2.5-0.5B` in 4-bit
2. Load `phxdev/psychology-qwen-0.5b` adapter and **merge**
3. Apply a **new** LoRA for dataset-specific fine-tuning

### Default hyperparameters (0.5B on T4)

| Parameter | Value |
|-----------|-------|
| `MAX_SEQ_LENGTH` | 1024 |
| `BATCH_SIZE` | 4 |
| `GRAD_ACCUM` | 4 |
| `MAX_STEPS` | 300 |
| `LEARNING_RATE` | 2e-4 |

Training time: ~15–45 min on T4 for 300 steps with ~10k examples.

### Dataset format

Each row in `clinical_synthetic_data.json`:

```json
{"text": "<Qwen chat template string with system/user/assistant roles>"}
```

Built by `scripts/prepare_datasets.py` — do not hand-edit unless adding curated examples.

---

## Recommended Drive Layout

```
MyDrive/ai-therapist/
├── clinical_synthetic_data.json
├── intima_eval.json
├── dataset_stats.json
├── ai-therapist-lora/           # checkpoints
└── ai-therapist-lora-final/     # final adapter
```

---

## After SFT

1. **Evaluate** on `intima_eval.json` (companionship / boundary behaviors)
2. **DPO alignment** — clinician preference pairs
3. **Red team** before user-facing deployment
4. **Deploy** with crisis routing, PII scrubbing, RAG

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `clinical_synthetic_data.json` not found | Run `prepare_training_data.ipynb` first |
| Article API errors | Set `SKIP_ARTICLES = True`; retry later |
| `CUDA out of memory` | Lower `BATCH_SIZE` to 2 or `MAX_SEQ_LENGTH` to 512 |
| `bitsandbytes` errors | Re-run pip install; restart runtime |
| Loss is `nan` | Lower learning rate; check for corrupt rows |
| Colab disconnects | Save to Drive; reduce `save_steps` |

---

## Legal & Ethical Reminders

- Do not train on HIPAA-protected transcripts without authorization
- This model is not a licensed therapist — disclose to users
- Never rely on the LLM alone for crisis detection — use deterministic routing
- INTIMA is for evaluation, not training (prompts lack safe assistant labels)

---

## Related Reading

- [`Models and datasets.md`](./Models%20and%20datasets.md) — dataset roles and run order
- [`ai_fine_tune.md`](./ai_fine_tune.md) — architecture, safety, DPO
- [phxdev/psychology-qwen-0.5b](https://huggingface.co/phxdev/psychology-qwen-0.5b)
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
