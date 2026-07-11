We will be using [phxdev/psychology-qwen-0.5b](https://huggingface.co/phxdev/psychology-qwen-0.5b) as the base model.

- **Base:** `Qwen/Qwen2.5-0.5B`
- **Starting adapter:** `phxdev/psychology-qwen-0.5b` (psychology LoRA, merged before new training)
- **Method:** QLoRA (4-bit + new LoRA adapters on top)

---

## Datasets

| Source | HF ID | Role | Notes |
|--------|-------|------|-------|
| Empathetic Counseling | [LuangMV97/Empathetic_counseling_Dataset](https://huggingface.co/datasets/LuangMV97/Empathetic_counseling_Dataset) | **SFT train** | ~38k `input`/`label` pairs; train split only; subsampled to 10k by default |
| Graph2Counsel | [UKPLab/Graph2Counsel](https://huggingface.co/datasets/UKPLab/Graph2Counsel) | **SFT train** | Multi-turn `dialog` → client/counselor turn pairs (~760 sessions) |
| Psychologie et Sérénité | [psychologie-et-serenite/articles-metadata](https://huggingface.co/datasets/psychologie-et-serenite/articles-metadata) | **SFT train** | Metadata on HF; article bodies fetched via public API (`/api/v1/articles/{slug}?lang=en`) |
| INTIMA | [AI-companionship/INTIMA](https://huggingface.co/datasets/AI-companionship/INTIMA) | **Eval only** | Prompts without assistant labels — not used for SFT |

Language: **English only** for v1 (matches Empathetic, Graph2Counsel, INTIMA).

---

## Run order (Colab)

1. **`prepare_training_data.ipynb`** — download, convert, filter, write JSON to Drive
2. **`ai_therapist_qlora_training.ipynb`** — QLoRA fine-tune on merged data

Local alternative:

```bash
pip install datasets transformers requests
python scripts/prepare_datasets.py --project-dir ./data/processed
```

---

## Output files (Drive: `MyDrive/ai-therapist/`)

| File | Purpose |
|------|---------|
| `clinical_synthetic_data.json` | Merged SFT rows: `[{"text": "..."}]` in Qwen chat format |
| `intima_eval.json` | INTIMA eval prompts: `[{"prompt", "code", "model"}]` |
| `dataset_stats.json` | Row counts per source after filtering |
| `ai-therapist-lora-final/` | Trained LoRA adapter weights |

---

## Filtering

All training rows pass through:

- Empty text drop
- Crisis keyword filter (self-harm / active crisis language dropped per `ai_fine_tune.md`)
- Qwen `apply_chat_template` formatting with shared non-clinical system prompt

---

## Links

- [psychologie-et-serenite org](https://huggingface.co/psychologie-et-serenite)
- [Empathetic_counseling_Dataset](https://huggingface.co/datasets/LuangMV97/Empathetic_counseling_Dataset)
- [Graph2Counsel](https://huggingface.co/datasets/UKPLab/Graph2Counsel)
- [INTIMA](https://huggingface.co/datasets/AI-companionship/INTIMA)
