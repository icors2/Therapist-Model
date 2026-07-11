#!/usr/bin/env python3
"""Download, convert, filter, and merge HF psychology datasets for Qwen SFT."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a supportive, non-clinical listening assistant. "
    "You do not diagnose, prescribe medication, or handle mental health crises. "
    "You use active listening and gentle Socratic questions. "
    "Encourage professional help when concerns are serious or persistent."
)

CRISIS_PATTERNS = re.compile(
    r"\b("
    r"kill myself|killing myself|end my life|suicide|suicidal|"
    r"self[- ]harm|cut myself|want to die|going to die|"
    r"overdose|988|911 emergency"
    r")\b",
    re.IGNORECASE,
)

ARTICLE_API_BASE = "https://psychologieetserenite.com/api/v1/articles"


def contains_crisis(text: str) -> bool:
    return bool(text and CRISIS_PATTERNS.search(text))


def format_qwen(
    tokenizer: AutoTokenizer,
    user: str,
    assistant: str,
    system: str = SYSTEM_PROMPT,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_empathetic(
    tokenizer: AutoTokenizer,
    max_rows: int,
    stats: dict[str, Any],
) -> list[dict[str, str]]:
    ds = load_dataset("LuangMV97/Empathetic_counseling_Dataset", split="train")
    rows: list[dict[str, str]] = []
    kept = dropped_crisis = dropped_empty = 0

    for i, example in enumerate(ds):
        if i >= max_rows:
            break
        if i > 0 and i % 1000 == 0:
            print(f"  Empathetic: processed {i}/{max_rows}...", flush=True)
        user = (example.get("input") or "").strip()
        assistant = (example.get("label") or "").strip()
        if not user or not assistant:
            dropped_empty += 1
            continue
        if contains_crisis(user) or contains_crisis(assistant):
            dropped_crisis += 1
            continue
        rows.append({"text": format_qwen(tokenizer, user, assistant)})
        kept += 1

    stats["empathetic"] = {
        "kept": kept,
        "dropped_empty": dropped_empty,
        "dropped_crisis": dropped_crisis,
        "max_rows": max_rows,
    }
    return rows


def convert_graph2counsel(
    tokenizer: AutoTokenizer,
    stats: dict[str, Any],
) -> list[dict[str, str]]:
    ds = load_dataset("UKPLab/Graph2Counsel", split="train")
    rows: list[dict[str, str]] = []
    kept = dropped_crisis = dropped_empty = 0

    for session_idx, example in enumerate(ds):
        if session_idx > 0 and session_idx % 100 == 0:
            print(f"  Graph2Counsel: session {session_idx}/{len(ds)}...", flush=True)
        dialog = example.get("dialog") or []
        for turn_idx, turn in enumerate(dialog):
            speaker = (turn.get("speaker") or "").lower()
            message = (turn.get("message") or "").strip()
            if speaker != "counselor" or not message:
                continue
            client_message = ""
            for prev in reversed(dialog[:turn_idx]):
                if (prev.get("speaker") or "").lower() == "client":
                    client_message = (prev.get("message") or "").strip()
                    break
            if not client_message:
                dropped_empty += 1
                continue
            if contains_crisis(client_message) or contains_crisis(message):
                dropped_crisis += 1
                continue
            rows.append({"text": format_qwen(tokenizer, client_message, message)})
            kept += 1

    stats["graph2counsel"] = {
        "kept": kept,
        "dropped_empty": dropped_empty,
        "dropped_crisis": dropped_crisis,
    }
    return rows


def fetch_article_body(slug: str, lang: str = "en") -> str:
    url = f"{ARTICLE_API_BASE}/{slug}"
    response = requests.get(url, params={"lang": lang}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    article = payload.get("article") or {}
    content = (
        article.get("content_markdown")
        or article.get("content")
        or article.get("description")
        or ""
    )
    return strip_markdown(content)


def convert_articles(
    tokenizer: AutoTokenizer,
    max_chars: int,
    rate_limit: float,
    stats: dict[str, Any],
) -> list[dict[str, str]]:
    metadata = load_dataset("psychologie-et-serenite/articles-metadata", split="train")
    rows: list[dict[str, str]] = []
    kept = dropped_empty = dropped_crisis = fetch_errors = 0
    total = len(metadata)

    for article_idx, example in enumerate(metadata):
        if article_idx > 0 and article_idx % 50 == 0:
            print(
                f"  Articles: {article_idx}/{total} fetched, {kept} kept...",
                flush=True,
            )
        slug = (example.get("slug_en") or example.get("slug_fr") or "").strip()
        title = (example.get("title_en") or example.get("title_fr") or "Psychology topic").strip()
        theme = (example.get("theme_en") or example.get("theme_fr") or "general").strip()
        if not slug:
            dropped_empty += 1
            continue

        try:
            body = fetch_article_body(slug, lang="en")
            time.sleep(rate_limit)
        except requests.RequestException:
            fetch_errors += 1
            continue

        if not body:
            dropped_empty += 1
            continue

        assistant = body[:max_chars]
        user = f"Explain: {title} (theme: {theme})"
        if contains_crisis(user) or contains_crisis(assistant):
            dropped_crisis += 1
            continue

        rows.append({"text": format_qwen(tokenizer, user, assistant)})
        kept += 1

    stats["articles"] = {
        "kept": kept,
        "dropped_empty": dropped_empty,
        "dropped_crisis": dropped_crisis,
        "fetch_errors": fetch_errors,
    }
    return rows


def convert_intima(stats: dict[str, Any]) -> list[dict[str, str]]:
    ds = load_dataset("AI-companionship/INTIMA", split="train")
    rows: list[dict[str, str]] = []
    for example in ds:
        prompt = (example.get("prompt") or "").strip()
        if not prompt:
            continue
        rows.append(
            {
                "prompt": prompt,
                "code": example.get("code", ""),
                "model": example.get("model", ""),
            }
        )
    stats["intima_eval"] = {"rows": len(rows)}
    return rows


def prepare_all(
    project_dir: str | Path,
    model_id: str = "Qwen/Qwen2.5-0.5B",
    max_empathetic: int = 10_000,
    article_max_chars: int = 1500,
    article_rate_limit: float = 0.5,
    skip_articles: bool = False,
) -> dict[str, Any]:
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    data_path = project_dir / "clinical_synthetic_data.json"
    eval_path = project_dir / "intima_eval.json"
    stats_path = project_dir / "dataset_stats.json"

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    stats: dict[str, Any] = {"model_id": model_id}
    all_rows: list[dict[str, str]] = []

    print("Converting Empathetic_counseling_Dataset...")
    all_rows.extend(convert_empathetic(tokenizer, max_empathetic, stats))

    print("Converting Graph2Counsel...")
    all_rows.extend(convert_graph2counsel(tokenizer, stats))

    if skip_articles:
        stats["articles"] = {"skipped": True}
        print("Skipping psychologie articles (skip_articles=True).")
    else:
        print("Fetching psychologie-et-serenite articles via API...")
        all_rows.extend(
            convert_articles(tokenizer, article_max_chars, article_rate_limit, stats)
        )

    print("Building INTIMA eval set...")
    intima_rows = convert_intima(stats)

    stats["total_sft_rows"] = len(all_rows)

    with data_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(intima_rows, f, ensure_ascii=False, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_rows)} SFT rows -> {data_path}")
    print(f"Wrote {len(intima_rows)} eval rows -> {eval_path}")
    print(f"Stats -> {stats_path}")
    print(json.dumps(stats, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare psychology SFT datasets.")
    parser.add_argument(
        "--project-dir",
        default="./data/processed",
        help="Output directory for merged JSON files.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B",
        help="Tokenizer model for Qwen chat template.",
    )
    parser.add_argument(
        "--max-empathetic",
        type=int,
        default=10_000,
        help="Max Empathetic_counseling rows to include.",
    )
    parser.add_argument(
        "--article-max-chars",
        type=int,
        default=1500,
        help="Truncate article bodies to this length.",
    )
    parser.add_argument(
        "--article-rate-limit",
        type=float,
        default=0.5,
        help="Seconds to wait between article API requests.",
    )
    parser.add_argument(
        "--skip-articles",
        action="store_true",
        help="Skip psychologie API article fetching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_all(
        project_dir=args.project_dir,
        model_id=args.model_id,
        max_empathetic=args.max_empathetic,
        article_max_chars=args.article_max_chars,
        article_rate_limit=args.article_rate_limit,
        skip_articles=args.skip_articles,
    )


if __name__ == "__main__":
    main()
