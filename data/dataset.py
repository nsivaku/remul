import os
import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from utils.prompts import format_prompt


def build_chat_prompt(question_text: str, system_msg: str = "") -> list:
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": question_text})
    return messages

def load_and_process_bbh(
    tasks: List[str],
    max_samples: int = 1250,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    all_rows = []

    for task in tasks:
        try:
            ds = load_dataset("maveriq/bigbenchhard", task, split="train")
        except Exception:
            try:
                ds = load_dataset("lukaemon/bbh", task, split="test")
            except Exception:
                print(f"Warning: Could not load BBH task '{task}', skipping.")
                continue

        for item in ds:
            input_text = item.get("input", item.get("question", ""))
            target = item.get("target", item.get("answer", ""))

            prompt_text = format_prompt("bbh", {
                "question": input_text,
                "options": _extract_options(input_text),
            })

            all_rows.append({
                "prompt": json.dumps(build_chat_prompt(prompt_text)),
                "data_source": f"bbh_{task}",
                "ground_truth": str(target),
                "extra_info": json.dumps({"task": task}),
            })

    random.seed(seed)
    if len(all_rows) > max_samples:
        all_rows = random.sample(all_rows, max_samples)

    random.shuffle(all_rows)
    split_idx = int(len(all_rows) * (1 - val_fraction))
    return pd.DataFrame(all_rows[:split_idx]), pd.DataFrame(all_rows[split_idx:])

def prepare_data(output_dir: str = "./data/parquet", seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        "logical_deduction_five_objects",
        "navigate",
        "temporal_sequences",
        "sports_understanding",
        "tracking_shuffled_objects_five_objects",
    ]

    train_df, val_bbh_df = load_and_process_bbh(tasks, max_samples=1250, val_fraction=0.2, seed=seed)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    val_bbh_path = os.path.join(output_dir, "val_bbh.parquet")
    val_bbh_df.to_parquet(val_bbh_path)

def _extract_options(text: str) -> str:
    lines = text.strip().split("\n")
    opts = []
    for line in lines:
        s = line.strip()
        if s and (s.startswith("(") or s.startswith("Option") or
                  (len(s) > 2 and s[0] in "ABCDEF" and s[1] in ":).")):
            opts.append(s)
    return "\n".join(opts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare parquet data for verl")
    parser.add_argument("--output_dir", default="./data/parquet")
    args = parser.parse_args()
    prepare_data(args.output_dir, args.seed)