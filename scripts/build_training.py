"""Build training/train.jsonl and training/test.jsonl for the r1 fine-tune.

  python3 scripts/build_training.py

Test: every hand-reviewed golden item (`syn-2-*`), held out because the rules were not fitted on them.
Train: the other hand-reviewed items (gold) plus every unreviewed synthetic item labelled by
scripts/style_rules.py on Haiku's proposal (silver). Each row: {id, label, input, output}, where
input is the flat text and output the same text with paragraphs separated by blank lines.
Gold is read live from the HF dataset's annotations/, so rerun after more reviews.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

from style_rules import apply as style_rules

ROOT = Path(__file__).resolve().parent.parent
DATASET = "danielrosehill/ASR-Paragraph-Breaks"
OUT = ROOT / "training"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def squash(t: str) -> str:
    return re.sub(r"\s+", "", t)


def main() -> None:
    tree = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET}/tree/main/annotations"))
    gold = [json.loads(fetch(f"https://huggingface.co/datasets/{DATASET}/resolve/main/{f['path']}"))
            for f in tree if f["path"].endswith(".jsonl")]
    gold_ids = {g["id"] for g in gold}

    train, test = [], []
    for g in gold:
        row = {"id": g["id"], "label": "gold", "input": g["text"], "output": g["paragraphed"]}
        (test if g["id"].startswith("syn-2-") else train).append(row)

    for f in sorted((ROOT / "data" / "synthetic").glob("synthetic-*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["id"] in gold_ids:
                continue
            out = "\n\n".join(style_rules(r["proposed"]))
            train.append({"id": r["id"], "label": "silver", "input": r["text"], "output": out})

    for row in train + test:
        assert squash(row["input"]) == squash(row["output"]), row["id"]
    random.Random(0).shuffle(train)
    test.sort(key=lambda r: r["id"])
    OUT.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_gold = sum(r["label"] == "gold" for r in train)
    print(f"train {len(train)} ({n_gold} gold, {len(train) - n_gold} silver), test {len(test)} gold")


if __name__ == "__main__":
    main()
