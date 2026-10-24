"""Build training/train.jsonl and training/test.jsonl (r1-r3: Gemma; r5: encoder classifier).

  python3 scripts/build_training.py            # style_rules v1.2 silver + ASR-noise copies (r5 set)
  python3 scripts/build_training.py --no-aug   # without the noise copies

The r3 train set (804 rows, style_rules v1.1, no augmentation) is training/train.jsonl at commit fb78a6e.

Test: every hand-reviewed golden item (`syn-2-*`), held out because the rules were not fitted on them.
Train: the other hand-reviewed items (gold) plus every unreviewed synthetic item labelled by
scripts/style_rules.py on Haiku's proposal (silver). Each row: {id, label, input, output}, where
input is the flat text and output the same text with paragraphs separated by blank lines.
Gold is read live from the HF dataset's annotations/, so rerun after more reviews.

ASR-noise augmentation (r5, 2026-10-24; seeded): for AUG_FRAC of train rows, a copy (id `<id>-aug`,
label `<label>-aug`) whose text imitates ASR output: the word after 1-2 sentence-final periods is
lowercased, 1-3 commas are dropped, fillers ("um", "so", "like") are inserted and "um"/"uh" removed;
a quarter of the copies also get a fused greeting and/or an inline valediction + name (fuse()).
Noise is applied per paragraph, so the target keeps the same breaks and still differs from the input in
whitespace only. Test is never augmented. A copy shares its source's id prefix, so a dev split carved
by id keeps both on the same side.
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from pathlib import Path

from style_rules import apply as style_rules

ROOT = Path(__file__).resolve().parent.parent
DATASET = "danielrosehill/ASR-Paragraph-Breaks"
OUT = ROOT / "training"
AUG_FRAC = 0.3
FILLERS = ("um", "so", "like")
FUSE_FRAC = 0.25  # of noisy copies: also fuse a greeting / a valediction + name (rules 1b and 3c)
NAMES = ("John", "Sarah", "Marcus", "Priya", "Dana", "Tom", "Rachel", "Ben", "Hannah", "Mike", "Aunt Carol", "Mr. Abrams")
GREETINGS = ("Hi", "Hey", "Hello", "Dear", "Morning")
VALEDICTIONS = ("all the best", "best", "cheers", "kind regards", "talk soon", "speak soon", "take care", "regards")
GREETING_START = re.compile(r"^(hi|hey|hello|dear|morning|good morning)\b", re.I)


def fuse(paras: list[str], rng: random.Random) -> list[str]:
    """Rule 1b/3c examples (r5b): "Hi Name," + the old opener with its first letter lowercased, as two
    paragraphs ("Hi John," / "just following up ..."), and the last sentence ending in ", <valediction>,"
    followed by the name as its own paragraph ("Great to hear from you, all the best," / "Daniel.")."""
    paras = list(paras)
    if rng.random() < 0.5 and not GREETING_START.match(paras[0]):
        first = paras[0]
        paras = [f"{rng.choice(GREETINGS)} {rng.choice(NAMES)},", first[:1].lower() + first[1:]] + paras[1:]
    if paras[-1].endswith(".") and (rng.random() < 0.5 or len(paras) < 2):
        paras = paras[:-1] + [paras[-1][:-1] + f", {rng.choice(VALEDICTIONS)},", rng.choice(NAMES).split()[-1] + "."]
    return paras


def noisy(paras: list[str], rng: random.Random) -> list[str]:
    """Apply ASR-style noise to paragraphs; returns new paragraphs (same count, so same breaks)."""
    words = [w for p in paras for w in p.split()]
    para_of = [i for i, p in enumerate(paras) for _ in p.split()]
    ends = [i for i, w in enumerate(words[:-1]) if w.endswith(".")]
    for i in rng.sample(ends, min(len(ends), rng.randint(1, 2))):
        words[i + 1] = words[i + 1].lower()
    commas = [i for i, w in enumerate(words) if w.endswith(",")]
    for i in rng.sample(commas, min(len(commas), rng.randint(1, 3))):
        words[i] = words[i][:-1]
    keep = [i for i, w in enumerate(words) if not (w.lower().strip(",") in ("um", "uh") and rng.random() < 0.5)]
    if keep and len(keep) < len(words) and all(para_of[k] in {para_of[j] for j in keep} for k in range(len(words))):
        words, para_of = [words[i] for i in keep], [para_of[i] for i in keep]
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(len(words))
        words.insert(i, rng.choice(FILLERS))
        para_of.insert(i, para_of[i])
    out = [[] for _ in paras]
    for w, k in zip(words, para_of):
        out[k].append(w)
    return [" ".join(x) for x in out]


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

    if "--no-aug" not in sys.argv:
        rng = random.Random(1)
        for row in [r for r in train if rng.random() < AUG_FRAC]:
            paras = noisy(row["output"].split("\n\n"), rng)
            if rng.random() < FUSE_FRAC:
                paras = fuse(paras, rng)
            train.append({"id": row["id"] + "-aug", "label": row["label"] + "-aug",
                          "input": " ".join(paras), "output": "\n\n".join(paras)})

    for row in train + test:
        assert squash(row["input"]) == squash(row["output"]), row["id"]
    random.Random(0).shuffle(train)
    test.sort(key=lambda r: r["id"])
    OUT.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n = {k: sum(r["label"] == k for r in train) for k in ("gold", "silver", "gold-aug", "silver-aug")}
    print(f"train {len(train)} {n}, test {len(test)} gold")


if __name__ == "__main__":
    main()
