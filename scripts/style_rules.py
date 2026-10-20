"""Rules 1-3 of docs/paragraphing-style.md as code (v1.1): refine a proposal toward Daniel's paragraphing.

  python3 scripts/style_rules.py score <annotations_dir>   # P/R/F1 vs gold annotations/*.jsonl

apply(proposed) only ever adds breaks at sentence boundaries, so the output differs from the input
in whitespace only. Thresholds were fitted 2026-10-21 on 19 gold items (see the doc).
"""

from __future__ import annotations

import glob
import json
import re
import sys

SENT = re.compile(r"(?<=[.!?])\s+")
# Titles and abbreviations whose full stop does not end a sentence ("Mr. Delgado", "St. Mary's").
ABBREV = re.compile(r"(?:^|\s)(?:Mr|Mrs|Ms|Dr|Prof|St|Mt|Jr|Sr|vs|etc|e\.g|i\.e|approx|No)\.$", re.I)


def sentences(p: str) -> list[str]:
    out: list[str] = []
    for piece in SENT.split(p.strip()):
        if out and ABBREV.search(out[-1]):
            out[-1] += " " + piece
        elif piece:
            out.append(piece)
    return out
LEAD_MAX_WORDS = 11   # rule 2: a paragraph-opening sentence this short is a signpost
CLOSE_MAX_WORDS = 10  # rule 3: trailing sentences this short are closing lines


# Rule 3b (v1.1): from a wrap-up line near the end, every following sentence stands alone, however long.
WRAP = re.compile(r"^(okay,? |anyway,? |right,? |so,? )*(that's (it|all|everything|the (plan|core|decision|whole))"
                  r"|i think that's|thanks|thank you|okay, that's)", re.I)
WRAP_WINDOW = 3  # only the last three paragraphs are searched for a wrap-up line


def apply(proposed: str, lead: int = LEAD_MAX_WORDS, close: int = CLOSE_MAX_WORDS) -> list[str]:
    out = _v1(proposed, lead, close)
    start = next((i for i in range(max(0, len(out) - WRAP_WINDOW), len(out)) if WRAP.match(out[i])), None)
    if start is not None:
        out = out[:start] + [x for p in out[start:] for x in sentences(p)]
    return out


def _v1(proposed: str, lead: int, close: int) -> list[str]:
    paras = [p for p in proposed.split("\n\n") if p.strip()]
    out: list[str] = []
    for i, p in enumerate(paras):
        s = sentences(p)
        if i == len(paras) - 1:
            tail = []
            while len(s) > 1 and len(s[-1].split()) <= close:
                tail.insert(0, s.pop())
            if i == 0 and len(s) > 1:  # one-paragraph memo: the opener still stands alone
                out += [s[0], " ".join(s[1:])]
            else:
                out.append(" ".join(s))
            out += tail
        elif len(s) > 1 and (i == 0 or len(s[0].split()) <= lead):
            out += [s[0], " ".join(s[1:])]
        else:
            out.append(p.strip())
    return out


def _offsets(paras: list[str]) -> set[int]:
    out, n = set(), 0
    for p in paras[:-1]:
        n += len(re.sub(r"\s+", "", p))
        out.add(n)
    return out


def score(ann_dir: str) -> None:
    """Score against gold, starting from Haiku's own proposal (data/synthetic/), not the one the
    annotator saw: for items reviewed after v1 shipped, the saved `proposed` already has rules applied."""
    from pathlib import Path
    haiku = {}
    for f in (Path(__file__).resolve().parent.parent / "data" / "synthetic").glob("synthetic-*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            haiku[r["id"]] = r["proposed"]
    by_set: dict[str, list[int]] = {}
    for f in glob.glob(f"{ann_dir}/syn-*.jsonl"):
        r = json.loads(open(f).read())
        pred, ref = _offsets(apply(haiku.get(r["id"], r["proposed"]))), _offsets(r["paragraphs"])
        for key in (r["id"].rsplit("-", 1)[0], "all"):
            t = by_set.setdefault(key, [0, 0, 0])
            t[0] += len(pred & ref); t[1] += len(pred); t[2] += len(ref)
    for key, (tp, npred, nref) in sorted(by_set.items()):
        p, rc = tp / npred, tp / nref
        print(f"{key:6s} precision {p:.2f} recall {rc:.2f} F1 {2 * p * rc / (p + rc):.2f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        score(sys.argv[2])
    else:
        sys.exit(__doc__)
