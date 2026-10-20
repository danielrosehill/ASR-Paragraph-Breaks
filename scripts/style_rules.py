"""Rules 1-3 of docs/paragraphing-style.md as code: refine a proposal toward Daniel's paragraphing.

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


def apply(proposed: str, lead: int = LEAD_MAX_WORDS, close: int = CLOSE_MAX_WORDS) -> list[str]:
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
    tp = npred = nref = 0
    for f in glob.glob(f"{ann_dir}/syn-*.jsonl"):
        r = json.loads(open(f).read())
        pred, ref = _offsets(apply(r["proposed"])), _offsets(r["paragraphs"])
        tp, npred, nref = tp + len(pred & ref), npred + len(pred), nref + len(ref)
    p, rc = tp / npred, tp / nref
    print(f"precision {p:.2f} recall {rc:.2f} F1 {2 * p * rc / (p + rc):.2f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        score(sys.argv[2])
    else:
        sys.exit(__doc__)
