"""Rules 1-3 of docs/paragraphing-style.md as code (v1.2): refine a proposal toward Daniel's paragraphing.

  python3 scripts/style_rules.py score <annotations_dir> [--v1.1]   # P/R/F1 vs gold annotations/*.jsonl

apply(proposed) only ever adds breaks, at sentence boundaries or (v1.2, rules 1b/3c) after a greeting
comma or before a signature name, so the output differs from the input in whitespace only. Thresholds
were fitted 2026-10-21 on 19 gold items (see the doc); v1.2 was added 2026-10-24 (r5).
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


# Rule 1b (v1.2): a greeting fused by a comma into a longer first sentence ("Hi John, just following up
# on ...") is split after the comma. Not when the rest is a self-introduction ("Hi Priya, it's Daniel.",
# "Hey Marcus, it's me, sorry ..."): Daniel kept those together in every review.
GREET = re.compile(r"^((?:Hi|Hey|Hello|Dear|Morning|Good morning)(?: [A-Z][\w.'-]+){0,3},)\s+(\S.*)$", re.S)
INTRO = re.compile(r"^(it's|it is|this is|i'm|i am|me\b|my name)", re.I)
GREET_MIN_REST = 6  # words after the greeting comma, within the same sentence
# Rule 3c (v1.2): a valediction and a name at the very end ("..., all the best, Daniel.", "Kind regards,
# Priya Raman") put the name on its own line, as in Daniel's reviews ("Kind regards," / "Priya Raman").
# "Thanks, Dana." addresses the listener and stays together, so thanks/thank you are not valedictions.
SIGN = re.compile(r"(?<![\w'])((?:all the best|best wishes|best regards|kind regards|warm regards|regards|best|cheers|"
                  r"talk soon|speak soon|take care|love),)\s+([A-Z][\w'-]+(?: [A-Z][\w'-]+){0,2}\.?)$", re.I)


def apply(proposed: str, lead: int = LEAD_MAX_WORDS, close: int = CLOSE_MAX_WORDS, v12: bool = True) -> list[str]:
    out = _v1(proposed, lead, close)
    start = next((i for i in range(max(0, len(out) - WRAP_WINDOW), len(out)) if WRAP.match(out[i])), None)
    if start is not None:
        out = out[:start] + [x for p in out[start:] for x in sentences(p)]
    if v12 and out:
        first = sentences(out[0])
        m = GREET.match(first[0]) if first else None
        if m and not INTRO.match(m.group(2)) and len(m.group(2).split()) >= GREET_MIN_REST:
            out = [m.group(1), out[0][len(m.group(1)):].strip()] + out[1:]
        m = SIGN.search(out[-1])
        if m and m.group(2)[0].isupper() and m.start(2) > 0 and out[-1][:m.start(2)].strip():
            out = out[:-1] + [out[-1][:m.start(2)].strip(), m.group(2)]
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


def score(ann_dir: str, v12: bool = True) -> None:
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
        pred, ref = _offsets(apply(haiku.get(r["id"], r["proposed"]), v12=v12)), _offsets(r["paragraphs"])
        for key in (r["id"].rsplit("-", 1)[0], "all"):
            t = by_set.setdefault(key, [0, 0, 0])
            t[0] += len(pred & ref); t[1] += len(pred); t[2] += len(ref)
    for key, (tp, npred, nref) in sorted(by_set.items()):
        p, rc = tp / npred, tp / nref
        print(f"{key:6s} precision {p:.2f} recall {rc:.2f} F1 {2 * p * rc / (p + rc):.2f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        score(sys.argv[2], v12="--v1.1" not in sys.argv)
    else:
        sys.exit(__doc__)
