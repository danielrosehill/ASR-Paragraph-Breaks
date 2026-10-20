# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic"]
# ///
"""Does Haiku 5.5, given the written rules, beat scripts/style_rules.py at Daniel's paragraphing? (No.)

  uv run scripts/extrapolate.py eval [--folds 4]   # k-fold over the gold reviews: break P/R/F1 by audience

The model gets the rules in docs/paragraphing-style.md, the memo's audience, the other folds' gold
reviews as worked examples and the Haiku proposal, and may only add breaks; the proposal's breaks are
re-imposed and outputs that alter the text are rejected. It is compared against the proposal alone,
the code rules alone, and both combined.

Result 2026-10-21 (19 gold items, 4-fold): code rules F1 0.83, LLM rules 0.73, both 0.71. For notes to
self the two tie (0.84 / 0.83); for memos addressed to someone the LLM over-splits (precision 0.50,
F1 0.66 vs 0.82), because the written rule 4 overstated how far those memos are split. An earlier,
vaguer prompt with no rules did worse (P 0.41). Decision: the code rules produce the proposals;
this script stays as the check to re-run once there is more gold.
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic import api_key  # noqa: E402
from style_rules import apply as code_rules  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "data" / "synthetic"
DATASET = "danielrosehill/ASR-Paragraph-Breaks"
MODEL = "claude-haiku-5-5"

SYSTEM = """You add paragraph breaks to voice-memo transcripts, following one person's rules exactly.

A memo has an opener, a body and a close.
1. The opener stands alone: the first sentence (a greeting, "it's me", or the line that says what the memo is) is its own paragraph.
2. A signpost is split from its detail: when a paragraph begins with a short sentence that announces what comes next ("The second part is the server.", "Here's what happened.", "So the plan is this."), that sentence becomes its own paragraph and the detail starts a new one. A first sentence that already carries real content is not a signpost.
3. Each closing line stands alone: wrap-ups, afterthoughts and sign-offs each get their own paragraph ("That's it.", "Oh, and one more thing...", "Let me know by Friday.", "Thanks, Dana.", "Talk soon,", the name, "Bye.").
4. Audience matters. A memo addressed to another person or organisation (an email, a message, a request) is split further, close to one sentence per paragraph through the body, keeping only tightly linked sentences together. A note to self keeps its multi-sentence body paragraphs as they are.
5. Keep every existing paragraph break.

You may only insert blank lines. Never change, add or remove any word, character or punctuation mark. Output only the transcript, with paragraphs separated by one blank line."""

AUD = {"self": "Audience: a note to self.", "other": "Audience: addressed to another person or organisation."}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def specs_by_id() -> dict[str, dict]:
    out = {}
    for f in SYN.glob("synthetic-*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def gold() -> list[dict]:
    tree = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET}/tree/main/annotations"))
    rows = [json.loads(fetch(f"https://huggingface.co/datasets/{DATASET}/resolve/main/{f['path']}"))
            for f in tree if f["path"].endswith(".jsonl")]
    syn = specs_by_id()
    rows = [dict(r, audience=syn[r["id"]]["spec"]["audience"]) for r in rows if r["source"] == "syn" and r["id"] in syn]
    return sorted(rows, key=lambda r: r["id"])


def squash(t: str) -> str:
    return re.sub(r"\s+", "", t)


def paras(t: str) -> list[str]:
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def breaks(ps: list[str]) -> set[int]:
    out, pos = set(), 0
    for p in ps[:-1]:
        pos += len(squash(p))
        out.add(pos)
    return out


def from_breaks(flat: str, bs: set[int]) -> str:
    out, n = [], 0
    for ch in flat:
        if ch == " " and n in bs:
            out.append("\n\n")
            continue
        out.append(ch)
        if not ch.isspace():
            n += 1
    return re.sub(r" ?\n\n ?", "\n\n", "".join(out))


def user(audience: str, proposed: str) -> str:
    return f"{AUD[audience]}\n\n{proposed}"


def params(examples: list[dict], audience: str, proposed: str) -> dict:
    msgs = []
    for e in examples:
        msgs += [{"role": "user", "content": user(e["audience"], e["proposed"])},
                 {"role": "assistant", "content": e["paragraphed"]}]
    msgs.append({"role": "user", "content": user(audience, proposed)})
    return dict(model=MODEL, max_tokens=8000, system=SYSTEM, messages=msgs, cache_control={"type": "ephemeral"})


def text_of(msg) -> str:
    return next((b.text for b in msg.content if b.type == "text"), "")


def finish(item_text: str, proposed: str, out_text: str) -> str | None:
    if squash(out_text) != squash(item_text):
        return None
    return from_breaks(item_text, breaks(paras(out_text)) | breaks(paras(proposed)))


def prf(tp: int, npred: int, nref: int) -> str:
    p, r = (tp / npred if npred else 0), (tp / nref if nref else 0)
    return f"P {p:.2f}  R {r:.2f}  F1 {2 * p * r / (p + r) if p + r else 0:.2f}"


def evaluate(folds: int) -> None:
    g = gold()
    random.Random(0).shuffle(g)
    client = anthropic.Anthropic(api_key=api_key())
    jobs = []
    for k in range(folds):
        test = g[k::folds]
        train = [x for x in g if x not in test]
        jobs += [(item, train) for item in test]
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda j: text_of(client.messages.create(**params(j[1], j[0]["audience"], j[0]["proposed"]))), jobs))
    keys = ("proposal", "code rules", "llm rules", "llm rules + code")
    tally = {(k, a): [0, 0, 0] for k in keys for a in ("self", "other", "all")}
    rejected = 0
    for (item, _), out in zip(jobs, outs):
        ref = breaks(item["paragraphs"])
        refined = finish(item["text"], item["proposed"], out)
        if refined is None:
            rejected += 1
            refined = item["proposed"]
        preds = {
            "proposal": breaks(paras(item["proposed"])),
            "code rules": breaks(code_rules(item["proposed"])),
            "llm rules": breaks(paras(refined)),
            "llm rules + code": breaks(code_rules(refined)),
        }
        for k, pred in preds.items():
            for a in (item["audience"], "all"):
                t = tally[(k, a)]
                t[0] += len(pred & ref); t[1] += len(pred); t[2] += len(ref)
    print(f"{len(g)} gold items, {folds}-fold, {rejected} outputs rejected for altering text")
    for a in ("all", "self", "other"):
        n = sum(1 for item, _ in jobs if a in ("all", item["audience"]))
        print(f" audience={a} ({n} items)")
        for k in keys:
            print(f"  {k:17s} {prf(*tally[(k, a)])}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    if cmd == "eval":
        evaluate(int(args.get("--folds", 4)))
    else:
        sys.exit(__doc__)
