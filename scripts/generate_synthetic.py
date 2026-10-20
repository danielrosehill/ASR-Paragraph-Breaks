# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic"]
# ///
"""Generate synthetic, already-paragraphed voice-memo transcripts with Claude Haiku 5.5 (Message Batches).

  uv run scripts/generate_synthetic.py submit [--n 500] [--seed 1]   # batch; writes data/synthetic/batch-<seed>.json
  uv run scripts/generate_synthetic.py collect [--seed 1]            # waits, writes data/synthetic/synthetic-<seed>.jsonl
  uv run scripts/generate_synthetic.py run --n 30 --seed 2           # small sets: direct calls, same output file
  uv run scripts/generate_synthetic.py run --n 300 --seed 3 --kinds email   # dictated emails only (EMAIL_KINDS)

Seed 1 (2026-10-21, 500 items) used STYLES only. Later seeds also draw from STYLES_VARIED, added after
the first reviews asked for more variation in style. Every spec records its `audience`
(self/other), which rule 4 of docs/paragraphing-style.md depends on.

Each output is one memo with paragraphs separated by blank lines. The flat input is derived from it
by joining the paragraphs, so the pair satisfies the only-whitespace-differs invariant by construction.
build_pool.py then adds these items as source `syn` with a `proposed` field, and the Space shows the
proposal for review instead of flat text.

API key: $ANTHROPIC_API_KEY if it works; the exported one on this machine is dead (2026-10-15), so the
script falls back to the 1Password item "Anthropic / Claude Experiments API Key" (vault
"API Keys & Programmatic", field password) via `op item get`.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "synthetic"
MODEL = "claude-haiku-5-5"

KINDS = [
    "a voice memo to self planning the week ahead",
    "a dictated email to a colleague",
    "a dictated email to a landlord or property manager",
    "a voice journal entry at the end of the day",
    "a project status update recorded for a team",
    "a brain-dump of ideas for a side project",
    "notes after a doctor's appointment",
    "a dictated message to a family member",
    "a walkthrough of a problem with some software and what has been tried",
    "a dictated outline for a blog post",
    "a recap of a meeting just finished",
    "instructions for someone looking after the house while away",
    "a voice note comparing options for a purchase",
    "a dictated prompt for an AI assistant explaining what is needed",
    "a voice memo about a home repair job",
    "a dictated complaint to a company's customer service",
    "a reflection on a book or podcast",
    "a travel plan being thought through out loud",
    "a voice note to a friend catching up on news",
    "a dictated set of requirements for a freelancer",
]
# Seed 3 (2026-10-21): emails only, for the sibling Email-Key-Line-Bolding project. Every entry starts
# "a dictated email to", which is how that repo's build_pool.py selects emails.
EMAIL_KINDS = [
    "a dictated email to a colleague",
    "a dictated email to a landlord or property manager",
    "a dictated email to a client about a project",
    "a dictated email to a supplier or vendor about an order",
    "a dictated email to a child's teacher or school office",
    "a dictated email to a doctor's or dentist's office",
    "a dictated email to an HR department or a manager",
    "a dictated email to a contractor or tradesperson",
    "a dictated email to a bank, insurer or utility company",
    "a dictated email to a group of volunteers or a club",
    "a dictated email to a freelancer with a brief",
    "a dictated email to an event organiser or venue",
]
TOPICS = [
    "moving apartment", "a home network and server setup", "a child's sleep schedule", "a tax filing deadline",
    "a broken washing machine", "learning a new language", "a job application", "a car service",
    "a birthday party", "a new podcast idea", "a garden", "a health insurance claim", "a kitchen renovation",
    "a mobile phone contract", "a backup strategy for photos", "a running plan", "a client invoice dispute",
    "a grocery budget", "a trip to visit relatives", "a speech recognition app", "a volunteer event",
    "a leaking roof", "choosing a laptop", "a school enrolment form", "a dentist visit", "a noisy neighbour",
    "a small business website", "meal prepping", "a conference talk", "a bank account closure",
    "a pet's vet appointment", "an online course", "a bike repair", "a wedding", "a power outage",
    "a shared family calendar", "a lost parcel", "an electricity bill", "a research question", "a hiking trip",
]
# Kinds whose recording is addressed to another person or organisation; everything else is a note to self.
OTHER_MARKERS = ("email", "message", "for a team", "instructions for someone", "complaint", "to a friend",
                 "for a freelancer", "prompt for an AI")


def audience(kind: str) -> str:
    return "other" if any(m in kind for m in OTHER_MARKERS) else "self"


LENGTHS = [(150, 250), (250, 400), (400, 600)]
STYLES = [
    "Calm and organised; the speaker has thought about it beforehand.",
    "Thinking out loud; the speaker occasionally backtracks or corrects themselves.",
    "Brisk and practical; short sentences.",
    "Warm and conversational.",
]
STYLES_VARIED = [
    "Rambling; drifts onto a tangent once or twice and comes back to the point.",
    "Terse and clipped, almost telegraphic; fragments are fine.",
    "Frustrated and venting; the speaker is annoyed but stays coherent.",
    "Enumerates a lot out loud: first, second, then, also, and finally.",
    "Formal and polite, as if reading out a letter.",
    "Excited and fast; one idea tumbles into the next.",
    "Tired, late at night; long run-on sentences joined with 'and' and 'so'.",
    "Hesitant; the speaker thinks aloud, weighs options and changes their mind once.",
]

SYSTEM = """You write realistic transcripts of English voice recordings, as a modern speech recognition engine would output them: correct punctuation and casing, sentence boundaries, spoken register (contractions, "so", "okay", "I mean", the occasional mid-sentence self-correction), but no "um"/"uh" and no transcription errors.

Then you place paragraph breaks the way a careful human editor would on a transcript of speech: a new paragraph at each change of subject or step, and before a sign-off. Prefer short paragraphs of one to four sentences. Do not split a single thought across paragraphs.

Output only the transcript. Separate paragraphs with one blank line. No title, no headings, no bullet points, no lists, no markdown, no quotation marks around the whole text, no speaker labels."""


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        try:
            anthropic.Anthropic(api_key=key).models.retrieve(MODEL)
            return key
        except anthropic.AuthenticationError:
            pass
    return subprocess.run(
        ["op", "item", "get", "Anthropic / Claude Experiments API Key", "--vault", "API Keys & Programmatic",
         "--fields", "password", "--reveal"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def specs(n: int, seed: int, kinds: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    styles = STYLES if seed == 1 else STYLES + STYLES_VARIED
    kinds = kinds or KINDS
    out = []
    for i in range(n):
        lo, hi = rng.choice(LENGTHS)
        kind = rng.choice(kinds)
        out.append({"id": f"syn-{seed}-{i:04d}", "kind": kind, "topic": rng.choice(TOPICS),
                    "words": [lo, hi], "style": rng.choice(styles), "audience": audience(kind)})
    return out


def params(s: dict) -> dict:
    return dict(model=MODEL, max_tokens=8000, system=SYSTEM, output_config={"effort": "low"},
                messages=[{"role": "user", "content": prompt(s)}])


def prompt(s: dict) -> str:
    return (f"Write {s['kind']}, about {s['topic']}. Length: {s['words'][0]} to {s['words'][1]} words. "
            f"Speaker style: {s['style']} Invent concrete names, dates and details.")


def submit(n: int, seed: int) -> None:
    client = anthropic.Anthropic(api_key=api_key())
    ss = specs(n, seed)
    batch = client.messages.batches.create(requests=[
        Request(custom_id=s["id"], params=MessageCreateParamsNonStreaming(**params(s))) for s in ss
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"batch-{seed}.json").write_text(json.dumps({"batch_id": batch.id, "model": MODEL, "seed": seed, "specs": ss}, indent=1))
    print(f"submitted {batch.id} ({n} requests)")


BAD = re.compile(r"^\s*(#|[-*•]\s|\d+[.)]\s)", re.M)


def collect(seed: int) -> None:
    meta = json.loads((OUT_DIR / f"batch-{seed}.json").read_text())
    client = anthropic.Anthropic(api_key=api_key())
    while (b := client.messages.batches.retrieve(meta["batch_id"])).processing_status != "ended":
        print(f"{b.processing_status}: {b.request_counts.processing} processing", flush=True)
        time.sleep(60)
    results = [(r.custom_id, r.result.message if r.result.type == "succeeded" else r.result.type)
               for r in client.messages.batches.results(meta["batch_id"])]
    write(seed, meta["specs"], results)


def run(n: int, seed: int, kinds: list[str] | None = None) -> None:
    """Direct (non-batch) calls in parallel: for small sets where waiting on a batch isn't worth it."""
    from concurrent.futures import ThreadPoolExecutor
    client = anthropic.Anthropic(api_key=api_key())
    ss = specs(n, seed, kinds)
    with ThreadPoolExecutor(8) as ex:
        msgs = list(ex.map(lambda s: client.messages.create(**params(s)), ss))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"batch-{seed}.json").write_text(json.dumps({"batch_id": None, "model": MODEL, "seed": seed, "specs": ss}, indent=1))
    write(seed, ss, [(s["id"], m) for s, m in zip(ss, msgs)])


def write(seed: int, ss: list[dict], results: list) -> None:
    by_id = {s["id"]: s for s in ss}
    rows, rejected, usage = [], {}, [0, 0]
    for custom_id, msg in results:
        if isinstance(msg, str):
            rejected[msg] = rejected.get(msg, 0) + 1
            continue
        usage[0] += msg.usage.input_tokens
        usage[1] += msg.usage.output_tokens
        text = next((blk.text for blk in msg.content if blk.type == "text"), "").strip()
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        reason = ("max_tokens" if msg.stop_reason == "max_tokens" else
                  "markdown" if BAD.search(text) or "**" in text else
                  "one_paragraph" if len(paras) < 2 else None)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        s = by_id[custom_id]
        rows.append({"id": custom_id, "spec": s, "proposed": "\n\n".join(paras), "text": " ".join(paras),
                     "n_paragraphs": len(paras), "words": len(" ".join(paras).split()), "model": MODEL})
    rows.sort(key=lambda x: x["id"])
    out = OUT_DIR / f"synthetic-{seed}.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"{len(rows)} kept, rejected {rejected}, usage in={usage[0]} out={usage[1]} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    if cmd == "submit":
        submit(int(args.get("--n", 500)), int(args.get("--seed", 1)))
    elif cmd == "collect":
        collect(int(args.get("--seed", 1)))
    elif cmd == "run":
        run(int(args.get("--n", 30)), int(args.get("--seed", 2)), EMAIL_KINDS if args.get("--kinds") == "email" else None)
    else:
        sys.exit(__doc__)
