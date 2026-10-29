# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic"]
# ///
"""Turn Daniel's own dictation history into obfuscated, already-paragraphed items for the private pool.

  with-secrets uv run scripts/obfuscate_history.py [--n 200] [--seed 1] [--db ~/.local/share/hebrish/history.db]

Reads the Hebrish desktop app's history (`dictations` table, source `dictate`, 100+ words), samples
`--n` takes (every take of 150+ words first, the rest from 100-149), and has Claude Haiku 5.5 rewrite
each one with names, places, products, numbers and small details changed, keeping the length,
register and rough ASR punctuation, then place paragraph breaks as generate_synthetic.py does.
Writes data/private/history-<seed>.jsonl in generate_synthetic.py's row format (ids `hist-<seed>-NNNN`).

Nothing links an item back to its history row: the take's uid is not stored. The raw history never
leaves the machine except as the Haiku request; only the rewrite is uploaded (private dataset).
Rewrites whose length is outside 0.75-1.33x the original are dropped. `--show-originals` prints the
start of each original next to its rewrite (pilot check; terminal only).
"""

from __future__ import annotations

import os
import random
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic import MODEL, PRIVATE_DIR, api_key, write  # noqa: E402

SYSTEM = """You rewrite a real voice-dictation transcript so it can be shared as training data without identifying anyone.

Change every personal name, company, project, product, app, place, street, date, amount and number to a different but plausible one, and change two or three small details of the situation. Keep everything else as close to the original as you can: the same length, the same sentence structure, the same speaking register and tone, the same order of ideas, and the same rough punctuation of speech recognition output (run-on sentences, an odd comma, a self-correction). Do not polish it, summarise it, or add anything. Romanised Hebrew words may stay.

Then place paragraph breaks the way a careful human editor would on a transcript of speech: a new paragraph at each change of subject or step, and before a sign-off. Prefer short paragraphs of one to four sentences. Do not split a single thought across paragraphs. A short transcript may stay one paragraph.

Output only the rewritten transcript. Separate paragraphs with one blank line. No title, no headings, no lists, no markdown, no commentary."""


def sample(db: Path, n: int, seed: int) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT text, words FROM dictations WHERE source = 'dictate' AND words >= 100 ORDER BY at").fetchall()
    seen, long_, short = set(), [], []
    for text, words in rows:
        key = text[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        (long_ if words >= 150 else short).append(text)
    rng = random.Random(seed)
    rng.shuffle(long_)
    rng.shuffle(short)
    return (long_ + short)[:n]


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--show-originals"]
    args = dict(zip(argv[::2], argv[1::2]))
    n, seed = int(args.get("--n", 200)), int(args.get("--seed", 1))
    db = Path(os.path.expanduser(args.get("--db", "~/.local/share/hebrish/history.db")))
    texts = sample(db, n, seed)
    client = anthropic.Anthropic(api_key=api_key())

    def call(text: str):
        return client.messages.create(model=MODEL, max_tokens=8000, system=SYSTEM, output_config={"effort": "low"},
                                      messages=[{"role": "user", "content": text}])

    with ThreadPoolExecutor(8) as ex:
        msgs = list(ex.map(call, texts))
    specs, results = [], []
    for i, (text, msg) in enumerate(zip(texts, msgs)):
        sid = f"hist-{seed}-{i:04d}"
        out = next((b.text for b in msg.content if b.type == "text"), "")
        ratio = len(out.split()) / len(text.split())
        specs.append({"id": sid, "kind": "own dictation, obfuscated", "style": "real dictation", "audience": None})
        results.append((sid, msg if 0.75 <= ratio <= 1.33 else "length_drift"))
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    # One-paragraph rewrites are kept: a real take that needs no break is a useful example too.
    write(seed, specs, results, PRIVATE_DIR, name="history", min_paragraphs=1)
    if "--show-originals" in sys.argv:  # pilot check, local only: is the rewrite really different?
        for t, m in zip(texts, msgs):
            print("ORIG:", t[:300], "\nNEW: ", next((b.text for b in m.content if b.type == "text"), "")[:300], "\n")


if __name__ == "__main__":
    main()
