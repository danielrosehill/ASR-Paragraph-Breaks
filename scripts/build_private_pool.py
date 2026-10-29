# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the private pool from data/private/ and push it to the private dataset.

  uv run scripts/build_private_pool.py            # writes data/private/pool.jsonl
  with-secrets uv run scripts/build_private_pool.py --upload

Sources (2026-10-29), all already paragraphed by Haiku, so every item opens as a proposal to review:

  hist  history-<seed>.jsonl from scripts/obfuscate_history.py: Daniel's own dictations, obfuscated
  syn   synthetic-<seed>.jsonl from scripts/generate_synthetic.py --private: seed 4 (memos), seed 5
        (emails with set openings and sign-offs)

Same schema as the public pool (docs/data-format.md), plus `kind`. Every item has priority 0, so the
Space serves them in random order across kinds. Dataset `danielrosehill/ASR-Paragraph-Breaks-Private`
and Space `danielrosehill/ASR-Paragraph-Annotator-Private` are private; data/ is gitignored, so
none of this reaches the public GitHub repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from style_rules import apply as style_rules  # same directory

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "data" / "private"
OUT = PRIVATE / "pool.jsonl"
DATASET = "danielrosehill/ASR-Paragraph-Breaks-Private"
PROPOSAL_METHOD = "haiku-5.5 + style_rules v1.1"


def rows():
    for path in sorted(PRIVATE.glob("history-*.jsonl")) + sorted(PRIVATE.glob("synthetic-*.jsonl")):
        source = "hist" if path.name.startswith("history-") else "syn"
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            yield {"id": r["id"], "source": source, "source_id": r["id"], "chunk": "1/1", "text": r["text"],
                   "words": r["words"], "proposed": "\n\n".join(style_rules(r["proposed"])),
                   "haiku_proposed": r["proposed"], "proposal_model": r["model"], "proposal_method": PROPOSAL_METHOD,
                   "audience": r["spec"].get("audience"), "style": r["spec"].get("style"),
                   "kind": r["spec"].get("kind"), "priority": 0}


def main() -> None:
    pool = list(rows())
    assert len({r["id"] for r in pool}) == len(pool), "duplicate ids"
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool), encoding="utf-8")
    by: dict[str, int] = {}
    for r in pool:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"{len(pool)} items, {sum(r['words'] for r in pool)} words -> {OUT.relative_to(ROOT)}  {by}")
    if "--upload" in sys.argv:
        env = {**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN") or os.environ["HUGGINGFACE_TOKEN"]}
        hf = os.path.expanduser("~/.local/bin/hf")
        subprocess.run([hf, "repo", "create", DATASET, "--repo-type", "dataset", "--private", "--exist-ok"], check=True, env=env)
        subprocess.run([hf, "upload", DATASET, str(OUT), "pool/pool.jsonl", "--repo-type", "dataset",
                        "--commit-message", f"Rebuild private pool: {len(pool)} items"], check=True, env=env)


if __name__ == "__main__":
    main()
