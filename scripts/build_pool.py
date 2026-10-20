# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow"]
# ///
"""Build pool/pool.jsonl: flat, punctuated ASR monologue waiting for paragraph annotation.

  uv run scripts/build_pool.py              # writes data/pool/pool.jsonl
  uv run scripts/build_pool.py --upload     # also pushes it to the HF dataset (needs HF_TOKEN)

Sources (all public; checked 2026-10-21):

  mwp  My-Weird-Prompts/episodes, column prompt_sanitized (CC-BY-4.0). Daniel's dictated podcast
       prompts. Kept: 300+ words, no line breaks, sender daniel/hannah/unset. 776 rows on 2026-10-21.
  svn  danielrosehill/STT-Voice-Notes-Evals, transcriptions/openai/whisper-1/raw/*.txt (Apache-2.0).
       LLM-written voice memos, read aloud and transcribed by whisper-1. 16 rows.
  nvn  danielrosehill/Noisy-Voice-Notes, transcripts/*.md (CC-BY-4.0). Real noisy voice notes. 7 rows.
  syn  data/synthetic/synthetic.jsonl from scripts/generate_synthetic.py (Claude Haiku 5.5). Already
       paragraphed: carries `proposed` (the model's breaks) for review in the Space. Listed first.

No large synthetic voice-memo dataset was found on the Hub (search of 2026-10-21; synthetic dictation
cleanup sets exist but are 10-90 words, too short to need paragraphs). Ids are stable across rebuilds.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pool" / "pool.jsonl"
HF = "https://huggingface.co/datasets"
DATASET = "danielrosehill/ASR-Paragraph-Breaks"

MIN_WORDS = 150   # below this there is rarely more than one paragraph
MAX_WORDS = 600   # longer items are split on sentence boundaries ...
TARGET_WORDS = 350  # ... into chunks of about this size


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def flatten(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # ASR segment joins drop the space after a full stop ("me.I live"). Restoring it is a
    # whitespace-only change, so it never conflicts with the only-whitespace-may-change rule.
    # The lowercase lookbehind leaves abbreviations like "U.S." alone.
    return re.sub(r"(?<=[a-z][.!?])(?=[A-Z][a-z])", " ", text)


def chunk(text: str) -> list[str]:
    if len(text.split()) <= MAX_WORDS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        if chunks and sum(len(x.split()) for x in cur) < MIN_WORDS:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def items(source: str, source_id: str, text: str):
    parts = [p for p in chunk(flatten(text)) if len(p.split()) >= MIN_WORDS]
    for i, part in enumerate(parts):
        yield {
            "id": f"{source}-{source_id}" + (f"-c{i + 1}" if len(parts) > 1 else ""),
            "source": source,
            "source_id": source_id,
            "chunk": f"{i + 1}/{len(parts)}",
            "text": part,
            "words": len(part.split()),
        }


def mwp():
    url = f"{HF}/My-Weird-Prompts/episodes/resolve/main/data/train-00000-of-00001.parquet"
    cols = ["episode_number", "prompt_sanitized", "prompt_sender_type"]
    for r in pq.read_table(io.BytesIO(fetch(url)), columns=cols).to_pylist():
        p = r["prompt_sanitized"] or ""
        if len(p.split()) >= 300 and "\n" not in p and r["prompt_sender_type"] in ("daniel", "hannah", "", None):
            yield from items("mwp", f"ep{r['episode_number']:04d}", p)


def svn():
    base = f"{HF}/danielrosehill/STT-Voice-Notes-Evals"
    tree = json.loads(fetch(f"https://huggingface.co/api/datasets/danielrosehill/STT-Voice-Notes-Evals/tree/main/transcriptions/openai/whisper-1/raw"))
    for f in sorted(tree, key=lambda f: f["path"]):
        name = Path(f["path"]).stem
        yield from items("svn", name, fetch(f"{base}/resolve/main/{f['path']}").decode("utf-8"))


def nvn():
    base = f"{HF}/danielrosehill/Noisy-Voice-Notes/resolve/main"
    for line in fetch(f"{base}/metadata.csv").decode("utf-8").splitlines()[1:]:
        nid = line.split(",", 1)[0]
        body = fetch(f"{base}/transcripts/{nid}.md").decode("utf-8").split("\n---\n", 1)[-1]
        yield from items("nvn", nid, re.sub(r"^#.*$", "", body, flags=re.M))


def syn():
    path = ROOT / "data" / "synthetic" / "synthetic.jsonl"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        yield {"id": r["id"], "source": "syn", "source_id": r["id"], "chunk": "1/1", "text": r["text"],
               "words": r["words"], "proposed": r["proposed"], "proposal_model": r["model"]}


def main() -> None:
    rows = [*syn(), *mwp(), *svn(), *nvn()]
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by: dict[str, int] = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"{len(rows)} items, {sum(r['words'] for r in rows)} words -> {OUT.relative_to(ROOT)}  {by}")

    if "--upload" in sys.argv:
        env = {**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN") or os.environ["HUGGINGFACE_TOKEN"]}
        subprocess.run(
            [os.path.expanduser("~/.local/bin/hf"), "upload", DATASET, str(OUT), "pool/pool.jsonl",
             "--repo-type", "dataset", "--commit-message", f"Rebuild pool: {len(rows)} items"],
            check=True, env=env,
        )


if __name__ == "__main__":
    main()
