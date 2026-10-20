"""Paragraph-break annotator for flat ASR text.

Shows one transcript at a time. Items with a model proposal (synthetic `syn` items) come first
and open with the proposed breaks for review; the rest open flat. The annotator adds or removes
line breaks and saves. Only whitespace may change: the save is refused if any
word differs from the source. Each annotation is committed to the dataset repo as
annotations/<id>.jsonl (one line, so the Hub's json loader reads it as one row), using the signed-in user's own OAuth token (no token is stored in
the Space), so only accounts with write access to the dataset can save.
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone

import gradio as gr
from huggingface_hub import HfApi, hf_hub_download

DATASET = os.environ.get("DATASET_REPO", "danielrosehill/ASR-Paragraph-Breaks")
POOL_PATH = "pool/pool.jsonl"

api = HfApi()


def load_pool() -> list[dict]:
    path = hf_hub_download(DATASET, POOL_PATH, repo_type="dataset", force_download=True)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids() -> set[str]:
    files = api.list_repo_files(DATASET, repo_type="dataset")
    return {f[len("annotations/"):-len(".jsonl")] for f in files if f.startswith("annotations/") and f.endswith(".jsonl")}


POOL = load_pool()
BY_ID = {r["id"]: r for r in POOL}


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def paragraphs_of(edited: str) -> list[str]:
    """Any line break counts as a paragraph break; whitespace inside a paragraph is collapsed."""
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n+", edited) if p.strip()]


def first_mismatch(a: str, b: str) -> str:
    """Show where the squashed texts diverge, so a stray edit is easy to find."""
    sa, sb = squash(a), squash(b)
    i = next((k for k in range(min(len(sa), len(sb))) if sa[k] != sb[k]), min(len(sa), len(sb)))
    return f"source …{sa[max(0, i - 30):i + 30]}…\nyours  …{sb[max(0, i - 30):i + 30]}…"


def pick_next(skipped: list[str]) -> tuple[dict | None, str]:
    done = done_ids()
    todo = [r for r in POOL if r["id"] not in done and r["id"] not in skipped]
    progress = f"{len(done)} annotated · {len(todo)} left in this session's queue · {len(POOL)} in pool"
    if not todo:
        return None, progress
    # Review model proposals before annotating flat text from scratch; random order within each.
    review = [r for r in todo if r.get("proposed")]
    return random.choice(review or todo), progress


def start_text(item: dict) -> str:
    return item.get("proposed") or item["text"]


def show(item: dict | None, progress: str):
    if item is None:
        return None, "", "Nothing left to annotate.", progress
    mode = "review the proposed breaks" if item.get("proposed") else "flat, add breaks"
    meta = f"**{item['id']}** · source `{item['source']}` · {item['words']} words · {mode}"
    return item["id"], start_text(item), meta, progress


def on_load(skipped):
    return show(*pick_next(skipped or []))


def on_skip(item_id, skipped):
    skipped = (skipped or []) + ([item_id] if item_id else [])
    return (*show(*pick_next(skipped)), skipped)


def on_save(item_id, edited, skipped, profile: gr.OAuthProfile | None, oauth_token: gr.OAuthToken | None):
    if profile is None or oauth_token is None:
        raise gr.Error("Sign in with Hugging Face first.")
    if not item_id:
        raise gr.Error("No item loaded.")
    item = BY_ID[item_id]
    if squash(edited) != squash(item["text"]):
        raise gr.Error("Only line breaks may change. First difference:\n" + first_mismatch(item["text"], edited))

    paras = paragraphs_of(edited)
    flat = " ".join(paras)
    breaks, pos = [], 0
    for p in paras[1:]:
        pos = flat.index(p, pos + 1)
        breaks.append(pos)
    record = {
        "id": item_id,
        "source": item["source"],
        "source_id": item.get("source_id"),
        "text": flat,
        "paragraphed": "\n\n".join(paras),
        "proposed": item.get("proposed"),
        "proposal_model": item.get("proposal_model"),
        "edited": None if not item.get("proposed") else "\n\n".join(paras) != item["proposed"],
        "paragraphs": paras,
        "break_offsets": breaks,
        "n_paragraphs": len(paras),
        "words": len(flat.split()),
        "annotator": profile.username,
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    HfApi(token=oauth_token.token).upload_file(
        path_or_fileobj=(json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
        path_in_repo=f"annotations/{item_id}.jsonl",
        repo_id=DATASET,
        repo_type="dataset",
        commit_message=f"Annotate {item_id} ({len(paras)} paragraphs)",
    )
    gr.Info(f"Saved {item_id}: {len(paras)} paragraphs.")
    return (*show(*pick_next(skipped or [])), skipped)


with gr.Blocks(title="ASR Paragraph Annotator") as demo:
    gr.Markdown(
        "# ASR paragraph annotator\n"
        "Fix the paragraph breaks, then **Save & next** (saving unchanged accepts the proposal). "
        "Press **Enter** to start a paragraph; delete a line break to merge. "
        "Any line break counts as a paragraph break. Don't change words or punctuation; the save is refused if you do. "
        f"Saves go to [`{DATASET}`](https://huggingface.co/datasets/{DATASET})."
    )
    gr.LoginButton()
    item_id = gr.State(None)
    skipped = gr.State([])
    meta = gr.Markdown()
    progress = gr.Markdown()
    box = gr.Textbox(label="Transcript", lines=22, max_lines=60, autoscroll=False)
    with gr.Row():
        save = gr.Button("Save & next", variant="primary")
        skip = gr.Button("Skip")
        reset = gr.Button("Reset text")

    demo.load(on_load, [skipped], [item_id, box, meta, progress])
    save.click(on_save, [item_id, box, skipped], [item_id, box, meta, progress, skipped])
    skip.click(on_skip, [item_id, skipped], [item_id, box, meta, progress, skipped])
    reset.click(lambda i: start_text(BY_ID[i]) if i else "", [item_id], [box])

if __name__ == "__main__":
    demo.launch()
