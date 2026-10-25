# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Bring every annotations/<id>.jsonl on the Hub to the schema in docs/data-format.md.

Files saved before space/app.py wrote every field leave some out, and the dataset viewer then fails
with a CastError on the `annotations` config. This adds the missing fields with their documented
values, in the Space's key order, and changes nothing else.

  uv run scripts/normalise_annotations.py            # dry run: lists what would change
  uv run scripts/normalise_annotations.py --upload   # one commit with the changed files (needs write token)

Missing `proposal_method` on an item whose `proposed` equals the pool's `haiku_proposed` (the
syn-1 batch, annotated before scripts/style_rules.py existed) is set to "haiku-5.5". Any other
missing field becomes null, as docs/data-format.md specifies for items without a proposal.
"""

import json
import sys

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

DATASET = "danielrosehill/ASR-Paragraph-Breaks"
KEYS = ["id", "source", "source_id", "text", "paragraphed", "proposed", "proposal_model",
        "proposal_method", "edited", "paragraphs", "break_offsets", "n_paragraphs", "words",
        "annotator", "annotated_at"]
RAW_HAIKU_METHOD = "haiku-5.5"


def main(upload):
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(DATASET, repo_type="dataset")
                   if f.startswith("annotations/") and f.endswith(".jsonl"))
    pool = {}
    for line in open(hf_hub_download(DATASET, "pool/pool.jsonl", repo_type="dataset")):
        r = json.loads(line)
        pool[r["id"]] = r
    ops = []
    for f in files:
        rec = json.loads(open(hf_hub_download(DATASET, f, repo_type="dataset")).readline())
        extra = set(rec) - set(KEYS)
        if extra:
            sys.exit(f"{f}: fields not in the schema {sorted(extra)}; update KEYS and docs first")
        missing = [k for k in KEYS if k not in rec]
        if not missing:
            continue
        if "proposal_method" in missing and rec.get("proposed"):
            p = pool.get(rec["id"], {})
            if rec["proposed"] == p.get("haiku_proposed"):
                rec["proposal_method"] = RAW_HAIKU_METHOD
        fixed = {k: rec.get(k) for k in KEYS}
        print(f"{f}: adding {missing}" + (f" (proposal_method={fixed['proposal_method']!r})" if "proposal_method" in missing else ""))
        ops.append(CommitOperationAdd(f, (json.dumps(fixed, ensure_ascii=False) + "\n").encode("utf-8")))
    print(f"{len(ops)} of {len(files)} files need normalising")
    if upload and ops:
        c = api.create_commit(DATASET, ops, repo_type="dataset",
                              commit_message=f"Normalise {len(ops)} annotation files to the documented schema (fixes viewer CastError)")
        print(c.commit_url)


if __name__ == "__main__":
    main("--upload" in sys.argv)
