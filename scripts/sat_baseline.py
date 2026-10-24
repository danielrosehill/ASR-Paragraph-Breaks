"""Score SaT (wtpsplit, `segment-any-text/sat-3l-sm`) paragraph segmentation on the r5 test sets (CPU).

  uv venv satv --python 3.12 && uv pip install -p satv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -p satv/bin/python wtpsplit
  satv/bin/python scripts/sat_baseline.py > runs/sat-3l-sm-baseline/metrics.json

Same scoring as training/train.py (break offsets into the whitespace-stripped text, exact match).
paragraph_threshold is reported at wtpsplit's default (0.5) and at the value with the best F1 on 100
clean train rows (never test). First run 2026-10-24 with wtpsplit 2.2.2.
"""
import json
import re
import sys
import time
from pathlib import Path

from wtpsplit import SaT

HERE = Path(__file__).resolve().parent.parent / "training"


def squash(t):
    return re.sub(r"\s+", "", t)


def breaks_from_paras(paras):
    out, n = set(), 0
    for p in paras[:-1]:
        n += len(squash(p))
        out.add(n)
    return out


def score(sat, rows, thr):
    tp = npred = nref = 0
    for r in rows:
        paras = ["".join(s) for s in sat.split(r["input"], do_paragraph_segmentation=True, paragraph_threshold=thr)]
        assert squash("".join(paras)) == squash(r["input"]), r["id"]
        pb = breaks_from_paras(paras)
        rb = breaks_from_paras([p for p in re.split(r"\n\s*\n", r["output"]) if p.strip()])
        tp, npred, nref = tp + len(pb & rb), npred + len(pb), nref + len(rb)
    p, rc = (tp / npred if npred else 0.0), (tp / nref if nref else 0.0)
    return {"threshold": thr, "precision": round(p, 3), "recall": round(rc, 3),
            "f1": round(2 * p * rc / (p + rc) if p + rc else 0.0, 3), "n_pred": npred, "n_ref": nref}


def main():
    load = lambda n: [json.loads(l) for l in open(HERE / n)]
    dev = [r for r in load("train.jsonl") if not r["label"].endswith("-aug")][:100]
    sat = SaT("sat-3l-sm")
    sweep = [score(sat, dev, t / 100) for t in range(5, 100, 5)]
    best = max(sweep, key=lambda s: s["f1"])
    print(f"dev sweep best {best}", file=sys.stderr)
    out = {"model": "segment-any-text/sat-3l-sm", "dev_rows": len(dev), "dev_best": best, "dev_sweep": sweep}
    for name in ("test.jsonl", "test_extra.jsonl"):
        rows = load(name)
        t = time.time()
        out[name] = {"default_0.5": score(sat, rows, 0.5), "dev_tuned": score(sat, rows, best["threshold"]),
                     "seconds_per_memo_cpu": round((time.time() - t) / (2 * len(rows)), 3)}
        print(name, out[name], file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
