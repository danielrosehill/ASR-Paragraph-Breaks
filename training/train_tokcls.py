"""r5: paragraph breaks as word-level token classification with a small WordPiece encoder.

Runs on Modal's `ai-training` app like train.py (command in docs/training.md, section "Encoder
classifier (r5)"). Expects train.jsonl, test.jsonl and test_extra.jsonl next to this file.

Formulation: the flat text is split on whitespace into words; label 1 = a paragraph starts at this word
(every paragraph after the first), else 0. Only the first WordPiece sub-token of a word carries the
label; the rest are -100. The model can only choose where to break, so the text cannot change.

Long texts: windows of up to 510 content tokens ([CLS] … [SEP] = 512) cut at word boundaries, stepping
by STRIDE tokens; each word takes its prediction from the window in which it is most central.
LIVE=1 trains and evaluates the "live" variant instead: for each sentence, the input is the text up to
the end of that sentence (left-truncated to 510 tokens), and only that sentence's words are labelled
or predicted. No right context beyond the current sentence.

Class imbalance (about 3% of words start a paragraph): positive class weight POS_WEIGHT in the loss,
and the decision threshold is tuned on a dev split (DEV_FRAC of train source ids; augmented copies of
dev items are dropped from training; dev is scored on clean rows only). Test is never used for any choice.
Each of SEEDS seeds trains EPOCHS epochs and keeps its best dev epoch; the seed with the best dev F1 is
saved to $OUT/model (with paragraph_threshold in config.json). Scoring matches train.py: breaks are
offsets into the whitespace-stripped text, exact match.
"""
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = os.environ.get("BASE", "google/electra-small-discriminator")
OUT = Path(os.environ.get("OUT", "out"))
HERE = Path(__file__).parent
TRAIN = os.environ.get("TRAIN", "train.jsonl")
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "1e-4"))
BATCH = int(os.environ.get("BATCH", "16"))
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
DEV_FRAC = float(os.environ.get("DEV_FRAC", "0.1"))
LIVE = os.environ.get("LIVE", "0") == "1"
MAXTOK = 510
STRIDE = int(os.environ.get("STRIDE", "256"))  # window step in tokens: overlap = 510 - 256 = 254
SENT_END = re.compile(r"[.!?][\"')\]]*$")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def squash(t):
    return re.sub(r"\s+", "", t)


def breaks(text):
    out, n = set(), 0
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in paras[:-1]:
        n += len(squash(p))
        out.add(n)
    return out


def word_labels(row):
    words = row["input"].split()
    labels, got = [], []
    for i, p in enumerate(q for q in re.split(r"\n\s*\n", row["output"]) if q.strip()):
        ws = p.split()
        labels += [1 if (i > 0 and j == 0) else 0 for j in range(len(ws))]
        got += ws
    assert got == words, row["id"]
    return words, labels


def paragraphed(words, starts):
    paras, cur = [], []
    for i, w in enumerate(words):
        if i in starts and cur:
            paras.append(" ".join(cur))
            cur = []
        cur.append(w)
    paras.append(" ".join(cur))
    return "\n\n".join(paras)


def word_pieces(tok, words):
    ids = tok(words, add_special_tokens=False)["input_ids"]
    return [x if x else [tok.unk_token_id] for x in ids]


def windows(pieces):
    """Full-context windows: list of (first_word, end_word) covering all words."""
    starts = [0]
    for p in pieces:
        starts.append(starts[-1] + len(p))
    n, out, w0 = len(pieces), [], 0
    while True:
        w1 = w0
        while w1 < n and starts[w1 + 1] - starts[w0] <= MAXTOK:
            w1 += 1
        w1 = max(w1, w0 + 1)
        out.append((w0, w1))
        if w1 >= n:
            return out
        target = starts[w0] + STRIDE
        nxt = w0 + 1
        while nxt < n and starts[nxt] < target:
            nxt += 1
        w0 = max(w0 + 1, min(nxt, w1))


def sentences(words):
    out, s = [], 0
    for i, w in enumerate(words):
        if SENT_END.search(w) or i == len(words) - 1:
            out.append((s, i + 1))
            s = i + 1
    return out


def live_windows(pieces, words):
    """Live windows: (context_start, sent_start, sent_end); context ends at the sentence end."""
    out = []
    for s0, s1 in sentences(words):
        c0, n = s1, 0
        while c0 > 0 and n + len(pieces[c0 - 1]) <= MAXTOK:
            n += len(pieces[c0 - 1])
            c0 -= 1
        out.append((c0, max(c0, s0), s1))
    return out


def examples(tok, row, with_labels=True):
    """Model inputs for one memo: list of (input_ids, labels, [(token_pos, word_idx, centrality)])."""
    words, labels = word_labels(row) if with_labels else (row["input"].split(), None)
    pieces = word_pieces(tok, words)
    spans = ([(c0, s0, s1) for c0, s0, s1 in live_windows(pieces, words)] if LIVE
             else [(w0, w0, w1) for w0, w1 in windows(pieces)])
    out = []
    for c0, s0, s1 in spans:
        ids, lab, pos = [tok.cls_token_id], [-100], []
        end = s1
        for w in range(c0, end):
            p = len(ids)
            ids += pieces[w]
            scored = s0 <= w < s1
            lab += [labels[w] if (scored and labels is not None) else -100] + [-100] * (len(pieces[w]) - 1)
            if scored:
                pos.append((p, w))
        ids.append(tok.sep_token_id)
        lab.append(-100)
        mid = (len(ids) - 1) / 2
        pos = [(p, w, -abs(p - mid)) for p, w in pos]
        out.append((ids, lab, pos))
    return words, labels, out


def batches(items, size, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = [items[j] for j in idx[i:i + size]]
        L = max(len(c[0]) for c in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        for k, (a, b, _) in enumerate(chunk):
            ids[k, :len(a)] = torch.tensor(a)
            att[k, :len(a)] = 1
            lab[k, :len(b)] = torch.tensor(b)
        yield ids, att, lab


@torch.no_grad()
def word_probs(model, tok, rows):
    """P(paragraph starts here) per word, per memo."""
    model.eval()
    res = []
    for r in rows:
        words, _, exs = examples(tok, r, with_labels=False)
        best = {}
        for ids, _, pos in exs:
            logits = model(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()
            pr = torch.softmax(logits, -1)[:, 1].cpu()
            for p, w, c in pos:
                if w not in best or c > best[w][0]:
                    best[w] = (c, pr[p].item())
        res.append([best[i][1] if i > 0 else 0.0 for i in range(len(words))])
    return res


def score(rows, probs, thr):
    tp = npred = nref = 0
    preds = []
    for r, pr in zip(rows, probs):
        words = r["input"].split()
        out = paragraphed(words, {i for i, p in enumerate(pr) if p >= thr})
        assert squash(out) == squash(r["input"])
        pb, rb = breaks(out), breaks(r["output"])
        tp, npred, nref = tp + len(pb & rb), npred + len(pb), nref + len(rb)
        preds.append({"id": r["id"], "pred": out})
    p, rc = (tp / npred if npred else 0.0), (tp / nref if nref else 0.0)
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(f, 3), "tp": tp, "n_pred": npred,
            "n_ref": nref}, preds


def best_threshold(rows, probs):
    cands = [(score(rows, probs, t / 100)[0]["f1"], -abs(t - 50), t / 100) for t in range(5, 96)]
    return max(cands)[2]


def train_seed(seed, tok, train_rows, dev_rows):
    random.seed(seed)
    torch.manual_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=2).to(DEV)
    items = [e for r in train_rows for e in examples(tok, r)[2]]
    steps = EPOCHS * math.ceil(len(items) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, POS_WEIGHT], device=DEV), ignore_index=-100)
    rng = random.Random(seed)
    best, log = None, []
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for ids, att, lab in batches(items, BATCH, True, rng):
            ids, att, lab = ids.to(DEV), att.to(DEV), lab.to(DEV)
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = lossf(logits.float().view(-1, 2), lab.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += loss.item()
        probs = word_probs(model, tok, dev_rows)
        thr = best_threshold(dev_rows, probs)
        dev = score(dev_rows, probs, thr)[0]
        log.append({"epoch": ep + 1, "loss": round(tot / max(1, math.ceil(len(items) / BATCH)), 4), "thr": thr, **dev})
        print(f"seed {seed} epoch {ep + 1} {log[-1]}", file=sys.stderr)
        if best is None or dev["f1"] > best["dev"]["f1"]:
            best = {"epoch": ep + 1, "thr": thr, "dev": dev,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best.pop("state"))
    return model, best, log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load = lambda n: [json.loads(l) for l in open(HERE / n)]
    train, test, extra = load(TRAIN), load("test.jsonl"), load("test_extra.jsonl")
    base_id = lambda r: r["id"].removesuffix("-aug")
    ids = sorted({base_id(r) for r in train})
    dev_ids = set(random.Random(42).sample(ids, max(1, int(DEV_FRAC * len(ids)))))
    dev_rows = [r for r in train if r["id"] in dev_ids]
    tr_rows = [r for r in train if base_id(r) not in dev_ids]
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    assert tok.is_fast, "need a fast tokenizer for word pieces"
    print(f"{BASE} live={LIVE} train {len(tr_rows)} dev {len(dev_rows)} test {len(test)} extra {len(extra)} "
          f"lower={getattr(tok, 'do_lower_case', tok.backend_tokenizer.normalizer.__class__.__name__)}", file=sys.stderr)

    t0 = time.time()
    runs = []
    best_model = None
    for seed in SEEDS:
        ts = time.time()
        model, best, log = train_seed(seed, tok, tr_rows, dev_rows)
        secs = time.time() - ts
        tprobs = word_probs(model, tok, test)
        eprobs = word_probs(model, tok, extra)
        t_res, t_pred = score(test, tprobs, best["thr"])
        e_res, e_pred = score(extra, eprobs, best["thr"])
        run = {"seed": seed, **best, "test": t_res, "test_extra": e_res, "train_seconds": round(secs), "log": log}
        runs.append(run)
        print(f"seed {seed}: dev {best['dev']['f1']} thr {best['thr']} test {t_res} extra {e_res}", file=sys.stderr)
        if best_model is None or best["dev"]["f1"] > best_model[1]["dev"]["f1"]:
            best_model = (model, best, t_pred, e_pred, tprobs, eprobs, run)
        else:
            del model
        torch.cuda.empty_cache()

    model, best, t_pred, e_pred, tprobs, eprobs, sel = best_model
    model.config.paragraph_threshold = best["thr"]
    model.config.paragraph_live = LIVE
    model.config.id2label, model.config.label2id = {0: "O", 1: "PARA"}, {"O": 0, "PARA": 1}
    model.save_pretrained(str(OUT / "model"))
    tok.save_pretrained(str(OUT / "model"))
    for name, preds, probs in (("pred_ft.jsonl", t_pred, tprobs), ("pred_extra.jsonl", e_pred, eprobs)):
        (OUT / name).write_text("".join(json.dumps({**p, "probs": [round(x, 4) for x in pr]}, ensure_ascii=False) + "\n"
                                        for p, pr in zip(preds, probs)))
    mean = lambda k, m: round(sum(r[k][m] for r in runs) / len(runs), 3)
    metrics = {"base": BASE, "live": LIVE, "train_file": TRAIN, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
               "pos_weight": POS_WEIGHT, "n_train": len(tr_rows), "n_dev": len(dev_rows), "n_test": len(test),
               "n_params": sum(p.numel() for p in model.parameters()),
               "selected": {"seed": sel["seed"], "epoch": best["epoch"], "threshold": best["thr"], "dev": best["dev"],
                            "test": sel["test"], "test_extra": sel["test_extra"]},
               "seed_mean": {"test_f1": mean("test", "f1"), "test_precision": mean("test", "precision"),
                             "test_recall": mean("test", "recall"), "extra_f1": mean("test_extra", "f1")},
               "seconds_total": round(time.time() - t0), "runs": runs}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("selected", "seed_mean", "seconds_total")}), file=sys.stderr)


if __name__ == "__main__":
    main()
