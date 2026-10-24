# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime>=1.17", "tokenizers>=0.15", "numpy", "huggingface_hub"]
# ///
"""Add paragraph breaks to flat ASR text with the ONNX model. Same windowing and merge as training.

  uv run paragraphs.py "flat transcript ..."        # or pipe the text in on stdin
  uv run paragraphs.py --fp32 < memo.txt             # onnx/model.onnx instead of model_int8.onnx

The input is split on whitespace; the output is the same words joined with a space, or with a blank
line before a word whose P(paragraph starts here) >= paragraph_threshold (config.json). The text
cannot change.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REPO = "danielrosehill/MiniLM-L12-ASR-Paragraph-Breaks"
MAXTOK, STRIDE = 510, 256  # content tokens per window (+ [CLS]/[SEP]); window step in tokens


def load(fp32=False, local=None):
    if local:
        d = Path(local)
    else:
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(REPO, allow_patterns=["onnx/*"])) / "onnx"
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    sess = ort.InferenceSession(str(d / ("model.onnx" if fp32 else "model_int8.onnx")), providers=["CPUExecutionProvider"])
    thr = json.loads((d / "config.json").read_text())["paragraph_threshold"]
    return tok, sess, thr


def windows(pieces):
    """(first_word, end_word) spans of at most MAXTOK tokens, cut at word boundaries, stepping STRIDE tokens."""
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
        nxt = w0 + 1
        while nxt < n and starts[nxt] < starts[w0] + STRIDE:
            nxt += 1
        w0 = max(w0 + 1, min(nxt, w1))


def word_probs(tok, sess, words):
    """P(paragraph starts at word i); each word takes it from the window where it is most central."""
    cls, sep, unk = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]"), tok.token_to_id("[UNK]")
    pieces = [e.ids or [unk] for e in tok.encode_batch(words, add_special_tokens=False)]
    best = {}
    for w0, w1 in windows(pieces):
        ids, pos = [cls], []
        for w in range(w0, w1):
            pos.append((len(ids), w))  # first sub-token of the word
            ids += pieces[w]
        ids.append(sep)
        a = np.array([ids], dtype=np.int64)
        logits = sess.run(["logits"], {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        pr = e[:, 1] / e.sum(-1)
        mid = (len(ids) - 1) / 2
        for p, w in pos:
            c = -abs(p - mid)
            if w not in best or c > best[w][0]:
                best[w] = (c, float(pr[p]))
    return [0.0] + [best[i][1] for i in range(1, len(words))]


def paragraphs(text, tok, sess, thr):
    words = text.split()
    if not words:
        return text
    out = []
    for w, p in zip(words, word_probs(tok, sess, words)):
        out.append(("\n\n" if p >= thr else " ") + w if out else w)
    return "".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fp32"]
    model = load(fp32="--fp32" in sys.argv)
    print(paragraphs(" ".join(args) if args else sys.stdin.read(), *model))
