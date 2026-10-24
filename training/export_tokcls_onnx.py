"""Export an r5 encoder classifier to ONNX (fp32 + int8 dynamic) and check it against PyTorch on CPU.

  python training/export_tokcls_onnx.py runs/<run>      # needs runs/<run>/model (from the Modal volume)

Runs locally on CPU (the models are 14-33M parameters). Writes runs/<run>/onnx/{model.onnx,
model_int8.onnx, tokenizer files, config.json} and runs/<run>/onnx_check.json. Inputs: input_ids,
attention_mask (int64, [batch, seq]); output: logits [batch, seq, 2]; P(paragraph) = softmax[..., 1],
break where >= config.paragraph_threshold on a word's first sub-token. Windowing and the word-level
merge are those of train_tokcls.py, reused here so the check covers the whole pipeline.

The check runs every test.jsonl and test_extra.jsonl memo through PyTorch and each ONNX file and reports:
words whose break decision differs, max |prob diff|, break P/R/F1, and mean/max milliseconds per memo
(all windows of one memo, ONNX Runtime CPU provider, default threads and 1 thread).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import train_tokcls as T  # noqa: E402


def export(model_dir: Path, onnx_dir: Path):
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    model.config.return_dict = False
    onnx_dir.mkdir(exist_ok=True)
    ids = torch.ones(1, 32, dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(model, (ids, torch.ones_like(ids)), str(onnx_dir / "model.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                        "logits": {0: "batch", 1: "seq"}},
                          opset_version=14, dynamo=False)
    quantize_dynamic(str(onnx_dir / "model.onnx"), str(onnx_dir / "model_int8.onnx"), weight_type=QuantType.QInt8)
    for f in model_dir.iterdir():
        if f.suffix in (".json", ".txt") and f.name != "model.safetensors.index.json":
            shutil.copy(f, onnx_dir / f.name)
    # Plain WordPiece vocab (one token per line, line number = id) for app-side tokenizers.
    vocab = AutoTokenizer.from_pretrained(model_dir).get_vocab()
    (onnx_dir / "vocab.txt").write_text("".join(t + "\n" for t, _ in sorted(vocab.items(), key=lambda x: x[1])))
    return model


def probs_with(fn, tok, rows):
    res, ms = [], []
    for r in rows:
        words, _, exs = T.examples(tok, r, with_labels=False)
        t = time.perf_counter()
        best = {}
        for ids, _, pos in exs:
            logits = fn(np.array([ids], dtype=np.int64))
            e = np.exp(logits - logits.max(-1, keepdims=True))
            pr = (e / e.sum(-1, keepdims=True))[:, 1]
            for p, w, c in pos:
                if w not in best or c > best[w][0]:
                    best[w] = (c, float(pr[p]))
        ms.append((time.perf_counter() - t) * 1000)
        res.append([best[i][1] if i > 0 else 0.0 for i in range(len(words))])
    return res, ms


def main():
    run = Path(sys.argv[1])
    model_dir, onnx_dir = run / "model", run / "onnx"
    model = export(model_dir, onnx_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    T.LIVE = bool(getattr(model.config, "paragraph_live", False))
    thr = model.config.paragraph_threshold
    rows = {n: [json.loads(l) for l in open(HERE / n)] for n in ("test.jsonl", "test_extra.jsonl")}

    def pt(ids):
        with torch.no_grad():
            return model(input_ids=torch.from_numpy(ids))[0][0].numpy()

    report = {"threshold": thr, "live": T.LIVE, "onnxruntime": ort.__version__, "opset": 14}
    ref = {n: probs_with(pt, tok, r) for n, r in rows.items()}
    report["pytorch"] = {n: {**T.score(r, ref[n][0], thr)[0], "ms_per_memo": round(float(np.mean(ref[n][1])), 1)}
                         for n, r in rows.items()}
    for name in ("model.onnx", "model_int8.onnx"):
        entry = {"mb": round((onnx_dir / name).stat().st_size / 2**20, 1)}
        for threads in (0, 1):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            sess = ort.InferenceSession(str(onnx_dir / name), so, providers=["CPUExecutionProvider"])
            fn = lambda ids: sess.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
            for n, r in rows.items():
                probs, ms = probs_with(fn, tok, r)
                key = f"{n}/threads={threads or 'default'}"
                flips = sum((a >= thr) != (b >= thr) for P, R in zip(probs, ref[n][0]) for a, b in zip(P, R))
                diff = max(abs(a - b) for P, R in zip(probs, ref[n][0]) for a, b in zip(P, R))
                entry[key] = {**T.score(r, probs, thr)[0], "decisions_differing_from_pytorch": int(flips),
                              "words": sum(len(P) for P in probs), "max_prob_diff": round(diff, 5),
                              "ms_per_memo_mean": round(float(np.mean(ms)), 1), "ms_per_memo_max": round(float(np.max(ms)), 1)}
        report[name] = entry
        print(name, json.dumps(entry), file=sys.stderr)
    (run / "onnx_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
