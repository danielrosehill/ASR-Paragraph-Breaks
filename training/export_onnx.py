"""Export a trained run to ONNX (fp32 + int8 dynamic-quantized) and check it against the PyTorch model.

Runs on Modal's `ai-training` app with --run set to the training run, so $OUT is that run's directory
and the PyTorch model is $OUT/model. Writes $OUT/onnx/ and $OUT/onnx_check.json. Command in
docs/training.md. Export uses optimum-onnx (its Gemma3TextOnnxConfig, model_type gemma3_text), which
pins transformers<4.58, so the command installs it into its own venv rather than the image's Python.

The check generates greedily on the first N test items with ONNX Runtime's CPU provider (the phone
case) and compares against the run's own pred_ft.jsonl: exact-match rate, the non-whitespace-text
invariant, and seconds per item. CPU timing on a Modal host is only a rough guide to a phone.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ["OUT"])
HERE = Path(__file__).parent
N = int(os.environ.get("N_CHECK", "6"))
SYSTEM = ("Add paragraph breaks to this voice-note transcript. Separate paragraphs with a blank line. "
          "Do not change, add or remove any word or punctuation mark.")


def squash(t):
    return re.sub(r"\s+", "", t)


def main():
    model_dir, onnx_dir = OUT / "model", OUT / "onnx"
    subprocess.run(["optimum-cli", "export", "onnx", "--model", str(model_dir), "--task", "text-generation-with-past",
                    str(onnx_dir)], check=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(onnx_dir / "model.onnx"), str(onnx_dir / "model_quantized.onnx"), weight_type=QuantType.QInt8)

    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(onnx_dir)
    rows = [json.loads(l) for l in open(HERE / "test.jsonl")][:N]
    ref = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(OUT / "pred_ft.jsonl")}
    report = {}
    for name in ("model.onnx", "model_quantized.onnx"):
        m = ORTModelForCausalLM.from_pretrained(onnx_dir, file_name=name, provider="CPUExecutionProvider")
        same = valid = 0
        secs = []
        for r in rows:
            prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"]}],
                                             tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt")
            t = time.time()
            out = m.generate(**enc, max_new_tokens=1600, do_sample=False)
            secs.append(time.time() - t)
            pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            same += pred == ref[r["id"]]
            valid += squash(pred) == squash(r["input"])
        report[name] = {"items": len(rows), "exact_match_vs_pytorch": same, "text_preserved": valid,
                        "seconds_per_item": round(sum(secs) / len(secs), 1),
                        "mb": round((onnx_dir / name).stat().st_size / 2**20, 1)}
        print(name, report[name], file=sys.stderr)
    (OUT / "onnx_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
