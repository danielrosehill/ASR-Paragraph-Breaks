# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Charts and Hub README for a published run.

  uv run scripts/model_card.py <run-dir> [<other-run-dir> ...]   # first dir is the published run

Each run dir holds metrics.json and pred_ft.jsonl (fetched from the Modal volume into runs/<name>/).
Writes hub/model/{README.md,charts/*.png} and hub/onnx/README.md. Colours follow the dataviz skill's
reference palette (light surface #fcfcfb, series blue #2a78d6 / orange #eb6834, slots 1-2, the
validated adjacent pair); text stays in ink colours.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODEL_ID = "danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks"
ONNX_ID = MODEL_ID + "-ONNX"
RULES_F1 = 0.96  # style_rules v1.1 from Haiku's paragraphed proposal, same 26 items (docs/paragraphing-style.md)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def paras(t):
    return [p for p in re.split(r"\n\s*\n", t) if p.strip()]


def main(run_dirs: list[Path]) -> None:
    runs = [(d.name, json.loads((d / "metrics.json").read_text())) for d in run_dirs]
    pub_name, pub = runs[0]
    test = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "training/test.jsonl").read_text().splitlines()}
    out = ROOT / "hub" / "model"
    (out / "charts").mkdir(parents=True, exist_ok=True)

    # 1. F1 by method: single series, one hue; the rules row is a grey reference, not a competitor.
    rows = [("Untrained Gemma 3 270M", pub["base_model"]["f1"], BLUE)]
    rows += [(f"Fine-tuned {n.replace('asr-paragraph-', '')} ({'this model' if i == 0 else 'not published'})",
              m["fine_tuned"]["f1"], BLUE) for i, (n, m) in enumerate(runs)]
    rows.append(("Reference: rules on a paragraphed draft", RULES_F1, "#b9b8b0"))
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.0), facecolor=SURFACE)
    style(ax)
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55, edgecolor=SURFACE, linewidth=2)
    for yi, (label, v, _) in zip(y, rows):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Paragraph-break F1 on 26 held-out hand-reviewed memos", color=INK2, fontsize=9)
    ax.set_title("Paragraph-break F1", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "charts/f1_by_method.png", dpi=160)
    plt.close(fig)

    # 2. Paragraph length: gold vs the published model, words per paragraph.
    preds = {json.loads(l)["id"]: json.loads(l)["pred"] for l in (run_dirs[0] / "pred_ft.jsonl").read_text().splitlines()}
    gold_len = [len(p.split()) for r in test.values() for p in paras(r["output"])]
    pred_len = [len(p.split()) for i, pr in preds.items() for p in paras(pr)]
    bins = list(range(0, max(gold_len + pred_len) + 10, 10))
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.hist(gold_len, bins=bins, histtype="step", linewidth=2, color=BLUE, label=f"Hand-reviewed ({len(gold_len)} paragraphs)")
    ax.hist(pred_len, bins=bins, histtype="step", linewidth=2, color=ORANGE, label=f"Model ({len(pred_len)} paragraphs)")
    ax.set_xlabel("Words per paragraph", color=INK2, fontsize=9)
    ax.set_ylabel("Paragraphs", color=INK2, fontsize=9)
    ax.set_title("Paragraph length: hand-reviewed vs model", loc="left", color=INK, fontsize=11)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(out / "charts/paragraph_lengths.png", dpi=160)
    plt.close(fig)

    # 3. Training loss of the published run.
    log = [e for e in pub["log"] if "loss" in e and "epoch" in e]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot([e["epoch"] for e in log], [e["loss"] for e in log], color=BLUE, linewidth=2)
    ax.set_xlabel("Epoch", color=INK2, fontsize=9)
    ax.set_ylabel("Training loss", color=INK2, fontsize=9)
    ax.set_title(f"Training loss ({pub_name.replace('asr-paragraph-', '')})", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "charts/training_loss.png", dpi=160)
    plt.close(fig)

    ft = pub["fine_tuned"]
    table = "\n".join(
        f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {round(m['fallback_rate'] * pub['n_test'])} of {pub['n_test']} |"
        for label, m in [("Untrained Gemma 3 270M", pub["base_model"])] + [
            (f"Fine-tuned {n.replace('asr-paragraph-', '')} ({'this model' if i == 0 else f'not published: {r_["n_train"]} pairs, {int(r_["epochs"])} epochs'})", r_["fine_tuned"])
            for i, (n, r_) in enumerate(runs)])
    readme = TEMPLATE.format(model_id=MODEL_ID, onnx_id=ONNX_ID, n_train=pub["n_train"], n_test=pub["n_test"],
                             epochs=int(pub["epochs"]), lr=pub["lr"], table=table, f1=ft["f1"], run=pub_name)
    (out / "README.md").write_text(readme)
    onnx = ROOT / "hub" / "onnx"
    onnx.mkdir(parents=True, exist_ok=True)
    check = run_dirs[0] / "onnx_check.json"
    check_md = ""
    if check.exists():
        c = json.loads(check.read_text())
        check_md = "\n".join(f"| `onnx/{k}` | {v['mb']:.0f} MB | {v['exact_match_vs_pytorch']} of {v['items']} | "
                             f"{v['text_preserved']} of {v['items']} | {v['seconds_per_item']} s |" for k, v in c.items())
    (onnx / "README.md").write_text(ONNX_TEMPLATE.format(model_id=MODEL_ID, onnx_id=ONNX_ID, check=check_md))
    print(f"wrote {out}/README.md, 3 charts, {onnx}/README.md")


TEMPLATE = """---
license: gemma
base_model: google/gemma-3-270m-it
language:
  - en
pipeline_tag: text-generation
library_name: transformers
datasets:
  - danielrosehill/ASR-Paragraph-Breaks
tags:
  - asr-post-processing
  - paragraph-segmentation
  - dictation
  - voice-notes
  - on-device
---

# Gemma 3 270M: ASR paragraph breaks

Adds paragraph breaks to flat speech-to-text output and changes nothing else. Speech recognition
(here, a Parakeet fine-tune) returns punctuated, cased text as one block; this model returns the same
text with paragraphs separated by blank lines. It is small enough to run on a phone after the ASR pass.
An ONNX export for ONNX Runtime is at [`{onnx_id}`](https://huggingface.co/{onnx_id}).
Try it in the [demo Space](https://huggingface.co/spaces/danielrosehill/ASR-Paragraph-Breaks-Demo) (CPU; optionally chains the email key-line bolding model). Write-up: [blog post](https://blog.danielrosehill.com/posts/asr-paragraph-breaks-gemma-3-270m-85hckk/).

## Results

Scored on {n_test} held-out synthetic voice memos whose paragraph breaks were reviewed by hand, as
break-position precision and recall. Any output that changes a non-whitespace character counts as a
fallback: the input is returned unchanged, as it should be in deployment.

| Model | Precision | Recall | F1 | Fallbacks |
|---|---|---|---|---|
{table}

![Paragraph-break F1 by method](charts/f1_by_method.png)

The grey reference row is the rule-based labeller used to make most of the training data. It starts
from an already-paragraphed draft, so it is a ceiling, not a like-for-like baseline: the model starts
from flat text.

![Paragraph length](charts/paragraph_lengths.png)

![Training loss](charts/training_loss.png)

## Usage

Always check the output: if anything other than whitespace changed, use the input instead.

```python
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained("{model_id}")
SYSTEM = ("Add paragraph breaks to this voice-note transcript. Separate paragraphs with a blank line. "
          "Do not change, add or remove any word or punctuation mark.")

def paragraph(text: str) -> str:
    prompt = tok.apply_chat_template([{{"role": "system", "content": SYSTEM}}, {{"role": "user", "content": text}}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=1600, do_sample=False)
    pred = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    same = re.sub(r"\\s+", "", pred) == re.sub(r"\\s+", "", text)
    return pred if same else text
```

## Training

Full fine-tune of [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) with TRL SFT
on prompt/completion pairs (completion-only loss): {n_train} training pairs, {epochs} epochs, learning
rate {lr}, bf16, one NVIDIA L4 on Modal (run `{run}`).

Data: [`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks)
(the exact train/test split is in its `training/` folder). Synthetic voice-memo transcripts were written
by Claude Haiku 5.5, a minority of them paragraphed by hand, and the rest labelled by rules derived from
those hand edits. Code: [github.com/danielrosehill/ASR-Paragraph-Breaks](https://github.com/danielrosehill/ASR-Paragraph-Breaks).

## Limitations

- English only, and trained on synthetic memos rather than real ASR output.
- The paragraphing style is one person's: short paragraphs, the opening line and each sign-off line
  on their own.
- Memos longer than about 700 words were not in training; split longer transcripts first.
- It is a generative model, so it retypes the whole text. Keep the fallback check.

Licence: Gemma Terms of Use, inherited from the base model.
"""

ONNX_TEMPLATE = """---
license: gemma
base_model: {model_id}
language:
  - en
pipeline_tag: text-generation
library_name: onnxruntime
tags:
  - onnx
  - asr-post-processing
  - paragraph-segmentation
  - on-device
---

# Gemma 3 270M: ASR paragraph breaks (ONNX)

ONNX export of [`{model_id}`](https://huggingface.co/{model_id}) for ONNX Runtime, so it can run on a
phone on the same runtime as an ONNX speech-recognition model. Exported with `optimum-onnx`
(`text-generation-with-past`, with a KV cache); `model_quantized.onnx` is int8 dynamic quantization of
`model.onnx`. See the main model card for what it does, results and the required fallback check.

## Check against the PyTorch model

Greedy decoding on held-out test memos, ONNX Runtime CPU provider. Timing is on a cloud CPU, so treat
it as a rough guide to a phone.

| File | Size | Same output as PyTorch | Text unchanged | Time per memo |
|---|---|---|---|---|
{check}

## Usage

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("{onnx_id}")
model = ORTModelForCausalLM.from_pretrained("{onnx_id}", subfolder="onnx", file_name="model.onnx")
```

Use the prompt and the fallback check from the main model card. Prefer `model.onnx`: the int8 file
matched the PyTorch output on only 2 of 6 memos in the check above and changed the text in one, so it
is published as an experiment. Generation is slow on CPU because the model retypes the whole memo;
a lower-bit weight-only quantization and prompt-lookup decoding are the next things to try.

Licence: Gemma Terms of Use, inherited from the base model.
"""

if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
