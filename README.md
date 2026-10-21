# ASR Paragraph Breaks

A small text-in/text-out model that adds paragraph breaks to flat ASR output and changes nothing else.

Speech-to-text output is punctuated and cased but arrives as one block, even from engines that mark
sentence boundaries. This project hand-annotates where the paragraphs go and fine-tunes a small model on
those annotations.

**Status (2026-10-21): first model released.** A Gemma 3 270M fine-tune (run r2) scores break F1 0.78
on 26 held-out hand-reviewed memos, with 2 of 26 outputs falling back to the input. Published on the Hub
with an ONNX export; weights archived in Wasabi (`small-models`). Training data: 500 synthetic memos
from Claude Haiku 5.5, hand-reviewed where possible and otherwise labelled by rules derived from those
reviews. Details: [`docs/training.md`](docs/training.md).

| | |
|---|---|
| Dataset | [`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks) |
| Annotation Space | [`danielrosehill/ASR-Paragraph-Annotator`](https://huggingface.co/spaces/danielrosehill/ASR-Paragraph-Annotator) |
| Model | [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks), ONNX: [`…-ONNX`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX) |
| Demo Space | [`danielrosehill/ASR-Paragraph-Breaks-Demo`](https://huggingface.co/spaces/danielrosehill/ASR-Paragraph-Breaks-Demo) (CPU; optional email key-line bolding step) |

## Why model-proposed, hand-reviewed pairs

Flattening edited prose (articles, Wikipedia) and learning to restore its breaks teaches the conventions
of written text, not of dictated speech. Annotating real voice-note transcripts from scratch works but is
slow, and the transcripts are scrappy. So Haiku writes spoken-register memos with paragraph breaks
already in place, the flat input is derived by joining the paragraphs (the pair can only differ in
whitespace), and every proposal is reviewed by hand. The `edited` rate is itself a measure of how close
a prompted model already gets.

## Why a generative model, given SaT exists

[SaT / wtpsplit](https://github.com/segment-any-text/wtpsplit) (`segment-any-text/sat-3l-sm`) already
does paragraph segmentation by predicting newline probability, and it is the baseline to beat. The goal
here is a plain text-in/text-out model trained on one person's preferences. Every output is checked
against the input: if anything other than whitespace changed, fall back to the input.

## Layout

| Path | What |
|---|---|
| [`scripts/generate_synthetic.py`](scripts/generate_synthetic.py) | Haiku 5.5 batch generation of paragraphed memos (`submit`, then `collect`); prompt, scenario grid and key lookup are in the file |
| [`scripts/build_pool.py`](scripts/build_pool.py) | Builds `pool/pool.jsonl` from public HF sources; `--upload` pushes it. Sources and filters are in its docstring |
| [`space/`](space/) | The Gradio annotation Space (deployed to HF as-is) |
| [`demo/`](demo/) | The public demo Space (deployed to HF as-is); examples are 4 held-out golden `syn-2-*` memos, flattened. Measured 2026-10-21 on CPU basic: 22 s for a 208-word memo, 42 s with the bolding step on a 220-word one |
| [`hf-dataset/README.md`](hf-dataset/README.md) | The dataset card, uploaded to the HF dataset repo |
| [`docs/data-format.md`](docs/data-format.md) | Pool and annotation record schemas |
| [`docs/paragraphing-style.md`](docs/paragraphing-style.md) | The paragraphing rules derived from the hand reviews, and how well code reproduces them |
| [`scripts/style_rules.py`](scripts/style_rules.py) | Rules 1–3 as code, plus a scorer against gold annotations. Applied to every `syn` proposal |
| [`scripts/extrapolate.py`](scripts/extrapolate.py) | Cross-validated check of Haiku-with-rules against the code rules (it lost; result in the docstring) |
| [`docs/training.md`](docs/training.md) | Training runs, base-model choice, commands |
| [`training/`](training/) | `train.py` (Modal) and the built train/test splits |
| [`project.json`](project.json) | Machine-readable project state |

## Deploying

```bash
export HF_TOKEN="$HUGGINGFACE_TOKEN"
uv run scripts/build_pool.py --upload                    # rebuild and push the pool
hf upload danielrosehill/ASR-Paragraph-Breaks hf-dataset/README.md README.md --repo-type dataset
hf upload danielrosehill/ASR-Paragraph-Annotator space . --repo-type space
hf upload danielrosehill/ASR-Paragraph-Breaks-Demo demo . --repo-type space
```

The annotation Space has no secrets. It saves with the signed-in user's OAuth token (`hf_oauth_scopes: write-repos`),
so only accounts with write access to the dataset can save.

## Related

- [ASR-Post-Processor-Model](https://github.com/danielrosehill/ASR-Post-Processor-Model): the broader
  light-touch cleanup model. Paragraph breaks are its taxonomy item `L1.paragraph-boundaries`, and its
  Phase 4 decides whether paragraphing is a separate model. This repo is that separate model.
  Its `docs/research.md` covers prior art: SaT, and Retkowski & Waibel,
  [arXiv 2512.24517](https://arxiv.org/abs/2512.24517) (TEDPara, YTSegPara, MiniSeg).
- [`retkowski/ytseg`](https://huggingface.co/datasets/retkowski/ytseg): Whisper YouTube transcripts with
  chapter boundaries. Those are topic segments, coarser than paragraphs; possible weak labels or an
  external check.
