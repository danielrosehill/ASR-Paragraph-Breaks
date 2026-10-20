# ASR Paragraph Breaks

A small text-in/text-out model that adds paragraph breaks to flat ASR output and changes nothing else.

Speech-to-text output is punctuated and cased but arrives as one block, even from engines that mark
sentence boundaries. This project hand-annotates where the paragraphs go and fine-tunes a small model on
those annotations.

**Status (2026-10-21): annotating.** The pool is built (959 items, about 368k words), and the annotation
Space and dataset are live. No annotations or model yet.

| | |
|---|---|
| Dataset | [`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks) |
| Annotation Space | [`danielrosehill/ASR-Paragraph-Annotator`](https://huggingface.co/spaces/danielrosehill/ASR-Paragraph-Annotator) |
| Model | not trained yet |

## Why hand annotation

The cheap way to get training data is to take paragraphed text, flatten it and learn to restore the
breaks. That learns the conventions of written text (edited articles, Wikipedia), not how someone wants a
dictated monologue broken up. Paragraphing is a judgement call, so the breaks here are placed by hand on
real ASR monologue that was never paragraphed.

## Why a generative model, given SaT exists

[SaT / wtpsplit](https://github.com/segment-any-text/wtpsplit) (`segment-any-text/sat-3l-sm`) already
does paragraph segmentation by predicting newline probability, and it is the baseline to beat. The goal
here is a plain text-in/text-out model trained on one person's preferences. Every output is checked
against the input: if anything other than whitespace changed, fall back to the input.

## Layout

| Path | What |
|---|---|
| [`scripts/build_pool.py`](scripts/build_pool.py) | Builds `pool/pool.jsonl` from public HF sources; `--upload` pushes it. Sources and filters are in its docstring |
| [`space/`](space/) | The Gradio annotation Space (deployed to HF as-is) |
| [`hf-dataset/README.md`](hf-dataset/README.md) | The dataset card, uploaded to the HF dataset repo |
| [`docs/data-format.md`](docs/data-format.md) | Pool and annotation record schemas |
| [`project.json`](project.json) | Machine-readable project state |

## Deploying

```bash
export HF_TOKEN="$HUGGINGFACE_TOKEN"
uv run scripts/build_pool.py --upload                    # rebuild and push the pool
hf upload danielrosehill/ASR-Paragraph-Breaks hf-dataset/README.md README.md --repo-type dataset
hf upload danielrosehill/ASR-Paragraph-Annotator space . --repo-type space
```

The Space has no secrets. It saves with the signed-in user's OAuth token (`hf_oauth_scopes: write-repos`),
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
