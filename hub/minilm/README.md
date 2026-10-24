---
license: mit
base_model: microsoft/MiniLM-L12-H384-uncased
language:
  - en
pipeline_tag: token-classification
library_name: transformers
datasets:
  - danielrosehill/ASR-Paragraph-Breaks
tags:
  - asr-post-processing
  - paragraph-segmentation
  - dictation
  - voice-notes
  - on-device
  - onnx
---

# MiniLM-L12: ASR paragraph breaks

Adds paragraph breaks to flat speech-to-text output. Speech recognition returns punctuated, cased text
as one block; this model decides, word by word, where a new paragraph starts. It is a classifier, not a
generator: the output is the input words joined with a space or a blank line, so it **cannot change the
text** and needs no fallback check.

- 33.2M parameters (MiniLM-L12-H384, WordPiece, uncased), fine-tuned for word-level token classification.
- `onnx/model_int8.onnx` is 32 MB and runs one forward pass per 510-token window on ONNX Runtime 1.17+.
- Successor to the generative [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks)
  (437 MB int8, 23-48 s per memo on CPU), built for the on-device dictation apps.

## Results

Break-position precision, recall and F1 on the same 26 held-out synthetic voice memos (290 breaks,
hand-reviewed) as the Gemma model, so the numbers compare directly. Threshold and epoch were chosen on a
separate dev split, never on test.

| Model | Precision | Recall | F1 | Fallbacks |
|---|---|---|---|---|
| SaT `segment-any-text/sat-3l-sm` (paragraph mode) | 0.48 | 1.00 | 0.65 | – |
| Gemma 3 270M fine-tune (r2) | 0.84 | 0.72 | 0.78 | 2 of 26 |
| **This model, PyTorch / `onnx/model.onnx`** | **0.86** | **0.84** | **0.85** | none possible |
| This model, `onnx/model_int8.onnx` | 0.86 | 0.83 | 0.85 | none possible |

Mean over 3 training seeds: F1 0.85. On 10 harder hand-written cases (fused greetings and sign-offs,
lowercase after a period, dropped commas, fillers; 36 breaks), F1 0.83 (int8 0.81); that set was also
used to pick this run over a tied one, so it is not a clean held-out figure.

`model.onnx` (fp32) makes the same break decision as PyTorch on all 8,824 test words; `model_int8.onnx`
differs on 33. Latency on a laptop CPU (Ryzen 7 5700U, 1 thread, ONNX Runtime 1.31): int8 0.65 s mean
for a 340-word memo; phone latency not measured yet.

## Files

| Path | What |
|---|---|
| `model.safetensors`, `config.json`, `tokenizer.json` | PyTorch weights (`BertForTokenClassification`, labels `O` / `PARA`) |
| `onnx/model_int8.onnx` | int8 dynamic quantization, 32 MB (the one an app ships) |
| `onnx/model.onnx` | fp32, 127 MB |
| `onnx/vocab.txt`, `onnx/tokenizer.json` | WordPiece vocabulary (line number = token id), lower-cased |
| `onnx/config.json` | `paragraph_threshold` (0.26) |
| `paragraphs.py` | Reference inference with ONNX Runtime: windowing, merge, output |

ONNX inputs `input_ids` and `attention_mask` (int64 `[batch, seq]`), output `logits` (`[batch, seq, 2]`).

## How it works

1. Split the text on whitespace into words, and tokenize each word separately (WordPiece, lower-cased).
2. Cut windows of at most 510 tokens at word boundaries, each starting about 256 tokens after the last;
   wrap each in `[CLS]` … `[SEP]`.
3. A word's score is softmax(logits)[1] at its first sub-token, taken from the window where that
   token is closest to the middle.
4. A paragraph starts at every word after the first with P >= `paragraph_threshold`; join with a blank
   line there and a space elsewhere.

Breaks can fall after a comma, so "Hi John, just following up…" can become "Hi John," + new paragraph.
Capitalisation is not a cue (the tokenizer lower-cases); punctuation is.

## Usage

```bash
uv run https://huggingface.co/danielrosehill/MiniLM-L12-ASR-Paragraph-Breaks/resolve/main/paragraphs.py "flat transcript ..."
```

or in Python (downloads `onnx/` only):

```python
from paragraphs import load, paragraphs
model = load()                     # load(fp32=True) for onnx/model.onnx
print(paragraphs(flat_text, *model))
```

The PyTorch weights load with `AutoModelForTokenClassification`; use the same windowing to score texts
over 510 tokens.

## Training

- Data: [`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks).
  Synthetic spoken-register memos written by Claude Haiku 5.5; 20 hand-reviewed and 784 labelled by
  rules derived from those reviews (style rules v1.2). 30% of rows are added again with ASR noise
  (lowercase after a period, dropped commas, inserted or removed fillers), and some of those also with
  a greeting or sign-off fused into the next sentence. 936 training rows, 80 dev.
- Label 1 = a paragraph starts at this word (first sub-token only). Class weight 3 on breaks (about 3%
  of words), 10 epochs, LR 8e-5, batch 16, AdamW with 6% warmup, best dev epoch of 3 seeds kept.
  About 140 s per seed on one L4.
- Code and full run table: [ASR-Paragraph-Breaks](https://github.com/danielrosehill/ASR-Paragraph-Breaks)
  (`training/train_tokcls.py`, `docs/training.md`, run `asr-paragraph-r5b-minilm-l12`).

## Limitations

- English only, trained and tested on synthetic memos in one person's paragraphing style; the 26 test
  memos are a small set (one break moves F1 by about 0.003).
- Full context: it reads the whole text, so it is meant to run once after transcription, not while
  dictating.
- It still keeps "Hi John," joined to the next sentence in some memos, and can break after a period
  followed by a lower-case word.
