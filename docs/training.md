# Training

Runs go to the Modal app `ai-training` (`~/repos/github/ai-models/Modal-AI-Training`, workspace
`danielrosehill`). Training uses the original PyTorch weights from the Hub; ONNX is an export step
afterwards, for on-device use next to Parakeet on the same runtime (ONNX Runtime). Never train from an
ONNX export.

```bash
python3 scripts/build_training.py          # training/train.jsonl + test.jsonl (gold read live from HF)
export MODAL_PROFILE=danielrosehill
cd ~/repos/github/ai-models/Modal-AI-Training
modal run --detach app.py --upload ~/repos/github/audio-dictation/ASR-Paragraph-Breaks/training \
  --cmd "BASE=google/gemma-3-270m-it LR=5e-5 python train.py" --run asr-paragraph-r2-gemma270m --gpu L4
modal volume get ai-training runs/asr-paragraph-r2-gemma270m/metrics.json runs/ --force
```

Split (2026-10-21): train 505 = 20 hand-reviewed + 485 silver (style_rules v1.1 on Haiku proposals);
test = the 26 reviewed golden `syn-2-*` items. The model sees the **flat** text; the rules' 0.96 on the
same test set starts from Haiku's paragraphed proposal, so it is a ceiling reference, not a like-for-like
baseline. An output that changes any non-whitespace text counts as a fallback (input returned flat).

## Base model choice

Target is on-device inference after Parakeet on a phone, so sub-1B. Checked 2026-10-21: Gemma 3 270M
(Google, built for on-device task fine-tunes; `onnx-community/gemma-3-270m-it-ONNX` shows the export
path) over Qwen3-0.6B (752M, too heavy), LFM2-350M and SmolLM2. NVIDIA's smallest current LM is
Nemotron 3 Nano 4B, so there is no NVIDIA option at this size; matching Parakeet's vendor buys nothing,
since on the phone Parakeet runs as ONNX, not NeMo. Fallback if generation is too slow on the phone:
a ~70M encoder that classifies break positions (single ONNX pass, cannot alter text).

## Runs

| Run | Base | GPU | Train time | Test P / R / F1 | Fallback | Notes |
|---|---|---|---|---|---|---|
| asr-paragraph-smoke | Qwen3-0.6B | L4 | 8 s | – | – | 16 rows, 1 epoch; pipeline check only |
| asr-paragraph-r1 | Qwen3-0.6B | L4 | – | – | – | Stopped by request (2026-10-21) in favour of the smaller Gemma |
| asr-paragraph-r2-gemma270m | google/gemma-3-270m-it | L4 | 233 s | 0.84 / 0.72 / 0.78 | 2 of 26 | 3 epochs, LR 5e-5, 505 train; untrained base: 26 of 26 fallback. **Published** |
| asr-paragraph-r3-gemma270m-e6 | google/gemma-3-270m-it | L4 | 725 s | 0.83 / 0.72 / 0.78 | 1 of 26 | 6 epochs, LR 5e-5, 804 train (adds 299 seed-3 emails as silver). Tied with r2; not published |
| asr-paragraph-r5-electra-small-r3data | google/electra-small-discriminator (13.5M) | L4 | 96 s/seed | 0.84 / 0.79 / 0.81 | 0 by construction | Encoder classifier (below) on the r3 train set (v1.1, no augmentation); seed mean F1 0.80; extra 0.78 |
| asr-paragraph-r5-electra-small | google/electra-small-discriminator (13.5M) | L4 | 101 s/seed | 0.84 / 0.79 / 0.81 | 0 | r5 set (v1.2 + noise copies); seed mean F1 0.81; extra 0.81 |
| asr-paragraph-r5-minilm-l12 | microsoft/MiniLM-L12-H384-uncased (33.2M) | L4 | 138 s/seed | 0.89 / 0.82 / 0.85 | 0 | r5 set; seed mean F1 0.84; extra 0.78 |
| asr-paragraph-r5b-electra-small | google/electra-small-discriminator (13.5M) | L4 | 79 s/seed | 0.83 / 0.81 / 0.82 | 0 | r5b set (r5 + fused greeting/valediction copies); seed mean F1 0.80; extra 0.84 |
| **asr-paragraph-r5b-minilm-l12** | microsoft/MiniLM-L12-H384-uncased (33.2M) | L4 | 142 s/seed | **0.86 / 0.84 / 0.85** | 0 | r5b set; seed mean F1 0.85; extra 0.83. **Best arm**; ONNX int8 32 MB, F1 0.85. Not published |
| asr-paragraph-r5-electra-small-live | google/electra-small-discriminator | L4 | ~9 min/seed | 0.82 / 0.79 / 0.80 | 0 | Live variant (no right context past the current sentence), r5 set, 5 epochs; seed mean F1 0.79 (full context 0.81); extra 0.79 |
| asr-paragraph-r5-minilm-l12-live | microsoft/MiniLM-L12-H384-uncased | L4 | ~13.5 min/seed | 0.83 / 0.79 / 0.81 | 0 | Live variant, r5 set, 5 epochs; seed mean F1 0.81 (full context 0.84); extra 0.71 |
| sat-3l-sm-baseline | segment-any-text/sat-3l-sm | laptop CPU | – | 0.48 / 1.00 / 0.65 | 0 | wtpsplit 2.2.2, `do_paragraph_segmentation=True`; same at threshold 0.5 and dev-tuned 0.35. sat-3l-sm has one boundary head, so its paragraphs are its sentences: it breaks after almost every sentence. extra 0.68 |

Per-run `metrics.json` and `pred_ft.jsonl` are in `runs/<run>/` (r5 also `pred_extra.jsonl`, with per-word
probabilities). r5 rows: P/R/F1 are the seed picked by dev F1 (never by test), "extra" is break F1 on
`training/test_extra.jsonl` with the corrected labels (see below; the r5 runs' own `metrics.json` scored extra
before that correction, the figures here are rescored from their `pred_extra.jsonl`). Train time is per seed
including per-epoch dev evaluation; each r5 run trains 3 seeds and the whole job took 4-8 min wall time
(live: 29 / 42 min).

## Encoder classifier (r5, 2026-10-24)

Why: the Hebrish apps need a paragraph step that is small, fast, cannot alter the text and runs on the
ONNX Runtime both apps already ship. The r2 Gemma is 437 MB int8 and generative (23-48 s per memo on CPU).

**Formulation.** Word-level token classification. The flat text is split on whitespace; label 1 = a
paragraph starts at this word (every paragraph after the first, from the target's breaks, equivalently
`break_offsets`), else 0. Word-level, so breaks after a comma (greeting, valediction) are possible. Only
the first WordPiece sub-token of each word is labelled (the rest are -100). Output is the input words
joined with a space, or with a blank line before a word whose P(break) >= the threshold, so the text
cannot change and there are no fallbacks. Scoring is the same as r2 (offsets into the whitespace-stripped
text, exact match), so the F1s compare directly. Code: `training/train_tokcls.py`.

- **Bases:** both WordPiece, both uncased (lower-cased by the tokenizer, so capitalisation is not a cue;
  punctuation is). `microsoft/MiniLM-L12-H384-uncased` loaded with its fast BERT tokenizer without problems,
  so no substitute was needed.
- **Long texts:** windows of up to 510 content tokens + [CLS]/[SEP], cut at word boundaries, stepping 256
  tokens; each word takes its probability from the window where it is most central. Test memos run 2-3
  windows; the 10 extra cases fit in one.
- **Imbalance** (about 3% of words start a paragraph): class weight 3 on the positive class, and the
  threshold is tuned for F1 on a dev split of 80 source memos carved from train (seed 42; their noise copies
  are dropped from training; dev scored on clean rows only). 3 seeds x 10 epochs, best dev epoch kept, the
  seed with the best dev F1 is saved with `paragraph_threshold` in its `config.json`.
  LR 2e-4 (ELECTRA), 8e-5 (MiniLM), batch 16, AdamW, linear warmup 6%.
- **Live variant** (`LIVE=1`): for each sentence, the input is the text up to the end of that sentence
  (left-truncated to 510 tokens) and only that sentence's words are scored, so the app can decide a break as
  each dictated sentence arrives. Sentence end = a word ending in `.!?` (plus closing quotes/brackets).
  About 25 examples per memo instead of 1-2, so 5 epochs.

**Training data** (`scripts/build_training.py`, same split as r2/r3: 20 gold + 784 silver, test = the 26
golden `syn-2-*`):

- r5: silver relabelled with style_rules **v1.2** (rules 1b and 3c, `docs/paragraphing-style.md`), plus
  232 ASR-noise copies (30% of rows; seeded): the word after 1-2 sentence-final periods lower-cased, 1-3
  commas dropped, fillers um/so/like inserted, um/uh removed. Breaks unchanged, test never augmented. 1,036
  rows (936 train + 80 dev after the dev carve).
- r5b (the current `training/train.jsonl`): as r5, but a quarter of the noise copies also get a fused
  greeting ("Hi John," / "just following up ...") and/or a valediction fused into the last sentence with the
  name split off ("Thanks again, all the best," / "Daniel."), the two shapes rules 1b/3c target and the
  silver data barely contains (37 comma-position first-paragraph breaks in 1,036 rows before this).
- The r3 train set (v1.1, no noise) is `training/train.jsonl` at commit `fb78a6e`; the `-r3data` run uses it.

**Extra test cases** (`training/test_extra.jsonl`, 10 memos, 36 breaks, never trained on): Daniel's
2026-10-21 demo failure (the "Hi John" memo, with the output he wanted) and 9 hand-written cases for fused
greetings, inline sign-offs, lowercase after a period, dropped commas and fillers, plus two that must *not*
split ("Hi Mr. Cohen, this is Yael ...", "Thanks, Ben."). Labelled by Claude from the documented rules, not
reviewed by Daniel. Two labels were corrected after the first r5 runs to follow rule 2 (a break after the
signposts "The other thing is the insurance renewal." and "The bigger question is the print run."); the
model had made those breaks. With 36 breaks, one break is about 2-3 F1 points: treat extra as a smoke test.

**Results** (table above). Every encoder arm beats r2's 0.78 on the same 26 memos, with no fallbacks. MiniLM-L12
is about 0.03-0.04 F1 above ELECTRA-small in every pairing. v1.2 relabelling plus noise copies left test F1
unchanged for ELECTRA (seed means 0.80 to 0.81) and moved extra from 0.78 to 0.81; the fused-greeting copies
(r5b) lifted extra for MiniLM from 0.78 to 0.83 at the same test F1 (0.85). On the "Hi John" memo r5b-MiniLM
now splits "Great hear from you, all the best," / "Daniel." as Daniel wanted, but still keeps "Hi John,"
fused and still breaks at "that. to make sure" (period + lowercase), as r2 did. On the other fused-greeting
cases ("Hey Sarah,", "Hi Rachel,") it splits the greeting; it keeps "Hi Mr. Cohen, this is Yael ..." and
"Thanks, Ben." together, as intended. The live variant costs 0.02-0.03 F1 on test (seed means: ELECTRA 0.79 vs 0.81, MiniLM 0.81 vs 0.84) and more on extra (MiniLM 0.71 vs 0.78), where the sign-off breaks need the next sentence; it trains about 6x slower.

Choice of best arm: r5b-MiniLM-L12 and r5-MiniLM-L12 tie on dev (0.881 / 0.880) and test (0.85); r5b was
taken for its better extra score. That uses the extra set for a choice, so its extra figure is not a clean
held-out number.

### ONNX export (r5b-minilm-l12)

`training/export_tokcls_onnx.py runs/<run>` on the laptop CPU (torch 2.14.1, TorchScript exporter,
`dynamo=False`, opset 14, so ONNX Runtime 1.17+ reads it), then `quantize_dynamic` int8 weights. Inputs
`input_ids`, `attention_mask` (int64 [batch, seq]); output `logits` [batch, seq, 2]. `onnx/` also holds
`vocab.txt` (WordPiece, line = id), `tokenizer.json` and `config.json` (`paragraph_threshold` 0.26). Checked
with onnxruntime 1.31.0, CPU provider, on all 26 test + 10 extra memos through the full window/merge pipeline,
on an AMD Ryzen 7 5700U laptop that other sessions were also using (latency is indicative only):

| File | Size | Test P / R / F1 | Break decisions differing from PyTorch | ms per test memo (1 thread, mean / max) | ms per extra memo (1 thread) |
|---|---|---|---|---|---|
| PyTorch | 127 MB (safetensors) | 0.86 / 0.84 / 0.85 | – | – | – |
| `model.onnx` (fp32) | 127 MB | 0.86 / 0.84 / 0.85 | 0 of 8,824 words (max prob diff 1e-5) | 955 / 2,330 | 59 |
| `model_int8.onnx` | 32 MB | 0.86 / 0.83 / 0.85 | 33 of 8,824 | 648 / 1,281 | 52 |

Default ONNX Runtime threading was not faster than 1 thread on this loaded machine (`onnx_check.json` has
both). A test memo averages 340 words and 2-3 windows; with STRIDE 384 or one window per short memo it would
be fewer passes. Weights: Modal volume `ai-training`, `runs/<run>/model` for every r5 run; the best arm's
PyTorch and ONNX files are also in `runs/asr-paragraph-r5b-minilm-l12/{model,onnx}/` on Daniel's laptop
(git-ignored). Nothing was published to the Hub.

```bash
python3 scripts/build_training.py                 # r5b set; --no-aug for none
export MODAL_PROFILE=danielrosehill; cd ~/repos/github/ai-models/Modal-AI-Training
modal run --detach app.py --upload <dir with train_tokcls.py, train.jsonl, test.jsonl, test_extra.jsonl> \
  --cmd "BASE=microsoft/MiniLM-L12-H384-uncased LR=8e-5 python train_tokcls.py" --run asr-paragraph-r5b-minilm-l12 --gpu L4
#   ELECTRA: BASE=google/electra-small-discriminator LR=2e-4; live: add LIVE=1 EPOCHS=5; r3 data: TRAIN=train_r3.jsonl
modal volume get ai-training runs/asr-paragraph-r5b-minilm-l12/model runs/asr-paragraph-r5b-minilm-l12/
uv run --with torch --with transformers --with onnx --with onnxruntime training/export_tokcls_onnx.py runs/asr-paragraph-r5b-minilm-l12
satv/bin/python scripts/sat_baseline.py > runs/sat-3l-sm-baseline/metrics.json   # SaT baseline, CPU
```

Cost (estimate, Modal L4 at about US$0.80/h plus CPU/memory): five full-context jobs, 30 min wall in total,
about US$0.40; two live jobs, 71 min wall, about US$1.00. Total about US$1.40.

## Where the weights are (2026-10-21)

| Copy | Location |
|---|---|
| Hub, PyTorch (public) | [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks) (r2) |
| Hub, ONNX (public) | [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX) (r2) |
| Archive (private) | `wasabi:hf-llms/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks/<YYYY-MM-DD>_<sha7>/` (latest HF revision, `SHA256SUMS` + `MODEL.json`, verified 2026-10-22; ONNX export not kept, it rebuilds). The per-run r2/r3 copies in `small-models` were retired with that bucket on 2026-10-22, and the Modal run weights deleted the same day |
| Modal volume `ai-training` | Weights and checkpoints deleted after the archive and Hub checks; small files kept |
| Exact split for r2 | dataset `danielrosehill/ASR-Paragraph-Breaks`, `training/r2/{train,test}.jsonl` |

The model card charts are built by `scripts/model_card.py runs/<published> runs/<other>...` into `hub/`.

## ONNX export (r2)

`training/export_onnx.py` on Modal (`--run asr-paragraph-r2-gemma270m --gpu T4`; it installs
optimum-onnx, which pins transformers<4.58, into its own venv; full command in the script's docstring
history and below). Exported with `optimum-cli export onnx --task text-generation-with-past`, then int8
dynamic quantization. Checked on 6 test memos with ONNX Runtime's CPU provider:

| File | Size | Same output as PyTorch | Text unchanged | Time per memo (cloud CPU) |
|---|---|---|---|---|
| `model.onnx` (fp32) | 1.66 GB | 6 of 6 | 6 of 6 | 48 s |
| `model_quantized.onnx` (int8 dynamic) | 0.42 GB | 2 of 6 | 5 of 6 | 23 s |

fp32 is faithful but large and slow; int8 dynamic quantization damages output. The file is larger than
270M × 4 bytes because Gemma's 262k-token vocabulary makes the embedding about 168M parameters. Next to
try for the phone: weight-only 4-bit (MatMulNBits) or fp16, and prompt-lookup decoding, since nearly
every output token copies the input.

```bash
modal run --detach app.py --upload ~/repos/github/audio-dictation/ASR-Paragraph-Breaks/training \
  --run asr-paragraph-r2-gemma270m --gpu T4 --cmd "export UV_CACHE_DIR=/tmp/uvc && uv venv -q /tmp/ex --python 3.12 \
  && uv pip install -q -p /tmp/ex torch --index-url https://download.pytorch.org/whl/cpu \
  && uv pip install -q -p /tmp/ex 'optimum-onnx[onnxruntime]' 'transformers<4.58' && PATH=/tmp/ex/bin:\$PATH python export_onnx.py"
```

Upload gotcha (2026-10-21): `hf upload` of the whole ONNX folder was killed with SIGTERM on the laptop
(memory pressure while hashing the 1.7 GB file); uploading each large file in its own `hf upload` call worked.

### What app integration (Hebrish backlog B-40 / B-40a) needs next

Status 2026-10-24: Hebrish apps 1.001 ship without paragraphs; integration moved to the next sprint.

1. Daniel's call on the arm: full-context r5b-MiniLM-L12 int8 (F1 0.85, 32 MB) runs once per memo after
   transcription; the live variant (F1 0.81) only if breaks must appear while dictating. Live was trained on
   r5 data only, so retrain it on r5b (`LIVE=1 EPOCHS=5`, about 42 min, US$0.60) if it is wanted.
2. Publish: `runs/asr-paragraph-r5b-minilm-l12/onnx/` to the Hub (model card from this section) and the CDN,
   once Daniel approves. Not done.
3. App side: WordPiece tokenizer (lower-case, BERT basic tokenisation, `vocab.txt`), the windowing and
   "most central window" merge from `train_tokcls.py` (`windows`, `word_probs`), threshold from
   `config.json`, then join words with a space or a blank line. Port and test against
   `runs/asr-paragraph-r5b-minilm-l12/pred_ft.jsonl` (per-word probabilities included) for parity.
4. Latency on a phone is unmeasured; the laptop figures above are 0.65 s mean for a 340-word memo on 1 thread.
