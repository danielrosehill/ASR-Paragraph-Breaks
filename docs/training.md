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

Per-run `metrics.json` and `pred_ft.jsonl` are in `runs/<run>/`.

## Where the weights are (2026-10-21)

| Copy | Location |
|---|---|
| Hub, PyTorch (public) | [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks) (r2) |
| Hub, ONNX (public) | [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX) (r2) |
| Archive (private) | `wasabi:small-models/ai-training/runs/<run>/` for r2 (model, onnx, metrics, predictions) and r3 (model, metrics, predictions), each with `SHA256SUMS`; `rclone check` 0 differences. Procedure: skill `voice-ops:wasabi-archive` section 6 |
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
