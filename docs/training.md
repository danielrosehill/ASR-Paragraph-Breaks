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
| asr-paragraph-r2-gemma270m | google/gemma-3-270m-it | L4 | 233 s | 0.84 / 0.72 / 0.78 | 2 of 26 | 3 epochs, LR 5e-5; untrained base: 26 of 26 fallback |

Outputs: `runs/r2-gemma270m-metrics.json`, `runs/r2-gemma270m-pred_ft.jsonl`; the model is on the Modal
volume at `runs/asr-paragraph-r2-gemma270m/model` (not on the Hub).
