---
title: ASR Paragraph Breaks
emoji: 🎙️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: gemma
models:
  - danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks
  - danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding
datasets:
  - danielrosehill/ASR-Paragraph-Breaks
short_description: Add paragraph breaks to flat ASR text (Gemma 270M)
---

# ASR Paragraph Breaks demo

Runs [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks)
on CPU. Paste flat speech-to-text output and get it back with paragraph breaks; nothing else may change,
and any output that changes more than whitespace falls back to the input. Optionally chains
[`Gemma-3-270M-Email-Key-Line-Bolding`](https://huggingface.co/danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding)
to bold the key lines afterwards.

Source: [`demo/`](https://github.com/danielrosehill/ASR-Paragraph-Breaks/tree/main/demo).
