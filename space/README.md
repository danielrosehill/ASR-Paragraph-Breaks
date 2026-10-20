---
title: ASR Paragraph Annotator
emoji: 📝
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - write-repos
short_description: Hand-annotate paragraph breaks in flat ASR text
---

# ASR Paragraph Annotator

Shows one unparagraphed transcript at a time from
[`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks)
(`pool/pool.jsonl`). Press Enter where a paragraph should start and save. Only whitespace may change.
Each save is committed as `annotations/<id>.jsonl` with the signed-in user's own OAuth token, so saving
works only for accounts with write access to the dataset.

Source: [github.com/danielrosehill/ASR-Paragraph-Breaks](https://github.com/danielrosehill/ASR-Paragraph-Breaks) (`space/`).
