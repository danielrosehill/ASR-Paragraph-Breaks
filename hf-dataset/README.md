---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
tags:
  - asr-post-processing
  - paragraph-segmentation
  - dictation
  - voice-notes
  - text-segmentation
pretty_name: ASR Paragraph Breaks
size_categories:
  - n<1K
configs:
  - config_name: annotations
    default: true
    data_files: annotations/*.jsonl
  - config_name: pool
    data_files: pool/pool.jsonl
---

# ASR Paragraph Breaks

Hand-annotated paragraph breaks for flat speech-to-text output. ASR engines emit punctuated, cased text
as one block. This dataset pairs that block (`text`) with the same text split into paragraphs by a person
(`paragraphed`). Nothing else changes: words, punctuation and casing are identical, and the annotation tool
refuses any save that alters them. It is training data for a small text-in/text-out model that does only
this one job.

Paragraphing is a judgement call, so the breaks are hand-placed rather than recovered from
already-paragraphed text that was flattened. Most items come from one annotator, so they reflect one
person's preference: short paragraphs, a break at each change of subject.

## Configs

| Config | File(s) | What |
|---|---|---|
| `annotations` (default) | `annotations/<id>.jsonl` | Finished annotations, one file per item |
| `pool` | `pool/pool.jsonl` | All flat source items, annotated or not |

Field reference: [`docs/data-format.md`](https://github.com/danielrosehill/ASR-Paragraph-Breaks/blob/main/docs/data-format.md).
Training pair: `text` → `paragraphed`.

## Sources

| Key | Upstream | Licence | What |
|---|---|---|---|
| `mwp` | [`My-Weird-Prompts/episodes`](https://huggingface.co/datasets/My-Weird-Prompts/episodes) `prompt_sanitized` | CC-BY-4.0 | Dictated podcast prompts, 300+ words, real ASR |
| `svn` | [`danielrosehill/STT-Voice-Notes-Evals`](https://huggingface.co/datasets/danielrosehill/STT-Voice-Notes-Evals) whisper-1 transcripts | Apache-2.0 | LLM-written voice memos, read aloud, transcribed by whisper-1 |
| `nvn` | [`danielrosehill/Noisy-Voice-Notes`](https://huggingface.co/datasets/danielrosehill/Noisy-Voice-Notes) | CC-BY-4.0 | Real voice notes recorded in noisy settings |

Items over 600 words are split on sentence boundaries into chunks of about 350 words. The only edit
made to source text is whitespace: runs are collapsed, and a missing space after a full stop
(`me.I`, left by ASR segment joins) is restored.

## Annotating

Annotation happens in the Space
[`danielrosehill/ASR-Paragraph-Annotator`](https://huggingface.co/spaces/danielrosehill/ASR-Paragraph-Annotator).
Sign in, press Enter wherever a paragraph should start, and save. Saving needs write access to this repo.

Build scripts and the Space source: [github.com/danielrosehill/ASR-Paragraph-Breaks](https://github.com/danielrosehill/ASR-Paragraph-Breaks).
