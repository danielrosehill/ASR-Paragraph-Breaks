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
# Features are declared so a file missing an optional field loads as null instead of breaking the
# viewer (CastError, 2026-10-25). Keep in step with docs/data-format.md and space/app.py.
dataset_info:
  - config_name: annotations
    features:
      - name: id
        dtype: string
      - name: source
        dtype: string
      - name: source_id
        dtype: string
      - name: text
        dtype: string
      - name: paragraphed
        dtype: string
      - name: proposed
        dtype: string
      - name: proposal_model
        dtype: string
      - name: proposal_method
        dtype: string
      - name: edited
        dtype: bool
      - name: paragraphs
        sequence: string
      - name: break_offsets
        sequence: int64
      - name: n_paragraphs
        dtype: int64
      - name: words
        dtype: int64
      - name: annotator
        dtype: string
      - name: annotated_at
        dtype: string
  - config_name: pool
    features:
      - name: id
        dtype: string
      - name: source
        dtype: string
      - name: source_id
        dtype: string
      - name: chunk
        dtype: string
      - name: text
        dtype: string
      - name: words
        dtype: int64
      - name: proposed
        dtype: string
      - name: haiku_proposed
        dtype: string
      - name: proposal_model
        dtype: string
      - name: proposal_method
        dtype: string
      - name: audience
        dtype: string
      - name: style
        dtype: string
      # class_label, not int64: the viewer's histogram fails on an int column holding only 0/1
      # (StatisticsComputationError, 2026-10-25). Stored values are still the ints 0 and 1.
      - name: priority
        dtype:
          class_label:
            names:
              '0': normal
              '1': golden
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

Paragraphing is a judgement call, so every break is checked by a person. Synthetic items (`syn`) arrive
with the generating model's breaks and are reviewed and corrected in the Space; the `edited` field says
whether the annotator changed them. Real ASR items arrive flat and are paragraphed from scratch. Most items come from one annotator, so they reflect one
person's preference: short paragraphs, a break at each change of subject.

## Configs

| Config | File(s) | What |
|---|---|---|
| `annotations` (default) | `annotations/<id>.jsonl` | Finished annotations, one file per item |
| `pool` | `pool/pool.jsonl` | All flat source items, annotated or not |

Field reference: [`docs/data-format.md`](https://github.com/danielrosehill/ASR-Paragraph-Breaks/blob/main/docs/data-format.md).
Training pair: `text` → `paragraphed`.

`training/r2/{train,test}.jsonl` is the exact split behind the published model
[`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks):
505 training pairs (20 hand-reviewed, 485 rule-labelled, `label` field says which) and 26 held-out
hand-reviewed test pairs. Rows are `{id, label, input, output}`.

## Sources

| Key | Upstream | Licence | What |
|---|---|---|---|
| `syn` | [`scripts/generate_synthetic.py`](https://github.com/danielrosehill/ASR-Paragraph-Breaks/blob/main/scripts/generate_synthetic.py) | CC-BY-4.0 | Synthetic voice-memo transcripts written and paragraphed by Claude Haiku 5.5, then reviewed and corrected by hand |
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
