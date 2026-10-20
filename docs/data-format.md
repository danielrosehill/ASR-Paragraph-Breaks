# Data format

Dataset repo: [`danielrosehill/ASR-Paragraph-Breaks`](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks).

## `pool/pool.jsonl`: text waiting to be annotated

Built by `scripts/build_pool.py`. One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable id, `<source>-<source_id>[-c<n>]`. Rebuilding the pool keeps ids, so saved annotations stay attached |
| `source` | string | Source key: `mwp`, `svn` or `nvn` (listed in the `scripts/build_pool.py` docstring) |
| `source_id` | string | Row id or index in the upstream dataset |
| `chunk` | string | `i/n` when a long transcript was split on sentence boundaries |
| `text` | string | Flat text: punctuated, cased, every run of whitespace collapsed to one space, no line breaks |
| `words` | int | Word count of `text` |

## `annotations/<id>.jsonl`: one hand annotation

Written by the Space (`space/app.py`), one file per item, so concurrent saves never conflict.

| Field | Type | Meaning |
|---|---|---|
| `id`, `source`, `source_id` | string | Copied from the pool item |
| `text` | string | Flat input, identical to the pool `text` |
| `paragraphed` | string | Target output: the same text with paragraphs joined by `\n\n` |
| `paragraphs` | list[string] | The paragraphs, in order |
| `break_offsets` | list[int] | Character offsets into `text` where each paragraph after the first starts |
| `n_paragraphs` | int | `len(paragraphs)` |
| `words` | int | Word count |
| `annotator` | string | HF username that saved it |
| `annotated_at` | string | UTC ISO 8601 timestamp |

Invariant, enforced at save time: `paragraphed` with all whitespace removed equals `text` with all
whitespace removed. Only paragraph breaks are learned; words and punctuation are never edited.

The training pair is `text` → `paragraphed`.
