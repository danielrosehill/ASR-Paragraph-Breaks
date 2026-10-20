# Paragraphing style (derived from Daniel's reviews)

Derived 2026-10-21 from the first 19 reviewed `syn` items (gold annotations `syn-1-0032` … `syn-1-0482`
in the HF dataset). Starting from Haiku 5.5's proposals, Daniel **kept all 108 proposed breaks, removed
none and added 91**. Every added break is at a sentence boundary, or after a greeting/sign-off comma.

## The rules

A memo has three parts: an opener, a body and a close. The body is left largely as Haiku proposed it;
almost all of the edits are at the edges and at signposts.

1. **The opener stands alone.** The first sentence (greeting, "it's me", or the framing line, as in
   "Okay, so this is for Marcus…") is its own paragraph. The edits did this in nearly every item.
2. **A signpost is split from its detail.** When a paragraph opens with a short announcing sentence
   ("The second part is the server.", "Here's what happened.", "So the plan is this.",
   "I asked about the cost, because that's always the question."), the detail after it becomes a new
   paragraph. The split is usually not made when the opening sentence already carries content.
3. **Each closing line stands alone.** Wrap-ups, afterthoughts and sign-offs are peeled off one per
   paragraph: "That's it.", "Oh, and she mentioned…", "Let me know by Friday.", "Thanks, Dana.",
   "Talk soon,", then the name, "Bye."
4. **Messages addressed to a person split further.** Dictated emails and voice messages to someone
   (`syn-1-0314`, `syn-1-0352`, `syn-1-0121`, `syn-1-0304`) go close to one sentence per paragraph through
   the body. Memos to self keep multi-sentence body paragraphs: in `syn-1-0189` (a weekly plan),
   only the opener and close changed.
5. **Breaks are never removed,** so a proposal that is already correct is never over-split.

## How well simple rules reproduce it

`scripts/style_rules.py` implements rules 1–3 as code on top of a proposal. Thresholds: a signpost is a
first sentence of up to 11 words, and closing lines are up to 10 words. Scored on the same 19 items, as
break-position precision/recall:

| | Precision | Recall | F1 |
|---|---|---|---|
| Haiku proposal alone | 1.00 | 0.54 | 0.70 |
| Proposal + rules 1–3 | 0.82 | 0.83 | 0.83 |

The two thresholds were fitted on these same 19 items, so 0.83 is optimistic. Most of the remaining
recall gap is rule 4, which the code does not implement (it needs to know who the memo is addressed to).
