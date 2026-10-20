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
   **3b (v1.1, 2026-10-21):** once a wrap-up line appears near the end ("That's it.", "Okay, I think
   that's the core of it.", "Thanks again."), every sentence after it stands alone, however long
   ("I'll send Marisol a text tonight with the photos…").
4. **Messages addressed to a person split somewhat further.** The two most letter-like messages
   (`syn-1-0314`, `syn-1-0352`) went close to one sentence per paragraph through the body. Most other
   messages did not, and memos to self keep multi-sentence body paragraphs: in `syn-1-0189` (a weekly
   plan), only the opener and close changed. Overstating this rule made an LLM over-split (below).
5. **Breaks are never removed,** so a proposal that is already correct is never over-split.

## How well the rules reproduce it

`scripts/style_rules.py` implements rules 1–3 as code on top of a proposal, with an abbreviation guard
so "Mr. Delgado" is not split. Thresholds: a signpost is a first sentence of up to 11 words, and closing
lines are up to 10 words. Break-position precision/recall:

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Haiku proposal alone | 1.00 | 0.54 | 0.70 |
| Proposal + code rules 1–3 | 0.83 | 0.84 | 0.84 |
| Haiku 5.5 given rules 1–5 + gold examples (4-fold) | 0.59 | 0.95 | 0.73 |

The code-rule thresholds were fitted on these same 19 items, so 0.84 is optimistic. The LLM figures are
cross-validated (`scripts/extrapolate.py eval`). The two tie on notes to self; on memos to someone else
the LLM over-splits (precision 0.50), following rule 4 too literally.

**Decision (2026-10-21):** every `syn` proposal in the pool is Haiku's breaks plus the code rules
(`proposal_method: "haiku-5.5 + style_rules v1"`, now v1.1). It costs nothing, is deterministic, and
is the better of the two methods measured.

## Check on the golden set (26 of 30 `syn-2-*` reviewed, 2026-10-21)

These memos use eight speaking styles the rules were not fitted on. Daniel saved 13 of 26 unchanged,
removed no rule-added break, and added 23 breaks in total. Scored from Haiku's original proposal
(`python3 scripts/style_rules.py score <annotations dir>`):

| Rules | `syn-1` (19, fitted on) | `syn-2` (26, held out) | All 45 |
|---|---|---|---|
| none (Haiku proposal) | 0.70 | 0.64 | |
| v1 (rules 1–3) | 0.84 | 0.95 | |
| v1.1 (+ rule 3b) | 0.85 | 0.96 | 0.91 |

`syn-2` precision is 1.00, but these were reviewed with the v1 output on screen, so some of that is
anchoring. What was still added: closing lines over the 10-word limit (fixed by 3b) and spoken
enumerations ("One, …" / "Two, …"). An enumeration rule was tested and dropped: F1 unchanged, precision
slightly lower.
