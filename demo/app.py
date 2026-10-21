"""Public demo: add paragraph breaks to flat ASR text with Gemma-3-270M-ASR-Paragraph-Breaks.

Optionally chains Gemma-3-270M-Email-Key-Line-Bolding afterwards, since the two run as a pipeline in
dictation. Both models run on the Space's CPU, greedy, with the system prompts they were trained on
(hub/model/README.md in each repo). Each output is validated the way it would be in deployment: the
paragraph model may only change whitespace, the bolding model may only add `**`; otherwise that step
falls back to its input.
"""
import html
import json
import re
import time
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

PARA_REPO = "danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks"
BOLD_REPO = "danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding"
PARA_SYSTEM = ("Add paragraph breaks to this voice-note transcript. Separate paragraphs with a blank line. "
               "Do not change, add or remove any word or punctuation mark.")
BOLD_SYSTEM = ("Put the key parts of this email in Markdown bold (**like this**): the requests, the purpose, "
               "the main points and the key facts. Do not change, add or remove anything else.")
MAX_WORDS = 600  # held-out memos are 176-566 words; longer inputs were not in training and are slow on CPU

torch.set_num_threads(max(1, torch.get_num_threads()))
MODELS = {}


def load(repo):
    if repo not in MODELS:
        MODELS[repo] = (AutoTokenizer.from_pretrained(repo),
                        AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval())
    return MODELS[repo]


load(PARA_REPO)
load(BOLD_REPO)


def stream(repo, system, text):
    """Yield the growing greedy output of `repo` for `text`."""
    tok, model = load(repo)
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    n_in = enc["input_ids"].shape[1]
    Thread(target=model.generate, kwargs=dict(**enc, streamer=streamer, do_sample=False,
                                               max_new_tokens=int(n_in * 1.3) + 64)).start()
    out = ""
    for piece in streamer:
        out += piece
        yield out


def render(text):
    """Paragraphed text (optionally with Markdown bold) -> HTML."""
    paras = []
    for para in re.split(r"\n\s*\n", text.strip()):
        parts = re.split(r"(\*\*)", para)
        out, inside = [], False
        for p in parts:
            if p == "**":
                out.append("</strong>" if inside else "<strong>")
                inside = not inside
            else:
                out.append(html.escape(p).replace("\n", "<br>"))
        if inside:
            out.append("</strong>")
        paras.append(f"<p>{''.join(out)}</p>")
    return f'<div class="memo">{"".join(paras)}</div>'


def n_paras(text):
    return len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])


def run(text, bold):
    text = " ".join((text or "").split())  # the model is trained on flat, single-block input
    if not text:
        yield "", "<p class='note'>Paste a transcript first.</p>", ""
        return
    if len(text.split()) > MAX_WORDS:
        yield "", f"<p class='note'>Please keep it under {MAX_WORDS} words (this runs on a small CPU).</p>", ""
        return
    t0 = time.time()
    pred = text
    for pred in stream(PARA_REPO, PARA_SYSTEM, text):
        yield pred, render(pred), "Adding paragraph breaks…"
    pred = pred.strip()
    notes = []
    if re.sub(r"\s+", "", pred) == re.sub(r"\s+", "", text):
        paragraphed = pred
        notes.append(f"{n_paras(pred)} paragraphs in {time.time() - t0:.0f} s.")
    else:
        paragraphed = text
        notes.append("The model changed more than whitespace, so its output was discarded and the input is "
                     "shown unchanged (this is the fallback a real pipeline would use).")
    yield paragraphed, render(paragraphed), " ".join(notes)
    if not bold:
        return
    t1 = time.time()
    marked = paragraphed
    for marked in stream(BOLD_REPO, BOLD_SYSTEM, paragraphed):
        yield marked, render(marked), " ".join(notes + ["Bolding key lines…"])
    marked = marked.strip()
    if marked.replace("**", "") != paragraphed or marked.count("**") % 2:
        notes.append("The bolding model changed more than the ** markers, so the text is shown unbolded.")
        marked = paragraphed
    elif "**" not in marked:
        notes.append("The bolding model chose not to bold anything.")
    else:
        notes.append(f"Bolded in {time.time() - t1:.0f} s.")
    yield marked, render(marked), " ".join(notes)


EXAMPLES = json.loads((Path(__file__).parent / "examples.json").read_text())

CSS = """
.memo { font-size: 15px; line-height: 1.55; max-width: 68ch; }
.memo p { margin: 0 0 0.9em; }
.note { opacity: 0.7; }
"""

INTRO = """
# ASR paragraph breaks

A 270M-parameter Gemma 3 fine-tune that adds paragraph breaks to flat speech-to-text output and changes
nothing else. Built for dictation pipelines: it runs after the ASR pass (here, a Parakeet fine-tune),
which returns punctuated, cased text as one block. It is sized to run on a phone.

Break F1 0.78 on 26 held-out, hand-reviewed voice memos (precision 0.84, recall 0.72), against 0.00 for
the untrained base model; 2 of 26 outputs fell back to the input. Every output is checked: if anything
other than whitespace changed, the input is returned instead. The examples are held-out synthetic memos
the model never saw. Runs on CPU, so allow 20-60 seconds.

[Model](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks) ·
[ONNX](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX) ·
[Dataset](https://huggingface.co/datasets/danielrosehill/ASR-Paragraph-Breaks) ·
[Code](https://github.com/danielrosehill/ASR-Paragraph-Breaks) ·
[Email key-line bolding demo](https://huggingface.co/spaces/danielrosehill/Email-Key-Line-Bolding-Demo)
"""

with gr.Blocks(title="ASR Paragraph Breaks", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Flat ASR text", lines=16, placeholder="Paste a transcript (one block of text)")
            bold = gr.Checkbox(label="Then bold the key lines (email)", value=False)
            btn = gr.Button("Add paragraph breaks", variant="primary")
            gr.Examples([[e["text"]] for e in EXAMPLES], inputs=[inp],
                        example_labels=[e["text"][:60] + "…" for e in EXAMPLES])
        with gr.Column():
            out_html = gr.HTML(label="Rendered")
            status = gr.Markdown()
            with gr.Accordion("Raw output", open=False):
                out_raw = gr.Textbox(lines=12, show_copy_button=True, label=None)
    btn.click(run, [inp, bold], [out_raw, out_html, status])

demo.queue(default_concurrency_limit=1).launch()
