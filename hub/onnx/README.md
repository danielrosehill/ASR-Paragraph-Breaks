---
license: gemma
base_model: danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks
language:
  - en
pipeline_tag: text-generation
library_name: onnxruntime
tags:
  - onnx
  - asr-post-processing
  - paragraph-segmentation
  - on-device
---

# Gemma 3 270M: ASR paragraph breaks (ONNX)

ONNX export of [`danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks`](https://huggingface.co/danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks) for ONNX Runtime, so it can run on a
phone on the same runtime as an ONNX speech-recognition model. Exported with `optimum-onnx`
(`text-generation-with-past`, with a KV cache); `model_quantized.onnx` is int8 dynamic quantization of
`model.onnx`. See the main model card for what it does, results and the required fallback check.

## Check against the PyTorch model

Greedy decoding on held-out test memos, ONNX Runtime CPU provider. Timing is on a cloud CPU, so treat
it as a rough guide to a phone.

| File | Size | Same output as PyTorch | Text unchanged | Time per memo |
|---|---|---|---|---|
| `onnx/model.onnx` | 1664 MB | 6 of 6 | 6 of 6 | 48.1 s |
| `onnx/model_quantized.onnx` | 417 MB | 2 of 6 | 5 of 6 | 23.2 s |

## Usage

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX")
model = ORTModelForCausalLM.from_pretrained("danielrosehill/Gemma-3-270M-ASR-Paragraph-Breaks-ONNX", subfolder="onnx", file_name="model.onnx")
```

Use the prompt and the fallback check from the main model card. Prefer `model.onnx`: the int8 file
matched the PyTorch output on only 2 of 6 memos in the check above and changed the text in one, so it
is published as an experiment. Generation is slow on CPU because the model retypes the whole memo;
a lower-bit weight-only quantization and prompt-lookup decoding are the next things to try.

Licence: Gemma Terms of Use, inherited from the base model.
