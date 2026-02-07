---
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: transformers
license: apache-2.0
tags:
- awq
- qwen
- qwen3-vl
- vision-language-model
- quantization
- 4-bit
- vllm
pipeline_tag: image-text-to-text
model_type: qwen3_vl
---

# Qwen3-VL-8B-Instruct-W4A16-AutoRound-AWQ

## Model Overview
This model is the **AWQ (Activation-aware Weight Quantization)** export of [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

It combines the **speed of AWQ** with the **accuracy of AutoRound**. The weights were fine-tuned for 1000 steps to ensure the 4-bit degradation is negligible. The vision tower remains in full precision (FP16) to maintain top-tier performance on OCR and visual tasks.

### Key Features
- **High Performance**: Optimized for `vLLM` serving and `Transformers`.
- **Low VRAM**: Fits comfortably on 12GB+ GPUs (RTX 3060/4070).
- **Accurate Vision**: Vision encoder is NOT quantized, preserving full visual acuity.

## Usage

### Option A: vLLM (Recommended for Speed)

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

model_id = "Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound-AWQ"

llm = LLM(
    model=model_id,
    quantization="awq",
    trust_remote_code=True,
    max_model_len=4096
)

# ... (Standard vLLM inference code)
```

### Option B: Transformers (Standard)

```bash
pip install autoawq transformers
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

model_id = "Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound-AWQ"

# Load with Flash Attention 2 for best performance
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(model_id)

# Inference Example
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "What does this image show?"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(generated_ids, skip_special_tokens=True))
```

## Citation
```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025}
}
```