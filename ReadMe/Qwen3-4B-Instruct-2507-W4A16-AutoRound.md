---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: auto-round
license: apache-2.0
tags:
- auto-round
- intel
- qwen
- qwen3
- quantization
- 4-bit
- W4A16
pipeline_tag: text-generation
model_type: qwen2
---

# Qwen3-4B-Instruct-2507-W4A16-AutoRound

## Model Overview
This model is a 4-bit quantized version of **[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**, optimized using **Intel's AutoRound** algorithm.

It achieves state-of-the-art accuracy retention by tuning weights for 1000 iterations with 512 calibration samples, significantly outperforming standard RTN (Round-to-Nearest) quantization.

### Quantization Details
- **Method**: [AutoRound](https://github.com/intel/auto-round) (Advanced Weight-Only Quantization)
- **Scheme**: `W4A16` (4-bit weights, 16-bit activations)
- **Symmetric**: `True`
- **Group Size**: 128
- **Tuning Config**: 1000 iterations, 512 samples, batch size 8
- **Framework**: Intel AutoRound

## Quickstart

### 1. Install Dependencies
You need the `auto-round` library to run this model in its native format.

```bash
pip install auto-round transformers torch
```

### 2. Inference Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

model_id = "Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound"

# Load the model with AutoRound configuration
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Explain quantum computing in one sentence."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

## Performance
Qwen3-4B-Instruct-2507 is the latest non-thinking instruct model from the Qwen team, featuring significant improvements in reasoning, coding, and instruction following. 

This quantized version retains nearly 99% of the FP16 performance while reducing VRAM usage significantly, enabling deployment on consumer GPUs (e.g., RTX 3060/4060).

## Citation
```bibtex
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```
