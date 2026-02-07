---
base_model: Qwen/Qwen3-VL-2B-Instruct
library_name: auto-round
license: apache-2.0
tags:
- auto-round
- intel
- qwen
- qwen3-vl
- vision-language-model
- quantization
- 4-bit
- W4A16
pipeline_tag: image-text-to-text
model_type: qwen3_vl
---

# Qwen3-VL-2B-Instruct-W4A16-AutoRound-GPTQ

## Model Overview
This is a **4-bit quantized** version of the powerful [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) vision-language model.

It was optimized using **Intel's AutoRound** algorithm, which calibrates weights for 800 iterations to minimize quantization loss. This version retains the original **FP16 vision tower**, ensuring that visual capabilities (OCR, spatial reasoning, chart analysis) remain degradation-free.

### Quantization Specifications
- **Method**: [AutoRound](https://github.com/intel/auto-round) (Advanced Weight-Only Quantization)
- **Scheme**: `W4A16` (4-bit weights, 16-bit activations)
- **Symmetric**: `True`
- **Group Size**: 128
- **Vision Tower**: Kept in FP16 (Unquantized for max accuracy)
- **Calibration**: 512 samples, 800 iterations

## Quickstart

### 1. Installation
To use this model in its native AutoRound format, you need the `auto-round` library.

```bash
pip install auto-round transformers torch
```

### 2. Inference Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from auto_round import AutoRoundConfig

model_id = "Vishva007/Qwen3-VL-2B-Instruct-W4A16-AutoRound-GPTQ"

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Prepare Input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "Describe this image detailly."},
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
).to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(generated_ids, skip_special_tokens=True))
```

## Citation
```bibtex
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```