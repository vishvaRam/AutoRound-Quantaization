---
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: transformers
license: apache-2.0
tags:
- gptq
- qwen
- qwen3-vl
- vision-language-model
- quantization
- 4-bit
- auto-round
pipeline_tag: image-text-to-text
model_type: qwen3_vl
---

# Qwen3-VL-8B-Instruct-W4A16-AutoRound-GPTQ

## Model Overview
This is a **4-bit quantized GPTQ** version of the state-of-the-art [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) vision-language model.

Unlike standard GPTQ conversions which rely on a greedy layer-wise algorithm, this model was optimized using **Intel's AutoRound**. AutoRound analyzes the model's weights over 1000 tuning steps with 512 calibration samples to find the optimal quantization points. This results in **significantly lower perplexity** and better reasoning retention than standard GPTQ, while maintaining full compatibility with all GPTQ inference backends.

### Key Highlights
*   **Best-in-Class Quality**: tuned for 1000 iterations to preserve the model's complex visual reasoning capabilities.
*   **Uncompromised Vision**: The visual encoder (Vision Tower) is kept in **FP16 (Unquantized)** to ensure no degradation in OCR, chart reading, or spatial analysis.
*   **Broad Compatibility**: Works with `AutoGPTQ`, `Transformers`, and older `vLLM` versions that support GPTQ.

### Technical Specifications
| Feature | Detail |
| :--- | :--- |
| **Quantization Format** | GPTQ |
| **Quantization Scheme** | W4A16 (4-bit weights, 16-bit activations) |
| **Optimization Algo** | Intel AutoRound (Symmetric, Group Size 128) |
| **Vision Tower** | FP16 (Original Precision) |
| **Model Size** | ~5.5 GB (vs ~16GB Original) |
| **VRAM Requirement** | ~6-8 GB for Inference |

---

## Quickstart

### 1. Installation
To run this model, you need `transformers` and the `auto-gptq` kernel library.

```bash
pip install auto-gptq transformers torch
```

### 2. Inference Example

This snippet demonstrates how to load the model and analyze an image.

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 1. Load the Model
model_id = "Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound-GPTQ"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# 2. Load the Processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 3. Prepare Input (Image + Text)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

# 4. Process Inputs
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# 5. Generate Output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(f"Model Response:\n{output_text}")
```

---

## Performance & Benchmarks

This quantized model aims to match the performance of the FP16 original model while reducing memory usage by nearly 70%.

*   **VRAM Usage**: reduced from **~16GB** (FP16) to **~5.5GB** (GPTQ).
*   **Throughput**: Higher token generation speed on memory-bandwidth limited GPUs (like RTX 3090, 4090, L40).

## Acknowledgements

*   **Base Model**: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
*   **Quantization Tool**: [Intel AutoRound](https://github.com/intel/auto-round)
*   **Paper**: [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516)

## Citation
If you use this model, please cite the original Qwen3-VL paper:

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025}
}
```
