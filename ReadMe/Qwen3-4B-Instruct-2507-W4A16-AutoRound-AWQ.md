---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: transformers
license: apache-2.0
tags:
- awq
- qwen
- qwen3
- quantization
- 4-bit
- vllm
pipeline_tag: text-generation
model_type: qwen2
---

# Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ

## Model Overview
This is the **AWQ (Activation-aware Weight Quantization)** version of **[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**.

It was generated using **Intel's AutoRound** algorithm, which optimizes the weight rounding to minimize quantization loss. This ensures superior accuracy compared to standard AWQ conversion methods.

### Key Features
- **4-bit Inference**: Runs efficiently on Nvidia GPUs.
- **High Accuracy**: Tuned for 1000 iterations using AutoRound.
- **Broad Compatibility**: Works natively with `vLLM`, `TGI`, and `Transformers`.

## Specifications
- **Scheme**: W4A16 (4-bit weights, 16-bit activations)
- **Group Size**: 128
- **Symmetric**: True
- **Calibration Data**: 512 samples
- **Format**: AutoAWQ (Compatible with standard AWQ kernels)

## Usage

### Option A: Using vLLM (Recommended for Speed)
This model is optimized for high-throughput serving with vLLM.

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

model_id = "Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ"

llm = LLM(
    model=model_id,
    quantization="awq",
    dtype="half", 
    max_model_len=8192,
    gpu_memory_utilization=0.90
)

prompts = ["What is the capital of France?"]
sampling_params = SamplingParams(temperature=0.7, top_p=0.8)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs.text)
```

### Option B: Using Hugging Face Transformers

```bash
pip install autoawq transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Write a python function to reverse a string."
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

## Benchmark & Performance
This model maintains the strong performance of the Qwen3-4B-Instruct-2507 base model, including its updated reasoning and coding capabilities.

| Model | Format | VRAM (Est.) |
| :--- | :--- | :--- |
| Qwen3-4B-Instruct (BF16) | Original | ~9 GB |
| **Qwen3-4B-Instruct (AWQ)** | **4-bit** | **~3.5 GB** |

## Citation
```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
