# AutoRound Quantization - Qwen Models

Comprehensive model quantization project using **Intel's AutoRound** algorithm to create production-ready 4-bit quantized versions of Qwen language models. This project demonstrates advanced weight-only quantization techniques optimized for both text and vision-language models.

## 🎯 Project Overview

This repository contains automated quantization pipelines for:
- **Qwen3-4B-Instruct-2507**: Efficient 4B text generation model
- **Qwen3-VL-8B-Instruct**: State-of-the-art 8B vision-language model

All models are quantized to **W4A16** (4-bit weights, 16-bit activations) using Intel's AutoRound algorithm with extensive calibration and tuning for optimal accuracy retention.

## 📊 Models Generated

### Text Model: Qwen3-4B-Instruct-2507

| Format | Repository | Size | Use Case |
|--------|-----------|------|----------|
| AutoRound | [Qwen3-4B-Instruct-2507-W4A16-AutoRound](https://huggingface.co/Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound) | ~2.5GB | Native format (Intel) |
| AWQ | [Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ](https://huggingface.co/Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ) | ~2.5GB | Nvidia GPUs (vLLM, TGI) |

### Vision-Language Model: Qwen3-VL-8B-Instruct

| Format | Repository | Size | Vision Tower | Use Case |
|--------|-----------|------|--------------|----------|
| AutoRound | [Qwen3-VL-8B-Instruct-W4A16-AutoRound](https://huggingface.co/Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound) | ~6GB | BF16 (Unquantized) | Native format (Intel) |
| AWQ | [Qwen3-VL-8B-Instruct-W4A16-AutoRound-AWQ](https://huggingface.co/Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound-AWQ) | ~6GB | BF16 (Unquantized) | Nvidia GPUs (vLLM, TGI) |
| GPTQ | [Qwen3-VL-8B-Instruct-W4A16-AutoRound-GPTQ](https://huggingface.co/Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound-GPTQ) | ~6GB | BF16 (Unquantized) | Broad compatibility |

## 🚀 Key Features

### Advanced Quantization Configuration
```python
TUNING_CONFIG = {
    "group_size": 128,              # Fine-grained quantization control
    "sym": True,                    # Symmetric quantization for better performance
    "iters": 1000,                  # High-precision weight tuning
    "nsamples": 512,                # Extensive calibration samples
    "batch_size": 8,                # Optimized for high-end GPUs
    "seqlen": 2048,                 # Long context support
    "low_gpu_mem_usage": False,     # Keep on GPU for speed
    "enable_torch_compile": True,   # JIT compilation acceleration
    "quant_nontext_module": False   # Preserve vision tower accuracy (VLM only)
}
```

### Quantization Scheme: W4A16
- **4-bit weights**: Reduced model size (~60% compression)
- **16-bit activations**: Maintained accuracy with reduced memory footprint
- **Symmetric quantization**: Better hardware support and inference speed
- **Group size 128**: Optimal balance between accuracy and efficiency

### Vision-Language Model Protection
For Qwen3-VL models, the vision tower is kept in **BF16 (unquantized)** to ensure:
- Perfect visual understanding accuracy
- OCR capability preservation
- Spatial reasoning maintenance
- Chart and diagram analysis accuracy

## 📁 Project Structure

```
AutoRound-Quantization/
├── README.md                              # This file
├── auto_round_Qwen_3_4B.ipynb            # Quantization pipeline for Qwen3-4B
├── auto_round_Qwen_3_VL_8B.ipynb         # Quantization pipeline for Qwen3-VL-8B
└── ReadMe/                                # Model documentation
    ├── Qwen3-4B-Instruct-2507-W4A16-AutoRound.md
    ├── Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ.md
    ├── Qwen3-VL-8B-Instruct-W4A16-AutoRound.md
    ├── Qwen3-VL-8B-Instruct-W4A16-AutoRound-AWQ.md
    └── Qwen3-VL-8B-Instruct-W4A16-AutoRound-GPTQ.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (for GPU quantization)
- High-end GPU with 48GB+ VRAM recommended (A40, A6000, L40, or similar)

### Install Dependencies

```bash
# Install core packages
pip install --upgrade transformers
pip install auto-round
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Hugging Face Hub integration
pip install huggingface-hub
```

### Quantization in Jupyter Notebooks

The project provides two ready-to-use notebooks:
1. **auto_round_Qwen_3_4B.ipynb** - Quantize the 4B text model
2. **auto_round_Qwen_3_VL_8B.ipynb** - Quantize the 8B vision-language model

Run these notebooks to:
- Load the base model from Hugging Face
- Configure quantization parameters
- Perform weight tuning (1000 iterations)
- Export to multiple formats (AutoRound, AWQ, GPTQ)
- Automatically push quantized models to Hugging Face Hub

## 💡 Usage Examples

### Load Quantized Text Model (AutoRound Format)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRoundConfig

model_id = "Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

# Inference
inputs = tokenizer("What is quantization?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Load Quantized Vision-Language Model (AutoRound Format)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from auto_round import AutoRoundConfig
from PIL import Image

model_id = "Vishva007/Qwen3-VL-8B-Instruct-W4A16-AutoRound"

tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

# Multimodal inference
image = Image.open("example.jpg")
conversation = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image"}
    ]}
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True)
image_inputs = processor(images=[image], return_tensors="pt")
text_inputs = tokenizer([text], return_tensors="pt")

output = model.generate(
    **text_inputs,
    **image_inputs,
    max_length=512
)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Load AWQ Format (Nvidia GPU Optimized)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

model_id = "Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ"

# Load with AutoAWQ
model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Use with vLLM or TGI for production
```

## 📈 Performance Metrics

### Quantization Accuracy
- **Iterations**: 1000 (production-grade weight tuning)
- **Calibration Samples**: 512 (comprehensive accuracy preservation)
- **Compression Ratio**: ~60% size reduction
- **Memory Savings**: 4x reduction compared to full precision

### Format Compatibility
| Format | Framework | Platform | Speed |
|--------|-----------|----------|-------|
| AutoRound | Auto-Round (Native) | CPU/GPU | High |
| AWQ | vLLM, TGI, llama.cpp | Nvidia GPU | Very High |
| GPTQ | llama.cpp, Ollama | CPU/GPU | High |

## 🔄 Export Formats

The quantization pipeline supports exporting to multiple formats:

1. **AutoRound Format** - Native Intel format for maximum compatibility with auto-round
2. **AWQ Format** - Optimized for Nvidia GPUs, best with vLLM and Text Generation Inference
3. **GPTQ Format** - Broad compatibility across different inference frameworks

## 🤝 Integration with Hugging Face Hub

All quantized models are automatically pushed to Hugging Face Hub:
- Models are public and available for download
- Full model cards with quantization details
- Integration with Hugging Face transformers library

To use any model:
```bash
# Download will be automatic
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Vishva007/[MODEL_NAME]")
```

## 📚 Documentation

For detailed information about each quantized model, see the [ReadMe/](ReadMe/) folder:
- Model-specific quantization parameters
- Inference code examples
- Format-specific usage instructions
- Performance benchmarks

## 🔧 Technical Details

### AutoRound Algorithm
AutoRound is an advanced weight-only quantization algorithm developed by Intel that:
- Uses iterative weight tuning to minimize quantization error
- Maintains activation precision at 16-bit for better accuracy
- Supports group-wise quantization for fine-grained control
- Achieves state-of-the-art accuracy on large language models

### Why This Approach?
- **Accuracy**: 1000 iterations of weight tuning preserves model quality
- **Speed**: Reduced model size enables faster inference
- **Memory**: 4x memory reduction for deployment on smaller GPUs
- **Compatibility**: Multiple export formats for different hardware
- **Production-Ready**: Extensively calibrated with 512 samples

## ⚙️ System Requirements

### For Quantization
- GPU with 48GB+ VRAM (A40, A6000, L40, A100)
- CUDA 11.8+
- Python 3.8+
- PyTorch compiled with CUDA support

### For Inference
- **AutoRound Format**: CPU or GPU (4GB+ VRAM)
- **AWQ Format**: Nvidia GPU (6GB+ VRAM)
- **GPTQ Format**: CPU or GPU (4GB+ VRAM)

## 🎓 Learning Resources

- [Intel AutoRound GitHub](https://github.com/intel/auto-round)
- [Qwen Models Documentation](https://huggingface.co/Qwen)
- [Quantization Concepts](https://huggingface.co/docs/transformers/quantization)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/index.html)

## 📝 License

This project uses models under the Apache 2.0 license. See individual model repositories for licensing details.

## 👤 Author

Created by [Vishva007](https://huggingface.co/Vishva007)

## 🙏 Acknowledgments

- **Intel** for the AutoRound quantization algorithm
- **Alibaba Qwen Team** for the high-quality base models
- **Hugging Face** for the model hosting and transformers library

## 📞 Support & Contributions

For issues, questions, or contributions:
1. Check the [ReadMe/](ReadMe/) folder for model-specific documentation
2. Review the quantization notebooks for implementation details
3. Visit the Hugging Face model repositories for community discussions

---

**Last Updated**: January 2026

**Project Status**: Production-Ready ✅

All quantized models are tested and ready for deployment in production environments.
