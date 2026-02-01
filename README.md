# AutoRound Quantization

Comprehensive model quantization project using **Intel's AutoRound** algorithm to create production-ready 4-bit quantized versions of language models. This project demonstrates advanced weight-only quantization techniques optimized for both text and vision-language models.

## 🎯 Project Overview

This repository contains automated quantization pipelines for various language models, quantized to **W4A16** (4-bit weights, 16-bit activations) using Intel's AutoRound algorithm with extensive calibration and tuning for optimal accuracy retention.

## 📊 Quantization Details

The project produces quantized models in multiple formats optimized for different deployment scenarios.

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
- Load base models from Hugging Face
- Configure quantization parameters
- Perform weight tuning (1000 iterations)
- Export to multiple formats (AutoRound, AWQ, GPTQ)
- Push quantized models to Hugging Face Hub

## 💡 Usage Examples

### Load Quantized Model (AutoRound Format)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model_id")
model = AutoModelForCausalLM.from_pretrained(
    "model_id",
    device_map="auto",
    torch_dtype="auto"
)

# Inference
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Load AWQ Format (Nvidia GPU Optimized)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized("model_id", fuse_layers=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("model_id")

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

All quantized models are automatically pushed to Hugging Face Hub with full model cards and quantization details.

## 📚 Documentation

For detailed information about quantization implementations, see the [ReadMe/](ReadMe/) folder for implementation details and examples.

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
- **Memory**: 4x memory reduction for deployment
- **Compatibility**: Multiple export formats for different hardware
- **Production-Ready**: Extensively calibrated with 512 samples

## ⚙️ System Requirements

### For Quantization
- GPU with 48GB+ VRAM (A40, A6000, L40, A100)
- CUDA 11.8+
- Python 3.8+
- PyTorch compiled with CUDA support

### For Inference
- CPU or GPU depending on format
- Minimal VRAM requirements for quantized models

## 🎓 Learning Resources

- [Intel AutoRound GitHub](https://github.com/intel/auto-round)
- [Quantization Concepts](https://huggingface.co/docs/transformers/quantization)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/quantization/index.html)

## 📝 License

This project is released under Apache 2.0 license.

## 🙏 Acknowledgments

- **Intel** for the AutoRound quantization algorithm
- **Hugging Face** for the model hosting and transformers library

## 📞 Support & Contributions

For questions or contributions, review the quantization notebooks for implementation details and the ReadMe folder for documentation.

---

**Last Updated**: February 2026

**Project Status**: Production-Ready ✅
