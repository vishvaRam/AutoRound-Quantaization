
# Vishva007/Qwen3.5-9B-W4A16-AutoRound-GPTQ

This is a **W4A16 (4-bit weight, 16-bit activation) GPTQ-format** quantized version of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), produced using [AutoRound](https://github.com/intel/auto-round) — Intel's sign gradient descent based quantization method designed for production-grade accuracy retention.

## Quantization Details

| Parameter | Value |
|---|---|
| Method | AutoRound (W4A16, GPTQ format) |
| Group Size | 128 |
| Symmetric | Yes |
| Iterations | 800 |
| Calibration Samples | 512 |
| Sequence Length | 2048 |
| Torch Compile | Enabled |

## Key Notes

- **GPTQ format** — Exported in the standard GPTQ format for broad ecosystem compatibility.
- **High accuracy configuration** — 800 iterations with 512 calibration samples targets production-grade quality with minimal degradation from the base model.
- **W4A16** — Weights are quantized to 4-bit integers; activations remain in FP16 for inference stability.
- **~50% memory reduction** compared to the FP16 base model, enabling deployment on consumer and mid-range GPUs.

## Usage

This model is compatible with `transformers`, `AutoGPTQ`, `vLLM`, and `SGLang` — any backend supporting GPTQ-format weights works out of the box. For full model details, architecture, and capabilities, refer to the [base model page](https://huggingface.co/Qwen/Qwen3.5-9B).
