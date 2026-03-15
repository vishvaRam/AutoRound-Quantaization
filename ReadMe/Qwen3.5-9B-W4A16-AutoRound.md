
# Vishva007/Qwen3.5-9B-W4A16-AutoRound

This is a **W4A16 (4-bit weight, 16-bit activation)** quantized version of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), produced using [AutoRound](https://github.com/intel/auto-round) — Intel's sign gradient descent based quantization method designed for production-grade accuracy retention.

## Quantization Details

| Parameter | Value |
|---|---|
| Method | AutoRound (W4A16) |
| Group Size | 128 |
| Symmetric | Yes |
| Iterations | 800 |
| Calibration Samples | 512 |
| Sequence Length | 2048 |
| Torch Compile | Enabled |

## Key Notes

- **High accuracy configuration** — 800 iterations with 512 calibration samples targets production-grade quality with minimal degradation from the base model.
- **W4A16** — Weights are quantized to 4-bit integers; activations remain in FP16 for inference stability.
- **~50% memory reduction** compared to the FP16 base model, enabling deployment on consumer and mid-range GPUs.

## Usage

This model is compatible with `transformers` and backends that support AutoRound GPTQ-format weights (e.g., vLLM, SGLang, AutoGPTQ). For full model details, architecture, and capabilities, refer to the [base model page](https://huggingface.co/Qwen/Qwen3.5-9B).

