

# Vishva007/Qwen3.5-0.8B-W4A16-AutoRound

This is a **W4A16 (4-bit weight, 16-bit activation)** quantized version of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), produced using [AutoRound](https://github.com/intel/auto-round) — Intel's sign gradient descent based quantization method designed for production-grade accuracy retention.

## Quantization Details

| Parameter | Value |
|---|---|
| Method | AutoRound (W4A16) |
| Group Size | 128 |
| Symmetric | Yes |
| Iterations | 1000 |
| Calibration Samples | 512 |
| Sequence Length | 2048 |
| Torch Compile | Enabled |

## Key Notes

- **Ultra-high accuracy configuration** — 1000 iterations with 512 calibration samples ensures near-lossless quantization, especially critical at this model scale where parameter budget is tight.
- **W4A16** — Weights are quantized to 4-bit integers; activations remain in FP16 for inference stability.
- **Extremely lightweight** — The quantized 0.8B model is suitable for edge deployment, low-latency inference, and resource-constrained environments.
- **~50% memory reduction** compared to the FP16 base model.

## Usage

This model is compatible with `transformers` and backends that support AutoRound format weights (e.g., vLLM, SGLang). For full model details, architecture, and capabilities, refer to the [base model page](https://huggingface.co/Qwen/Qwen3.5-0.8B).
