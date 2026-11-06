import asyncio
import json
import statistics
import time

import httpx


async def benchmark_long_generation(
    url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "LLM",
    max_tokens: int = 6000,
) -> None:
    headers = {"Content-Type": "application/json"}
    
    # Prompt structured to demand deep, multi-section text to sustain long output
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Provide a thorough, comprehensive technical architecture breakdown "
                    "of hybrid linear attention models versus standard transformers. "
                    "Analyze memory complexity, recurrent state updates, kernel fusion, "
                    "hardware bottlenecks, and post-training quantization behavior across W4A16. "
                    "Explain every section in granular depth with examples."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    print(f"Connecting to {url}...")
    print(f"Target model: {model}")
    print(f"Requested max_tokens: {max_tokens}\n" + "-" * 50)

    token_timestamps: list[float] = []
    generated_chunks: list[str] = []
    usage_tokens: int | None = None

    t_start = time.perf_counter()
    ttft: float | None = None

    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(f"Error {response.status_code}: {body.decode()}")
                return

            async for line in response.aiter_lines():
                t_now = time.perf_counter()
                if not line.startswith("data: "):
                    continue

                raw_data = line[6:].strip()
                if raw_data == "[DONE]":
                    break

                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue

                # Capture usage block if emitted at the end
                if "usage" in data and data["usage"]:
                    usage_tokens = data["usage"].get("completion_tokens")

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    if ttft is None:
                        ttft = t_now - t_start
                    token_timestamps.append(t_now)
                    generated_chunks.append(content)
                    print(content, end="", flush=True)

    t_end = time.perf_counter()
    total_time = t_end - t_start

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    # Use server-reported usage if available, else count chunk intervals
    total_tokens = usage_tokens if usage_tokens else len(token_timestamps)
    
    if total_tokens <= 1 or ttft is None:
        print("Not enough tokens received to measure speed.")
        return

    # Decode duration excludes initial prefill latency
    decode_time = t_end - token_timestamps[0]
    decode_tokens = total_tokens - 1
    decode_throughput = decode_tokens / decode_time if decode_time > 0 else 0
    e2e_throughput = total_tokens / total_time

    # Calculate Inter-Token Latency (ITL) in milliseconds
    itls_ms = [
        (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
        for i in range(1, len(token_timestamps))
    ]

    median_itl = statistics.median(itls_ms) if itls_ms else 0.0
    p95_itl = (
        statistics.quantiles(itls_ms, n=20)[18]
        if len(itls_ms) >= 20
        else max(itls_ms)
    )

    print(f"Total Tokens Generated:     {total_tokens} tokens")
    print(f"Total Wall Clock Time:      {total_time:.3f} s")
    print(f"Time to First Token (TTFT): {ttft * 1000.0:.2f} ms")
    print(f"Pure Decode Duration:       {decode_time:.3f} s")
    print(f"Decode Throughput:          {decode_throughput:.2f} tokens/s")
    print(f"End-to-End Throughput:      {e2e_throughput:.2f} tokens/s")
    print(f"Median Inter-Token Latency: {median_itl:.2f} ms")
    print(f"P95 Inter-Token Latency:    {p95_itl:.2f} ms")


if __name__ == "__main__":
    # Prerequisite: pip install httpx
    asyncio.run(benchmark_long_generation())