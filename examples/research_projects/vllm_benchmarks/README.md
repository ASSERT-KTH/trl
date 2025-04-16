# vLLM Server Benchmarking Tools

This directory contains benchmark tools for measuring and comparing the performance of vLLM's synchronous and asynchronous server implementations.

## Tools Overview

1. **server_comparison_benchmark.py** - Compares the performance of vLLM's synchronous and asynchronous servers across different model sizes, prompt lengths, and batch sizes.

2. **async_arrival_benchmark.py** - Tests the asynchronous server's performance under different request arrival patterns to evaluate how it handles various levels of concurrency and burstiness.

## Key Metrics Measured

- **Latency**: Time to complete individual requests (mean, p50, p90, p99)
- **Throughput**: Number of requests processed per second
- **Resource Utilization**: CPU and GPU utilization during benchmark runs
- **Determinism**: Verifies that outputs match between sync and async servers for the same inputs
- **Success Rate**: Percentage of requests that complete successfully

## Realistic Testing with WikiText-103

Both benchmark scripts can use real-world text from the WikiText-103 dataset to generate realistic prompts. This provides several advantages:
- Creates prompts with natural language patterns instead of synthetic text
- Tests how servers handle the complexity and variety of real text
- More accurately represents production workloads

The scripts automatically:
- Load and preprocess the WikiText-103 dataset
- Extract clean paragraphs of suitable length
- Select random paragraphs for each test
- Truncate to approximate the desired token length

## Server Comparison Benchmark

This benchmark systematically compares sync and async server performance using a comprehensive parameter sweep.

### Features:

- Tests a range of model sizes from small to large
- Varies prompt lengths from short to long
- Tests multiple batch sizes (1 to 32)
- Measures end-to-end request latency
- Monitors GPU utilization and memory usage
- Ensures deterministic outputs match between server implementations
- Creates detailed visualizations for performance analysis
- Uses WikiText-103 for realistic prompts

### Usage:

```bash
python server_comparison_benchmark.py \
  --models Qwen/Qwen2.5-1.5B TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tensor_parallel_sizes 1 \
  --prompt_lengths 10 100 500 \
  --batch_sizes 1 4 16 32 \
  --trials 2 \
  --num_requests 10 \
  --use_wikitext \
  --output_dir ./benchmark_results
```

## Async Arrival Pattern Benchmark

This benchmark focuses specifically on how the async server handles different request arrival patterns.

### Features:

- Tests different arrival patterns:
  - **Constant Rate**: Requests arrive at fixed intervals
  - **Uniform Random**: Arrival times have uniform random jitter
  - **Poisson Process**: Exponential inter-arrival times
  - **Burst**: Groups of requests arrive together followed by quiet periods
- Simulates various load levels (requests per second)
- Visualizes concurrent request patterns and latency distributions
- Measures how server performance degrades under increasing load
- Uses WikiText-103 for realistic prompts
- Generates enhanced visualization suite

### Usage:

```bash
python async_arrival_benchmark.py \
  --model Qwen/Qwen2.5-1.5B \
  --tensor_parallel_size 1 \
  --prompt_lengths 10 100 500 \
  --load_levels 1.0 5.0 10.0 20.0 \
  --trials 2 \
  --total_requests 10 \
  --use_wikitext \
  --output_dir ./async_benchmark_results
```

## Requirements

- `vllm>=0.2.0`
- `trl` (current repository)
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `requests`
- `psutil`
- `datasets` (for WikiText-103)

## Enhanced Visualizations

The benchmark scripts generate a comprehensive set of visualizations to help analyze performance:

### Server Comparison Benchmark
- **Latency Grid**: Compares latency across prompt lengths and batch sizes
- **Throughput Analysis**: Shows how throughput scales with batch size
- **Resource Usage**: Compares CPU/GPU utilization between server types
- **Tokens per Second**: Measures token processing efficiency
- **Latency Distribution**: Compares p50/p90/p99 latency distribution

### Async Arrival Benchmark
- **Pattern Comparison**: Compares how different arrival patterns affect performance
- **Throughput vs Load**: Shows how actual throughput scales against target load
- **Success Rate Analysis**: Visualizes reliability at increasing load levels
- **Concurrent Request Patterns**: Shows request queuing behavior over time
- **Combined Performance Score**: Aggregates metrics for overall pattern comparison

All visualizations are saved to the specified output directory with timestamps.

## Results Interpretation

Both benchmarks generate CSV and JSON files with detailed metrics, as well as visualizations to help interpret the results:

- **Latency vs. Load**: How response time changes with increasing load
- **Throughput vs. Target Load**: How actual throughput compares to target request rate
- **Concurrency Patterns**: How many requests are in-flight at any given time
- **Latency Distributions**: Statistical distribution of response times by pattern

The determinism verification ensures that both server implementations produce identical outputs when using the same random seed and greedy sampling.

## Notes

- All benchmarks run the servers with a fixed random seed and temperature=0 to ensure deterministic output.
- Both benchmarks handle server startup/shutdown automatically.
- The tests require a CUDA-capable GPU.
- For multi-GPU testing, adjust the `tensor_parallel_sizes` parameter.
- The number of requests can be limited to 10 for quicker testing.
- For production-scale testing, increase the `num_requests`/`total_requests` values. 