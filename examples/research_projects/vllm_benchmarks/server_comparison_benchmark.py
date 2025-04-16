#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script to compare the performance of vLLM's synchronous and asynchronous server implementations.

This script tests both server types across varying model sizes, prompt lengths, and batch sizes,
measuring metrics such as latency, throughput, and CPU/GPU utilization. It handles the entire 
lifecycle of server processes, ensuring controlled test conditions.

Usage:
    python server_comparison_benchmark.py --output_dir ./benchmark_results
"""

import argparse
import csv
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import requests
import seaborn as sns
import torch
from datasets import load_dataset  # Add datasets import for WikiText

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
SYNC_SCRIPT = "vllm_serve_sync.py"
ASYNC_SCRIPT = "vllm_serve_async.py"
DEFAULT_HOST = "localhost"
DEFAULT_BASE_PORT = 8000  # Base port, we'll increment for multiple servers
REQUEST_TIMEOUT = 120  # Seconds
WARMUP_ITERATIONS = 5
DEFAULT_NUM_REQUESTS = 10  # Reduce to 10 as requested


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Server settings
    models: List[str] = field(default_factory=lambda: ["Qwen/Qwen2.5-1.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    tensor_parallel_sizes: List[int] = field(default_factory=lambda: [1])
    server_types: List[Literal["sync", "async"]] = field(default_factory=lambda: ["sync", "async"])
    
    # Benchmark parameters
    prompt_lengths: List[int] = field(default_factory=lambda: [10, 100, 500])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 16, 32])
    max_tokens: int = 32
    trials_per_config: int = 2  # Reduce number of trials
    num_requests: int = DEFAULT_NUM_REQUESTS  # New parameter for number of requests
    use_wikitext: bool = True  # Flag to use WikiText-103 data instead of synthetic
    
    # Async-specific parameters
    arrival_patterns: List[str] = field(
        default_factory=lambda: ["constant", "uniform", "poisson", "burst"]
    )
    arrival_rates: Dict[str, float] = field(
        default_factory=lambda: {
            "constant": 0.1,  # seconds between requests (10 req/sec)
            "uniform": 0.1,   # mean, with ±20ms jitter
            "poisson": 0.1,   # mean arrival rate (λ) in seconds
            "burst": [5, 1.0],  # [burst_size, seconds_between_bursts]
        }
    )
    
    # Output settings
    output_dir: str = "benchmark_results"
    random_seed: int = 42
    
    # Sampling parameters (fixed for deterministic comparison)
    temperature: float = 0.0  # 0.0 = greedy/deterministic
    top_p: float = 1.0
    top_k: int = 0 


@dataclass
class BenchmarkResult:
    """Stores the results of a single benchmark run."""
    
    # Config identification
    model: str
    server_type: str
    tensor_parallel_size: int
    prompt_length: int
    batch_size: int
    arrival_pattern: Optional[str] = None
    
    # Performance metrics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_req_per_sec: float = 0.0
    
    # Resource utilization
    mean_gpu_utilization: float = 0.0
    max_gpu_memory_usage_mb: float = 0.0
    mean_cpu_utilization: float = 0.0
    
    # Token-related metrics
    tokens_per_second: float = 0.0
    output_token_count: int = 0
    total_token_count: int = 0
    
    # Other metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success_rate: float = 1.0
    trial_id: int = 0
    
    # Output hashes for determinism check
    output_hash: str = ""


class ServerManager:
    """Manages the lifecycle of vLLM server processes."""
    
    def __init__(self, script_name: str, model: str, port: int, tensor_parallel: int = 1):
        self.script_name = script_name
        self.model = model
        self.port = port
        self.tensor_parallel = tensor_parallel
        self.process = None
        self.stdout_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        self.stderr_file = tempfile.NamedTemporaryFile(delete=False, suffix=".err")
        
    def start(self) -> None:
        """Start the server as a subprocess."""
        # Environment for the server process
        env = os.environ.copy()
        
        # Set the script path based on server type
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                  "scripts", self.script_name)
        
        if "sync" in self.script_name:
            cmd = [
                "python", script_path,
                "--model", self.model,
                "--tensor_parallel_size", str(self.tensor_parallel),
                "--port", str(self.port),
                "--host", DEFAULT_HOST,
                "--gpu_memory_utilization", "0.9"
            ]
        else:
            cmd = [
                "python", script_path,
                "--model", self.model,
                "--tensor_parallel_size", str(self.tensor_parallel),
                "--port", str(self.port),
                "--host", DEFAULT_HOST,
            ]
        
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            env=env,
        )
        
        # Wait for server to start up
        server_ready = False
        max_attempts = 120  # 2 minutes at 1 second per attempt
        for attempt in range(max_attempts):
            try:
                health_url = f"http://{DEFAULT_HOST}:{self.port}/health/"
                if "async" in self.script_name:
                    # Adjust URL for vLLM async server
                    health_url = f"http://{DEFAULT_HOST}:{self.port}/v1/models"
                    
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    server_ready = True
                    break
            except requests.RequestException:
                pass
            
            # Check if the process has terminated
            if self.process.poll() is not None:
                returncode = self.process.poll()
                raise RuntimeError(f"Server process terminated with code {returncode} while waiting for startup")
                
            time.sleep(1)
            
        if not server_ready:
            self.stop()
            raise TimeoutError(f"Server failed to start after {max_attempts} seconds")
            
        logger.info(f"Server started on port {self.port}")
    
    def stop(self) -> None:
        """Stop the server process and cleanup."""
        if self.process is None:
            return
            
        # Send SIGTERM to the process group
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.send_signal(signal.SIGTERM)
            
            self.process.terminate()
            
            # Wait up to 30 seconds for process to terminate
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if process doesn't terminate
                self.process.kill()
                self.process.wait()
                
        except (psutil.NoSuchProcess, ProcessLookupError):
            # Process already terminated
            pass
            
        # Close and cleanup log files
        self.stdout_file.close()
        self.stderr_file.close()
        
        # Optional: retain logs for debugging
        # os.unlink(self.stdout_file.name)
        # os.unlink(self.stderr_file.name)
        
        self.process = None
        logger.info(f"Server on port {self.port} stopped")


class BenchmarkClient:
    """Client for benchmarking vLLM servers with different configurations."""
    
    def __init__(self, server_type: str, host: str, port: int, config: BenchmarkConfig):
        self.server_type = server_type
        self.host = host
        self.port = port
        self.config = config
        
        # Initialize session for connection pooling
        self.session = requests.Session()
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Load WikiText-103 dataset if using real data
        if config.use_wikitext:
            logger.info("Loading WikiText-103 dataset for realistic prompts")
            self.wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            # Preprocess to get clean paragraphs of sufficient length
            self.paragraphs = self._preprocess_wikitext()
            logger.info(f"Loaded {len(self.paragraphs)} suitable paragraphs from WikiText-103")
    
    def _preprocess_wikitext(self) -> List[str]:
        """Process WikiText-103 to extract clean paragraphs."""
        paragraphs = []
        current_para = []
        
        for line in self.wikitext["text"]:
            # Skip section headers, empty lines, etc.
            if line.startswith("=") or not line.strip():
                if current_para:
                    paragraph = " ".join(current_para)
                    if len(paragraph.split()) >= 50:  # Only keep substantial paragraphs
                        paragraphs.append(paragraph)
                    current_para = []
                continue
                
            # Add line to current paragraph
            current_para.append(line.strip())
        
        # Add the last paragraph if exists
        if current_para:
            paragraph = " ".join(current_para)
            if len(paragraph.split()) >= 50:
                paragraphs.append(paragraph)
        
        return paragraphs
    
    def generate_prompt(self, length: int) -> str:
        """Generate a prompt of the specified token length."""
        if self.config.use_wikitext and self.paragraphs:
            # Choose a random paragraph
            paragraph = np.random.choice(self.paragraphs)
            
            # Truncate to approximate token length
            words = paragraph.split()
            # Roughly estimate tokens as words/1.3 as a heuristic
            estimated_word_count = int(length * 1.3)
            if len(words) > estimated_word_count:
                words = words[:estimated_word_count]
            
            prompt = "Summarize the following text: " + " ".join(words)
            return prompt
        else:
            # Fallback to synthetic data
            return "Generate a concise summary of the following text: " + "lorem ipsum " * (length // 2)
    
    def create_batch(self, prompt: str, batch_size: int) -> List[str]:
        """Create a batch of identical prompts."""
        return [prompt] * batch_size
    
    def run_benchmark_sync(
        self, 
        prompt_length: int, 
        batch_size: int,
        trial_id: int,
    ) -> BenchmarkResult:
        """Run a synchronous benchmark with the specified parameters."""
        prompt = self.generate_prompt(prompt_length)
        batch = self.create_batch(prompt, batch_size)
        
        # Prepare benchmark result object
        result = BenchmarkResult(
            model=self.config.models[0],  # Current model
            server_type=self.server_type,
            tensor_parallel_size=self.config.tensor_parallel_sizes[0],  # Current tensor parallel size
            prompt_length=prompt_length,
            batch_size=batch_size,
            trial_id=trial_id,
        )
        
        # Run warmup iterations
        for _ in range(WARMUP_ITERATIONS):
            if self.server_type == "sync":
                self._call_sync_generate(batch)
            else:
                self._call_async_generate(batch)
        
        # Measure performance
        latencies = []
        outputs = []
        start_time = time.time()
        
        # Run actual benchmark
        num_successful = 0
        num_requests = self.config.num_requests  # Use the configured number of requests
        
        for i in range(num_requests):
            try:
                req_start = time.time()
                
                if self.server_type == "sync":
                    response = self._call_sync_generate(batch)
                else:
                    response = self._call_async_generate(batch)
                
                req_end = time.time()
                latency = (req_end - req_start) * 1000  # Convert to ms
                latencies.append(latency)
                outputs.append(response)
                num_successful += 1
                
            except Exception as e:
                logger.error(f"Error during benchmark: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        if latencies:
            result.mean_latency_ms = np.mean(latencies)
            result.p50_latency_ms = np.percentile(latencies, 50)
            result.p90_latency_ms = np.percentile(latencies, 90)
            result.p99_latency_ms = np.percentile(latencies, 99)
        
        if num_successful > 0:
            result.throughput_req_per_sec = num_successful / total_time
            result.success_rate = num_successful / num_requests
            
            # Calculate token statistics from the first successful output
            if outputs:
                # This depends on the response format, adjust as needed
                result.output_token_count = len(outputs[0]["completion_ids"][0]) if self.server_type == "sync" else \
                    len(outputs[0]["choices"][0]["message"]["content"].split())
                result.total_token_count = prompt_length + result.output_token_count
                result.tokens_per_second = (result.total_token_count * num_successful) / total_time
                
                # Create a deterministic hash of the output for comparison
                import hashlib
                output_str = str(outputs[0])
                result.output_hash = hashlib.md5(output_str.encode()).hexdigest()
        
        # Monitor resource usage - simplified version
        try:
            gpu_stats = self._get_gpu_stats()
            result.mean_gpu_utilization = gpu_stats.get("utilization.gpu", 0)
            result.max_gpu_memory_usage_mb = gpu_stats.get("memory.used", 0)
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            result.mean_cpu_utilization = cpu_percent
        except Exception as e:
            logger.warning(f"Failed to get resource stats: {e}")
        
        return result
    
    def run_benchmark_async_random_arrivals(
        self,
        prompt_length: int,
        batch_size: int,
        arrival_pattern: str,
        trial_id: int,
    ) -> BenchmarkResult:
        """Run an async benchmark with randomized arrival patterns."""
        prompt = self.generate_prompt(prompt_length)
        
        # Prepare benchmark result object
        result = BenchmarkResult(
            model=self.config.models[0],  # Current model
            server_type=self.server_type,
            tensor_parallel_size=self.config.tensor_parallel_sizes[0],  # Current tensor parallel size
            prompt_length=prompt_length,
            batch_size=batch_size,
            arrival_pattern=arrival_pattern,
            trial_id=trial_id,
        )
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            self._call_async_generate([prompt])
        
        # Prepare arrival times based on pattern
        num_requests = 50  # Total number of requests to send
        arrival_times = []
        
        if arrival_pattern == "constant":
            # Constant interval between requests
            interval = self.config.arrival_rates["constant"]
            arrival_times = [i * interval for i in range(num_requests)]
            
        elif arrival_pattern == "uniform":
            # Uniform random jitter around the mean
            mean_interval = self.config.arrival_rates["uniform"]
            jitter = 0.02  # ±20ms
            arrival_times = [0]
            for i in range(1, num_requests):
                interval = np.random.uniform(mean_interval - jitter, mean_interval + jitter)
                arrival_times.append(arrival_times[-1] + interval)
                
        elif arrival_pattern == "poisson":
            # Poisson process (exponential inter-arrival times)
            mean_interval = self.config.arrival_rates["poisson"]
            arrival_times = [0]
            for i in range(1, num_requests):
                interval = np.random.exponential(mean_interval)
                arrival_times.append(arrival_times[-1] + interval)
                
        elif arrival_pattern == "burst":
            # Burst traffic pattern
            burst_size, time_between_bursts = self.config.arrival_rates["burst"]
            arrival_times = []
            burst_count = num_requests // burst_size
            
            for b in range(burst_count):
                burst_start = b * time_between_bursts
                for i in range(burst_size):
                    # Within a burst, requests are sent with minimal delay
                    arrival_times.append(burst_start + i * 0.001)  # 1ms between requests in a burst
        
        # Run the benchmark with controlled arrival times
        results = []
        latencies = []
        outputs = []
        num_successful = 0
        
        benchmark_start = time.time()
        
        for i, arrival_time in enumerate(arrival_times):
            target_time = benchmark_start + arrival_time
            
            # Wait until the scheduled time to send this request
            now = time.time()
            if now < target_time:
                time.sleep(target_time - now)
            
            try:
                # Async mode - single request at a time but with controlled arrival
                batch = [prompt]
                req_start = time.time()
                response = self._call_async_generate(batch)
                req_end = time.time()
                
                latency = (req_end - req_start) * 1000  # ms
                latencies.append(latency)
                outputs.append(response)
                num_successful += 1
                
                # Store detailed results for this request
                results.append({
                    "request_id": i,
                    "arrival_time": arrival_time,
                    "start_time": req_start - benchmark_start,
                    "end_time": req_end - benchmark_start,
                    "latency_ms": latency,
                })
                
            except Exception as e:
                logger.error(f"Error during async benchmark request {i}: {e}")
        
        benchmark_end = time.time()
        total_time = benchmark_end - benchmark_start
        
        # Calculate metrics
        if latencies:
            result.mean_latency_ms = np.mean(latencies)
            result.p50_latency_ms = np.percentile(latencies, 50)
            result.p90_latency_ms = np.percentile(latencies, 90)
            result.p99_latency_ms = np.percentile(latencies, 99)
        
        if num_successful > 0:
            result.throughput_req_per_sec = num_successful / total_time
            result.success_rate = num_successful / len(arrival_times)
            
            # Calculate token statistics
            if outputs:
                result.output_token_count = len(outputs[0]["choices"][0]["message"]["content"].split())
                result.total_token_count = prompt_length + result.output_token_count
                result.tokens_per_second = (result.total_token_count * num_successful) / total_time
                
                # Create a deterministic hash of the output for comparison
                import hashlib
                output_str = str(outputs[0])
                result.output_hash = hashlib.md5(output_str.encode()).hexdigest()
        
        # Monitor resource usage
        try:
            gpu_stats = self._get_gpu_stats()
            result.mean_gpu_utilization = gpu_stats.get("utilization.gpu", 0)
            result.max_gpu_memory_usage_mb = gpu_stats.get("memory.used", 0)
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            result.mean_cpu_utilization = cpu_percent
        except Exception as e:
            logger.warning(f"Failed to get resource stats: {e}")
        
        # Save detailed per-request data for this run
        if results:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            details_file = output_dir / f"async_details_{arrival_pattern}_{prompt_length}_{batch_size}_{trial_id}.csv"
            
            with open(details_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        
        return result

    def _call_sync_generate(self, prompts: List[str]) -> Dict:
        """Call the synchronous generate endpoint."""
        url = f"http://{self.host}:{self.port}/generate/"
        
        payload = {
            "prompts": prompts,
            "n": 1,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_tokens": self.config.max_tokens,
        }
        
        response = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    
    def _call_async_generate(self, prompts: List[str]) -> Dict:
        """Call the async (vLLM OpenAI-compatible) generate endpoint."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        
        # Convert prompts to OpenAI chat format
        messages = []
        for prompt in prompts:
            messages.append([{"role": "user", "content": prompt}])
        
        # Only sending one prompt at a time in this format
        payload = {
            "model": "model",  # Placeholder, vLLM doesn't use this
            "messages": messages[0],  # Just the first prompt's messages
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        
        response = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics using nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"
            ]
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            
            # Parse output
            values = output.split(',')
            stats = {
                "utilization.gpu": float(values[0]),
                "memory.used": float(values[1]),  # MB
            }
            return stats
        except (subprocess.SubprocessError, ValueError, IndexError):
            logger.warning("Failed to get GPU stats from nvidia-smi")
            return {"utilization.gpu": 0, "memory.used": 0}


def run_benchmarks(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run all benchmarks according to the configuration."""
    results = []
    base_port = DEFAULT_BASE_PORT
    
    for model in config.models:
        for tensor_parallel in config.tensor_parallel_sizes:
            for server_type in config.server_types:
                # Use different ports for different server types
                port = base_port
                base_port += 1
                
                # Start server
                server_script = SYNC_SCRIPT if server_type == "sync" else ASYNC_SCRIPT
                server = ServerManager(
                    script_name=server_script,
                    model=model,
                    port=port,
                    tensor_parallel=tensor_parallel
                )
                
                try:
                    server.start()
                    client = BenchmarkClient(
                        server_type=server_type,
                        host=DEFAULT_HOST,
                        port=port,
                        config=config
                    )
                    
                    # Run benchmarks for standard combinations
                    for prompt_length in config.prompt_lengths:
                        for batch_size in config.batch_sizes:
                            for trial in range(config.trials_per_config):
                                logger.info(
                                    f"Running benchmark: {model}, {server_type}, "
                                    f"TP={tensor_parallel}, prompt={prompt_length}, "
                                    f"batch={batch_size}, trial={trial+1}/{config.trials_per_config}"
                                )
                                
                                result = client.run_benchmark_sync(
                                    prompt_length=prompt_length,
                                    batch_size=batch_size,
                                    trial_id=trial,
                                )
                                
                                results.append(result)
                    
                    # For async server, also test random arrival patterns
                    if server_type == "async":
                        for arrival_pattern in config.arrival_patterns:
                            for prompt_length in config.prompt_lengths:
                                # Use batch_size=1 for arrival pattern tests since we're testing concurrency
                                batch_size = 1
                                
                                for trial in range(config.trials_per_config):
                                    logger.info(
                                        f"Running async benchmark with {arrival_pattern} arrivals: "
                                        f"{model}, TP={tensor_parallel}, prompt={prompt_length}, "
                                        f"trial={trial+1}/{config.trials_per_config}"
                                    )
                                    
                                    result = client.run_benchmark_async_random_arrivals(
                                        prompt_length=prompt_length,
                                        batch_size=batch_size,
                                        arrival_pattern=arrival_pattern,
                                        trial_id=trial,
                                    )
                                    
                                    results.append(result)
                
                finally:
                    # Always stop the server
                    server.stop()
    
    return results


def save_results(results: List[BenchmarkResult], output_dir: str) -> None:
    """Save benchmark results to CSV and JSON files and generate visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_file = Path(output_dir) / f"benchmark_results_{timestamp}.csv"
    data = [asdict(result) for result in results]
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    # Save as JSON for easier parsing and full data retention
    json_file = Path(output_dir) / f"benchmark_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {csv_file} and {json_file}")
    
    # Generate visualizations
    create_visualizations(df, output_dir, timestamp)
    
    # Generate comparisons for deterministic output check
    compare_outputs(results)


def create_visualizations(df: pd.DataFrame, output_dir: Path, timestamp: str) -> None:
    """Create enhanced visualizations for benchmark results."""
    # Set up the visualization style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})
    
    # 1. Server Type Comparison: Latency by Prompt Length and Batch Size
    plt.figure(figsize=(15, 10))
    g = sns.FacetGrid(df, col="batch_size", row="prompt_length", hue="server_type",
                     margin_titles=True, height=3, aspect=1.5)
    g.map(sns.barplot, "server_type", "mean_latency_ms")
    g.add_legend(title="Server Type")
    g.set_axis_labels("Server Type", "Mean Latency (ms)")
    g.set_titles(col_template="Batch Size: {col_name}", row_template="Prompt Length: {row_name}")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"latency_by_prompt_batch_{timestamp}.png")
    
    # 2. Throughput Comparison
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x="server_type", y="throughput_req_per_sec", hue="batch_size")
    plt.title("Throughput by Server Type and Batch Size")
    plt.xlabel("Server Type")
    plt.ylabel("Throughput (requests/second)")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"throughput_by_server_{timestamp}.png")
    
    # 3. Tokens per Second
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="batch_size", y="tokens_per_second", hue="server_type", 
                marker="o", style="server_type", err_style="band")
    plt.title("Tokens per Second by Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Tokens per Second")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"tokens_per_second_{timestamp}.png")
    
    # 4. Resource Usage
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x="server_type", y="mean_gpu_utilization", hue="batch_size")
    plt.title("GPU Utilization by Server Type")
    plt.xlabel("Server Type")
    plt.ylabel("Mean GPU Utilization (%)")
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x="server_type", y="max_gpu_memory_usage_mb", hue="batch_size")
    plt.title("GPU Memory Usage by Server Type")
    plt.xlabel("Server Type")
    plt.ylabel("Max GPU Memory Usage (MB)")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"resource_usage_{timestamp}.png")
    
    # 5. Latency Distribution
    plt.figure(figsize=(15, 8))
    
    # Get a representative subset for each server type
    for server_type in df["server_type"].unique():
        server_data = df[df["server_type"] == server_type]
        
        plt.subplot(1, 2, 1 if server_type == "sync" else 2)
        server_plot_data = pd.DataFrame({
            "Mean": server_data["mean_latency_ms"],
            "P50": server_data["p50_latency_ms"],
            "P90": server_data["p90_latency_ms"],
            "P99": server_data["p99_latency_ms"],
        })
        server_plot_data = server_plot_data.melt(var_name="Metric", value_name="Latency (ms)")
        sns.boxplot(data=server_plot_data, x="Metric", y="Latency (ms)")
        plt.title(f"{server_type.capitalize()} Server Latency Distribution")
        plt.xlabel("Latency Metric")
        plt.ylabel("Latency (ms)")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"latency_distribution_{timestamp}.png")
    
    logger.info(f"Visualizations saved to {output_dir}")


def compare_outputs(results: List[BenchmarkResult]) -> None:
    """Compare output hashes between sync and async servers to ensure deterministic output."""
    # Group results by model, prompt length, and batch size
    groups = {}
    for result in results:
        key = (result.model, result.prompt_length, result.batch_size)
        
        if key not in groups:
            groups[key] = {}
        
        if result.server_type not in groups[key]:
            groups[key][result.server_type] = []
            
        groups[key][result.server_type].append(result.output_hash)
    
    # Check if each group has both sync and async results
    mismatches = []
    for key, server_results in groups.items():
        model, prompt_length, batch_size = key
        
        if len(server_results) < 2:
            # Skip if we don't have both server types
            continue
            
        # Compare hashes
        sync_hashes = set(server_results.get("sync", []))
        async_hashes = set(server_results.get("async", []))
        
        if not sync_hashes.intersection(async_hashes):
            mismatches.append({
                "model": model,
                "prompt_length": prompt_length,
                "batch_size": batch_size,
                "sync_hashes": list(sync_hashes),
                "async_hashes": list(async_hashes),
            })
    
    if mismatches:
        logger.warning(f"Found {len(mismatches)} output mismatches between sync and async servers")
        for mismatch in mismatches:
            logger.warning(f"Mismatch: {mismatch}")
    else:
        logger.info("No output mismatches detected. Both servers produced identical outputs.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM synchronous vs asynchronous servers")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save results")
    parser.add_argument("--models", type=str, nargs="+", default=["Qwen/Qwen2.5-1.5B"], 
                        help="Models to benchmark")
    parser.add_argument("--tensor_parallel_sizes", type=int, nargs="+", default=[1],
                        help="Tensor parallel sizes to test")
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[10, 100, 500],
                        help="Prompt lengths (tokens) to test")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 16, 32],
                        help="Batch sizes to test")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials per configuration")
    parser.add_argument("--num_requests", type=int, default=DEFAULT_NUM_REQUESTS, 
                        help="Number of requests to run per configuration")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_wikitext", action="store_true", default=True,
                        help="Use WikiText-103 data for realistic prompts")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        models=args.models,
        tensor_parallel_sizes=args.tensor_parallel_sizes,
        prompt_lengths=args.prompt_lengths,
        batch_sizes=args.batch_sizes,
        trials_per_config=args.trials,
        num_requests=args.num_requests,
        use_wikitext=args.use_wikitext,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )
    
    # Run benchmarks
    results = run_benchmarks(config)
    
    # Save results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main() 