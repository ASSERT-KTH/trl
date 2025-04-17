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
Specialized benchmark script for testing vLLM's asynchronous server under various arrival patterns.

This script focuses on how the async server responds to different request timing patterns:
- Constant rate arrivals
- Uniform random jitter
- Poisson process (exponential inter-arrival)
- Burst patterns (groups of requests arriving together)

It visualizes the server's behavior under these patterns and measures metrics like 
latency distribution, queue depth, and throughput.

Usage:
    python async_arrival_benchmark.py --model Qwen/Qwen2.5-1.5B --output_dir ./async_results
"""

import argparse
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
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import requests
import seaborn as sns
import torch
from datasets import load_dataset  # Import for WikiText

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
ASYNC_SCRIPT = "vllm_serve_async.py"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
REQUEST_TIMEOUT = 60  # Seconds
WARMUP_ITERATIONS = 5
DEFAULT_NUM_REQUESTS = 10  # Reduce to 10 as requested


@dataclass
class AsyncBenchmarkConfig:
    """Configuration for async server benchmark runs."""
    
    # Server settings
    model: str = "Qwen/Qwen2.5-1.5B"
    tensor_parallel_size: int = 1
    
    # Benchmark parameters
    prompt_lengths: List[int] = field(default_factory=lambda: [10, 100, 500])
    max_tokens: int = 32
    trials_per_config: int = 2  # Reduced to 2
    total_requests: int = DEFAULT_NUM_REQUESTS  # Reduced to 10
    use_wikitext: bool = True  # Flag to use WikiText data
    
    # Arrival patterns
    arrival_patterns: List[str] = field(
        default_factory=lambda: ["constant", "uniform", "poisson", "burst"]
    )
    arrival_rates: Dict[str, Union[float, List[float]]] = field(
        default_factory=lambda: {
            "constant": 0.1,  # seconds between requests (10 req/sec)
            "uniform": [0.1, 0.02],  # [mean, jitter] in seconds
            "poisson": 0.1,  # mean arrival rate (λ) in seconds
            "burst": [5, 1.0],  # [burst_size, seconds_between_bursts]
        }
    )
    
    # Load levels to test (requests per second)
    load_levels: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0, 20.0])
    
    # Output settings
    output_dir: str = "async_benchmark_results"
    random_seed: int = 42
    
    # Sampling parameters (fixed for deterministic comparison)
    temperature: float = 0.0  # Deterministic
    top_p: float = 1.0
    top_k: int = 0


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    # Request identification
    request_id: int
    arrival_time: float  # Scheduled arrival time
    actual_arrival_time: float  # Actual time request was sent
    prompt_length: int
    
    # Timing metrics
    start_time: float  # When request was sent
    end_time: float  # When response was received
    latency: float  # end_time - start_time
    
    # Response metrics
    status_code: int
    success: bool
    output_length: int = 0  # Number of tokens in the response
    
    # Server state at time of request
    concurrent_requests: int = 0  # Estimated concurrent requests


@dataclass
class TrialResults:
    """Results for a full benchmark trial."""
    
    # Trial configuration
    model: str
    tensor_parallel_size: int
    prompt_length: int
    arrival_pattern: str
    load_level: float  # Targeted requests per second
    trial_id: int
    
    # Aggregated metrics
    mean_latency: float = 0.0
    p50_latency: float = 0.0
    p90_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0  # Actual requests per second achieved
    success_rate: float = 1.0
    
    # Request-level data
    requests: List[RequestMetrics] = field(default_factory=list)
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Resource utilization
    mean_gpu_utilization: float = 0.0
    max_gpu_memory_mb: float = 0.0


class AsyncServerManager:
    """Manages the lifecycle of a vLLM async server process."""
    
    def __init__(self, model: str, port: int = DEFAULT_PORT, tensor_parallel: int = 1):
        self.model = model
        self.port = port
        self.tensor_parallel = tensor_parallel
        self.process = None
        self.stdout_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        self.stderr_file = tempfile.NamedTemporaryFile(delete=False, suffix=".err")
    
    def start(self) -> None:
        """Start the async server as a subprocess."""
        # Environment for the server process
        env = os.environ.copy()
        
        # Set the script path
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                   "scripts", ASYNC_SCRIPT)
        
        cmd = [
            "python", script_path,
            "--model", self.model,
            "--tensor_parallel_size", str(self.tensor_parallel),
            "--port", str(self.port),
            "--host", DEFAULT_HOST,
        ]
        
        logger.info(f"Starting async server with command: {' '.join(cmd)}")
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
                # Check OpenAI-compatible health endpoint
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
            
        logger.info(f"Async server started on port {self.port}")
    
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
        
        self.process = None
        logger.info(f"Async server on port {self.port} stopped")


class AsyncBenchmarkClient:
    """Client for benchmarking vLLM async server with different arrival patterns."""
    
    def __init__(self, host: str, port: int, config: AsyncBenchmarkConfig):
        self.host = host
        self.port = port
        self.config = config
        
        # Initialize session for connection pooling
        self.session = requests.Session()
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Request tracking
        self.active_requests = 0
        
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
        if hasattr(self, 'paragraphs') and self.paragraphs:
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
    
    def run_benchmark(
        self,
        prompt_length: int,
        arrival_pattern: str,
        load_level: float,
        trial_id: int,
    ) -> TrialResults:
        """Run a benchmark with the specified arrival pattern and load level."""
        prompt = self.generate_prompt(prompt_length)
        
        # Create results object
        results = TrialResults(
            model=self.config.model,
            tensor_parallel_size=self.config.tensor_parallel_size,
            prompt_length=prompt_length,
            arrival_pattern=arrival_pattern,
            load_level=load_level,  # Target requests per second
            trial_id=trial_id,
        )
        
        # Run warmup requests
        for _ in range(WARMUP_ITERATIONS):
            self._call_generate(prompt)
        
        # Calculate arrival times based on pattern
        arrival_times = self._generate_arrival_schedule(
            arrival_pattern=arrival_pattern,
            load_level=load_level,
            total_requests=self.config.total_requests,
        )
        
        # Track concurrent requests
        self.active_requests = 0
        concurrent_tracking = []
        request_metrics = []
        
        # Start benchmark
        benchmark_start = time.time()
        
        for i, arrival_time in enumerate(arrival_times):
            target_time = benchmark_start + arrival_time
            
            # Wait until scheduled time
            now = time.time()
            if now < target_time:
                time.sleep(target_time - now)
            
            # Record actual arrival time
            actual_arrival = time.time() - benchmark_start
            
            # Make the request
            try:
                self.active_requests += 1
                concurrent_tracking.append((actual_arrival, self.active_requests))
                
                req_start = time.time()
                response, status_code = self._call_generate(prompt)
                req_end = time.time()
                
                self.active_requests -= 1
                concurrent_tracking.append((req_end - benchmark_start, self.active_requests))
                
                # Record metrics
                success = status_code == 200
                output_length = 0
                if success and response:
                    try:
                        output_length = len(response["choices"][0]["message"]["content"].split())
                    except (KeyError, IndexError):
                        pass
                
                metrics = RequestMetrics(
                    request_id=i,
                    arrival_time=arrival_time,
                    actual_arrival_time=actual_arrival,
                    prompt_length=prompt_length,
                    start_time=req_start - benchmark_start,
                    end_time=req_end - benchmark_start,
                    latency=(req_end - req_start) * 1000,  # milliseconds
                    status_code=status_code,
                    success=success,
                    output_length=output_length,
                    concurrent_requests=self.active_requests,
                )
                
                request_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error during async benchmark request {i}: {e}")
                # Record failed request
                metrics = RequestMetrics(
                    request_id=i,
                    arrival_time=arrival_time,
                    actual_arrival_time=actual_arrival,
                    prompt_length=prompt_length,
                    start_time=time.time() - benchmark_start,
                    end_time=time.time() - benchmark_start,
                    latency=0,
                    status_code=0,
                    success=False,
                    concurrent_requests=self.active_requests,
                )
                request_metrics.append(metrics)
                self.active_requests -= 1
        
        # Benchmark complete
        benchmark_end = time.time()
        total_time = benchmark_end - benchmark_start
        
        # Calculate aggregate metrics
        successful_requests = [r for r in request_metrics if r.success]
        if successful_requests:
            latencies = [r.latency for r in successful_requests]
            results.mean_latency = np.mean(latencies)
            results.p50_latency = np.percentile(latencies, 50)
            results.p90_latency = np.percentile(latencies, 90)
            results.p99_latency = np.percentile(latencies, 99)
            
            results.success_rate = len(successful_requests) / len(request_metrics)
            results.throughput = len(successful_requests) / total_time
        
        # Resource usage
        try:
            gpu_stats = self._get_gpu_stats()
            results.mean_gpu_utilization = gpu_stats.get("utilization.gpu", 0)
            results.max_gpu_memory_mb = gpu_stats.get("memory.used", 0)
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
        
        # Store all request metrics
        results.requests = request_metrics
        
        return results, concurrent_tracking
    
    def _generate_arrival_schedule(
        self,
        arrival_pattern: str,
        load_level: float,  # requests per second
        total_requests: int,
    ) -> List[float]:
        """Generate a schedule of request arrival times based on pattern and load level."""
        
        # Convert load level (req/sec) to mean interval between requests (sec)
        mean_interval = 1.0 / load_level if load_level > 0 else 0.1
        
        if arrival_pattern == "constant":
            # Constant rate arrivals
            return [i * mean_interval for i in range(total_requests)]
            
        elif arrival_pattern == "uniform":
            # Uniform random jitter around mean
            mean, jitter = self.config.arrival_rates["uniform"]
            # Scale jitter based on mean_interval to maintain relative jitter
            scaled_jitter = jitter * (mean_interval / mean)
            
            arrival_times = [0]
            for i in range(1, total_requests):
                # Add uniform jitter to the mean interval
                interval = np.random.uniform(
                    mean_interval - scaled_jitter,
                    mean_interval + scaled_jitter
                )
                arrival_times.append(arrival_times[-1] + interval)
            return arrival_times
            
        elif arrival_pattern == "poisson":
            # Poisson process (exponential inter-arrival times)
            arrival_times = [0]
            for i in range(1, total_requests):
                # Generate exponential random variable with mean = mean_interval
                interval = np.random.exponential(mean_interval)
                arrival_times.append(arrival_times[-1] + interval)
            return arrival_times
            
        elif arrival_pattern == "burst":
            # Burst arrivals
            burst_size, time_between_bursts = self.config.arrival_rates["burst"]
            
            # Scale the time between bursts based on the load level
            scaled_time = time_between_bursts * (mean_interval * burst_size)
            
            arrival_times = []
            num_bursts = (total_requests + burst_size - 1) // burst_size  # Ceiling division
            
            for b in range(num_bursts):
                burst_start = b * scaled_time
                for i in range(burst_size):
                    if len(arrival_times) < total_requests:
                        # Requests within a burst arrive with minimal delay
                        arrival_times.append(burst_start + i * 0.005)  # 5ms between requests in burst
            
            return arrival_times[:total_requests]  # Ensure we don't exceed total_requests
        
        else:
            raise ValueError(f"Unknown arrival pattern: {arrival_pattern}")
    
    def _call_generate(self, prompt: str) -> Tuple[Optional[Dict], int]:
        """Call the vLLM async server's OpenAI-compatible endpoint."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        
        # Format as OpenAI API request
        payload = {
            "model": "model",  # Placeholder
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            return response.json() if response.status_code == 200 else None, response.status_code
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None, 0
    
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


def save_results(
    trial_results: List[TrialResults],
    concurrent_data: Dict[Tuple[str, float, int], List[Tuple[float, int]]],
    output_dir: str
) -> None:
    """Save benchmark results to files and generate visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary results
    summary_data = [asdict(r) for r in trial_results]
    for result in summary_data:
        # Remove detailed request data from summary
        result.pop("requests", None)
    
    summary_file = output_dir / f"async_benchmark_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed request data for each trial
    for trial in trial_results:
        trial_key = f"{trial.arrival_pattern}_{trial.load_level}_{trial.prompt_length}_{trial.trial_id}"
        requests_file = output_dir / f"requests_{trial_key}_{timestamp}.csv"
        
        if trial.requests:
            df = pd.DataFrame([asdict(r) for r in trial.requests])
            df.to_csv(requests_file, index=False)
    
    # Create visualizations
    create_visualizations(trial_results, concurrent_data, output_dir, timestamp)
    
    logger.info(f"Results saved to {output_dir}")


def create_visualizations(
    trial_results: List[TrialResults],
    concurrent_data: Dict[Tuple[str, float, int], List[Tuple[float, int]]],
    output_dir: Path,
    timestamp: str
) -> None:
    """Create and save enhanced visualizations of the benchmark results."""
    # Set up the visualization style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})
    
    # Convert results to DataFrame for easier plotting
    summary_rows = []
    for trial in trial_results:
        summary_rows.append({
            "Pattern": trial.arrival_pattern,
            "Load (req/s)": trial.load_level,
            "Prompt Length": trial.prompt_length,
            "Mean Latency (ms)": trial.mean_latency,
            "P50 Latency (ms)": trial.p50_latency,
            "P90 Latency (ms)": trial.p90_latency,
            "P99 Latency (ms)": trial.p99_latency,
            "Throughput (req/s)": trial.throughput,
            "Success Rate": trial.success_rate,
            "Trial": trial.trial_id,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # 1. Latency vs Load Level by Pattern - Enhanced with multiple prompt lengths
    plt.figure(figsize=(15, 10))
    g = sns.FacetGrid(summary_df, col="Pattern", row="Prompt Length", 
                     margin_titles=True, height=3, aspect=1.5)
    g.map(sns.lineplot, "Load (req/s)", "Mean Latency (ms)", marker="o")
    g.add_legend()
    g.set_titles(col_template="Pattern: {col_name}", row_template="Prompt Length: {row_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"latency_vs_load_by_pattern_{timestamp}.png")
    
    # 2. Throughput vs Load Level by Pattern - Enhanced
    plt.figure(figsize=(15, 10))
    g = sns.FacetGrid(summary_df, col="Pattern", row="Prompt Length", 
                     margin_titles=True, height=3, aspect=1.5)
    g.map(sns.lineplot, "Load (req/s)", "Throughput (req/s)", marker="o")
    
    # Add reference line (y=x) for throughput vs load - ideal scaling
    for ax in g.axes.flat:
        # Get the x and y limits of the current subplot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Determine the common limit for x and y
        common_min = min(xlim[0], ylim[0])
        common_max = min(xlim[1], ylim[1])  # Use min to prevent reference line from extending too far
        
        # Plot the reference line
        ax.plot([common_min, common_max], [common_min, common_max], 'k--', alpha=0.5)
    
    g.add_legend()
    g.set_titles(col_template="Pattern: {col_name}", row_template="Prompt Length: {row_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"throughput_vs_load_by_pattern_{timestamp}.png")
    
    # 3. Latency Distribution Across Patterns - Enhanced
    # Group by prompt length for clearer comparisons
    for prompt_length in summary_df["Prompt Length"].unique():
        prompt_data = summary_df[summary_df["Prompt Length"] == prompt_length]
        
        plt.figure(figsize=(12, 8))
        
        # Create a plot showing different latency percentiles by pattern
        melted_data = pd.melt(
            prompt_data, 
            id_vars=["Pattern", "Load (req/s)"], 
            value_vars=["Mean Latency (ms)", "P50 Latency (ms)", "P90 Latency (ms)", "P99 Latency (ms)"],
            var_name="Metric", value_name="Latency (ms)"
        )
        
        sns.barplot(data=melted_data, x="Pattern", y="Latency (ms)", hue="Metric")
        plt.title(f"Latency Distribution by Pattern (Prompt Length: {prompt_length})")
        plt.xlabel("Arrival Pattern")
        plt.ylabel("Latency (ms)")
        plt.yscale("log")  # Log scale to better visualize percentiles
        plt.tight_layout()
        plt.savefig(output_dir / f"latency_distribution_prompt{prompt_length}_{timestamp}.png")
    
    # 4. Success Rate Analysis
    plt.figure(figsize=(12, 8))
    successful_df = summary_df.copy()
    # Convert success rate to percentage for better readability
    successful_df["Success Rate (%)"] = successful_df["Success Rate"] * 100
    
    sns.lineplot(
        data=successful_df, 
        x="Load (req/s)", 
        y="Success Rate (%)", 
        hue="Pattern", 
        style="Pattern",
        markers=True, 
        dashes=False
    )
    
    plt.title("Success Rate vs Load Level by Pattern")
    plt.xlabel("Target Load (requests/second)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)  # Set y-axis limits for percentage
    plt.tight_layout()
    plt.savefig(output_dir / f"success_rate_vs_load_{timestamp}.png")
    
    # 5. Concurrent Requests Visualization - Enhanced
    # Sample a few representative patterns/loads for clarity
    pattern_samples = list(summary_df["Pattern"].unique())
    load_samples = sorted(summary_df["Load (req/s)"].unique())
    
    # Choose middle load level
    if len(load_samples) >= 3:
        load_samples = [load_samples[len(load_samples) // 2]]
    
    # Create a grid of concurrent request plots
    plt.figure(figsize=(15, 10))
    subplot_idx = 1
    
    for pattern in pattern_samples:
        for load in load_samples:
            # Find any trial with this pattern/load combination
            for trial_id in range(10):  # Check first 10 trial IDs
                key = (pattern, load, trial_id)
                if key in concurrent_data:
                    plt.subplot(len(pattern_samples), len(load_samples), subplot_idx)
                    
                    df = pd.DataFrame(concurrent_data[key], columns=["Time (s)", "Concurrent Requests"])
                    plt.step(df["Time (s)"], df["Concurrent Requests"], where="post")
                    plt.title(f"{pattern} @ {load} req/s")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Concurrent")
                    plt.grid(True)
                    break
            
            subplot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_dir / f"concurrent_requests_grid_{timestamp}.png")
    
    # 6. Combined Performance Summary
    plt.figure(figsize=(15, 8))
    
    # Prepare data - average across trials
    avg_data = summary_df.groupby(["Pattern", "Load (req/s)", "Prompt Length"]).agg({
        "Mean Latency (ms)": "mean",
        "Throughput (req/s)": "mean",
        "Success Rate": "mean"
    }).reset_index()
    
    # Create a normalized score for each metric
    for metric in ["Mean Latency (ms)", "Throughput (req/s)", "Success Rate"]:
        if metric == "Mean Latency (ms)":
            # Lower is better for latency
            max_val = avg_data[metric].max()
            avg_data[f"{metric} Score"] = 1 - (avg_data[metric] / max_val)
        else:
            # Higher is better for throughput and success rate
            max_val = avg_data[metric].max()
            if max_val > 0:
                avg_data[f"{metric} Score"] = avg_data[metric] / max_val
    
    # Combined score (equal weighting)
    avg_data["Combined Score"] = (
        avg_data["Mean Latency (ms) Score"] + 
        avg_data["Throughput (req/s) Score"] + 
        avg_data["Success Rate Score"]
    ) / 3
    
    # Plot the combined score for each pattern
    plt.subplot(1, 2, 1)
    sns.barplot(data=avg_data, x="Pattern", y="Combined Score", hue="Load (req/s)")
    plt.title("Combined Performance Score by Pattern")
    plt.xlabel("Arrival Pattern")
    plt.ylabel("Combined Score (higher is better)")
    
    # Summary bubble chart - size by success rate
    plt.subplot(1, 2, 2)
    for pattern in avg_data["Pattern"].unique():
        pattern_data = avg_data[avg_data["Pattern"] == pattern]
        plt.scatter(
            pattern_data["Mean Latency (ms)"], 
            pattern_data["Throughput (req/s)"],
            s=pattern_data["Success Rate"] * 300,  # Scale bubble size
            alpha=0.7,
            label=pattern
        )
    
    plt.xscale("log")
    plt.title("Performance Summary")
    plt.xlabel("Mean Latency (ms) - log scale")
    plt.ylabel("Throughput (req/s)")
    plt.legend(title="Pattern")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"combined_performance_{timestamp}.png")
    
    logger.info(f"Enhanced visualizations saved to {output_dir}")


def run_async_benchmarks(config: AsyncBenchmarkConfig) -> Tuple[List[TrialResults], Dict]:
    """Run all async benchmarks according to the configuration."""
    # Initialize server
    server = AsyncServerManager(
        model=config.model,
        port=DEFAULT_PORT,
        tensor_parallel=config.tensor_parallel_size
    )
    
    trial_results = []
    concurrent_data = {}  # Store concurrent request tracking
    
    try:
        # Start server
        server.start()
        
        # Initialize client
        client = AsyncBenchmarkClient(
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
            config=config
        )
        
        # Run benchmarks for each pattern/load combination
        for arrival_pattern in config.arrival_patterns:
            for load_level in config.load_levels:
                for prompt_length in config.prompt_lengths:
                    for trial in range(config.trials_per_config):
                        logger.info(
                            f"Running async benchmark with {arrival_pattern} pattern at {load_level} req/s: "
                            f"prompt={prompt_length}, trial={trial+1}/{config.trials_per_config}"
                        )
                        
                        # Run the benchmark
                        trial_result, concurrent_tracking = client.run_benchmark(
                            prompt_length=prompt_length,
                            arrival_pattern=arrival_pattern,
                            load_level=load_level,
                            trial_id=trial,
                        )
                        
                        # Store results
                        trial_results.append(trial_result)
                        concurrent_data[(arrival_pattern, load_level, trial)] = concurrent_tracking
                        
                        # Display summary metrics for this trial
                        logger.info(
                            f"Trial completed - Mean latency: {trial_result.mean_latency:.2f} ms, "
                            f"Throughput: {trial_result.throughput:.2f} req/s, "
                            f"Success rate: {trial_result.success_rate:.2%}"
                        )
    
    finally:
        # Always stop the server
        server.stop()
    
    return trial_results, concurrent_data


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM async server with various request arrival patterns")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Model to benchmark")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[10, 100, 500], 
                       help="Prompt lengths (tokens) to test")
    parser.add_argument("--load_levels", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0], 
                       help="Load levels (requests per second) to test")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials per configuration")
    parser.add_argument("--total_requests", type=int, default=DEFAULT_NUM_REQUESTS, 
                       help="Number of requests per trial")
    parser.add_argument("--output_dir", type=str, default="async_benchmark_results", 
                       help="Directory to save results")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_wikitext", action="store_true", default=True,
                       help="Use WikiText-103 data for realistic prompts")
    
    args = parser.parse_args()
    
    # Create config
    config = AsyncBenchmarkConfig(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_lengths=args.prompt_lengths,
        trials_per_config=args.trials,
        total_requests=args.total_requests,
        use_wikitext=args.use_wikitext,
        load_levels=args.load_levels,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )
    
    # Run benchmarks
    trial_results, concurrent_data = run_async_benchmarks(config)
    
    # Save results and generate visualizations
    save_results(trial_results, concurrent_data, args.output_dir)


if __name__ == "__main__":
    main() 