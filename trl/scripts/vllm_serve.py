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

import argparse
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)


if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        import sys
        sys.stderr.write(f"\n\n!!! WeightSyncWorkerExtension.init_communicator START: host={host}, port={port}, world_size={world_size} !!!\n\n")
        sys.stderr.flush()
        
        if self.pynccl_comm is not None:
            sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.init_communicator ERROR: Weight update group already initialized !!!\n\n")
            sys.stderr.flush()
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank
        sys.stderr.write(f"\n\n!!! WeightSyncWorkerExtension.init_communicator: Current rank={rank} !!!\n\n")
        sys.stderr.flush()

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.init_communicator: Creating StatelessProcessGroup... !!!\n\n")
        sys.stderr.flush()
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.init_communicator: StatelessProcessGroup created !!!\n\n")
        sys.stderr.flush()

        # Initialize the NCCL-based communicator for weight synchronization.
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.init_communicator: Initializing PyNcclCommunicator... !!!\n\n")
        sys.stderr.flush()
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.init_communicator: PyNcclCommunicator initialized !!!\n\n")
        sys.stderr.flush()

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1
        sys.stderr.write(f"\n\n!!! WeightSyncWorkerExtension.init_communicator: Client rank set to {self.client_rank} !!!\n\n")
        sys.stderr.flush()
        
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.init_communicator COMPLETE !!!\n\n")
        sys.stderr.flush()

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        import sys
        sys.stderr.write(f"\n\n!!! WeightSyncWorkerExtension.update_named_param START: name={name}, dtype={dtype}, shape={shape} !!!\n\n")
        sys.stderr.flush()
        
        if self.pynccl_comm is None:
            sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.update_named_param ERROR: Communicator not initialized !!!\n\n")
            sys.stderr.flush()
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        sys.stderr.write(f"\n\n!!! WeightSyncWorkerExtension.update_named_param: Allocating tensor on device={self.device} !!!\n\n")
        sys.stderr.flush()
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        
        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        sys.stderr.write(f"\n\n!!! WeightSyncWorkerExtension.update_named_param: Broadcasting weights from client_rank={self.client_rank} !!!\n\n")
        sys.stderr.flush()
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.update_named_param: Broadcast complete !!!\n\n")
        sys.stderr.flush()
        
        self.pynccl_comm.group.barrier()
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.update_named_param: Barrier complete !!!\n\n")
        sys.stderr.flush()

        # Load the received weights into the model.
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.update_named_param: Loading weights into model !!!\n\n")
        sys.stderr.flush()
        self.model_runner.model.load_weights(weights=[(name, weight)])
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.update_named_param: Weights loaded successfully !!!\n\n")
        sys.stderr.flush()
        
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.update_named_param COMPLETE !!!\n\n")
        sys.stderr.flush()

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """
        import sys
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.close_communicator START !!!\n\n")
        sys.stderr.flush()

        if self.pynccl_comm is not None:
            sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.close_communicator: Deleting PyNcclCommunicator !!!\n\n")
            sys.stderr.flush()
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None
            sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.close_communicator: Communicator deleted !!!\n\n")
            sys.stderr.flush()
        else:
            sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.close_communicator: No communicator to close !!!\n\n")
            sys.stderr.flush()
        
        sys.stderr.write("\n\n!!! WeightSyncWorkerExtension.close_communicator COMPLETE !!!\n\n")
        sys.stderr.flush()


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )


def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    # Set required environment variables for DP to work with vLLM
    import sys
    sys.stderr.write(f"\n\n!!! llm_worker START: data_parallel_rank={data_parallel_rank}, master_port={master_port} !!!\n\n")
    sys.stderr.flush()
    
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    sys.stderr.write(f"\n\n!!! llm_worker: Set environment variables !!!\n\n")
    sys.stderr.flush()

    sys.stderr.write(f"\n\n!!! llm_worker: Loading model {script_args.model} !!!\n\n")
    sys.stderr.flush()
    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
    )
    sys.stderr.write(f"\n\n!!! llm_worker: Model loaded successfully !!!\n\n")
    sys.stderr.flush()

    # Send ready signal to parent process
    sys.stderr.write(f"\n\n!!! llm_worker: Sending ready signal to parent !!!\n\n")
    sys.stderr.flush()
    connection.send({"status": "ready"})
    sys.stderr.write(f"\n\n!!! llm_worker: Ready signal sent !!!\n\n")
    sys.stderr.flush()

    while True:
        # Wait for commands from the parent process
        sys.stderr.write(f"\n\n!!! llm_worker: Waiting for commands from parent !!!\n\n")
        sys.stderr.flush()
        try:
            command = connection.recv()
            sys.stderr.write(f"\n\n!!! llm_worker: Received command: {command['type']} - {command.get('method', '')} !!!\n\n")
            sys.stderr.flush()
        except KeyboardInterrupt:
            sys.stderr.write(f"\n\n!!! llm_worker: Received KeyboardInterrupt, closing communicator !!!\n\n")
            sys.stderr.flush()
            llm.collective_rpc(method="close_communicator")
            break

        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(llm, method_name)
            sys.stderr.write(f"\n\n!!! llm_worker: Executing method {method_name} !!!\n\n")
            sys.stderr.flush()
            result = method(*args, **kwargs)
            sys.stderr.write(f"\n\n!!! llm_worker: Method {method_name} completed !!!\n\n")
            sys.stderr.flush()
            if command["type"] == "call":
                sys.stderr.write(f"\n\n!!! llm_worker: Sending result back to parent !!!\n\n")
                sys.stderr.flush()
                connection.send(result)
                sys.stderr.write(f"\n\n!!! llm_worker: Result sent !!!\n\n")
                sys.stderr.flush()
        elif command["type"] == "shutdown":
            sys.stderr.write(f"\n\n!!! llm_worker: Received shutdown command !!!\n\n")
            sys.stderr.flush()
            break
            
    sys.stderr.write(f"\n\n!!! llm_worker END !!!\n\n")
    sys.stderr.flush()


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
        [[1, 2, 3], [4, 5, 6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
        [[1, 2], [3, 4], [5], [6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
        [[1], [2], [3], [4], [5], [6], [], []]
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f"Process {process} is still alive after 10 seconds, attempting to terminate...")
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    app = FastAPI(lifespan=lifespan)

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        import sys
        sys.stderr.write(f"\n\n!!! Endpoint /init_communicator/ START: host={request.host}, port={request.port}, world_size={request.world_size} !!!\n\n")
        sys.stderr.flush()
        
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        sys.stderr.write(f"\n\n!!! Endpoint /init_communicator/: Calculated world_size={world_size} !!!\n\n")
        sys.stderr.flush()

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, world_size)}
        sys.stderr.write(f"\n\n!!! Endpoint /init_communicator/: Sending rpc to {len(connections)} workers !!!\n\n")
        sys.stderr.flush()
        for i, connection in enumerate(connections):
            sys.stderr.write(f"\n\n!!! Endpoint /init_communicator/: Sending to worker {i} !!!\n\n")
            sys.stderr.flush()
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
            sys.stderr.write(f"\n\n!!! Endpoint /init_communicator/: Sent to worker {i} !!!\n\n")
            sys.stderr.flush()

        sys.stderr.write("\n\n!!! Endpoint /init_communicator/ COMPLETE !!!\n\n")
        sys.stderr.flush()
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        import sys
        sys.stderr.write(f"\n\n!!! Endpoint /update_named_param/ START: name={request.name}, dtype={request.dtype}, shape={request.shape} !!!\n\n")
        sys.stderr.flush()
        
        # The function update_named_param is called this way: update_named_param("name", torch.float32, (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        kwargs = {"method": "update_named_param", "args": (request.name, dtype, tuple(request.shape))}
        sys.stderr.write(f"\n\n!!! Endpoint /update_named_param/: Sending rpc to {len(connections)} workers !!!\n\n")
        sys.stderr.flush()
        for i, connection in enumerate(connections):
            sys.stderr.write(f"\n\n!!! Endpoint /update_named_param/: Sending to worker {i} !!!\n\n")
            sys.stderr.flush()
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
            sys.stderr.write(f"\n\n!!! Endpoint /update_named_param/: Sent to worker {i} !!!\n\n")
            sys.stderr.flush()

        sys.stderr.write("\n\n!!! Endpoint /update_named_param/ COMPLETE !!!\n\n")
        sys.stderr.flush()
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        import sys
        import time
        start_time = time.time()
        sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/ START at {start_time:.6f} !!!\n\n")
        sys.stderr.flush()
        
        sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: Sending to {len(connections)} workers !!!\n\n")
        sys.stderr.flush()
        
        # First send to all workers
        send_times = []
        for i, connection in enumerate(connections):
            send_start = time.time()
            sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: Sending to worker {i} at {send_start:.6f} !!!\n\n")
            sys.stderr.flush()
            connection.send({"type": "call", "method": "reset_prefix_cache"})
            send_end = time.time()
            send_times.append(send_end - send_start)
            sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: Sent to worker {i} at {send_end:.6f} (took {send_times[-1]:.6f}s) !!!\n\n")
            sys.stderr.flush()
            
        # Then wait for all results
        sys.stderr.write("\n\n!!! Endpoint /reset_prefix_cache/: Waiting for all results !!!\n\n")
        sys.stderr.flush()
        all_outputs = []
        recv_times = []
        for i, connection in enumerate(connections):
            recv_start = time.time()
            sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: Waiting for worker {i} at {recv_start:.6f} !!!\n\n")
            sys.stderr.flush()
            output = connection.recv()
            recv_end = time.time()
            recv_times.append(recv_end - recv_start)
            all_outputs.append(output)
            sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: Received from worker {i} at {recv_end:.6f} (took {recv_times[-1]:.6f}s): {output} !!!\n\n")
            sys.stderr.flush()
            
        success = all(output for output in all_outputs)
        end_time = time.time()
        total_time = end_time - start_time
        sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: Overall success={success} !!!\n\n")
        sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/: TIMING SUMMARY !!!\n")
        sys.stderr.write(f"!!! Total time: {total_time:.6f}s !!!\n")
        sys.stderr.write(f"!!! Send times: {send_times} !!!\n")
        sys.stderr.write(f"!!! Receive times: {recv_times} !!!\n\n")
        sys.stderr.flush()
        
        sys.stderr.write(f"\n\n!!! Endpoint /reset_prefix_cache/ COMPLETE at {end_time:.6f} (took {total_time:.6f}s) !!!\n\n")
        sys.stderr.flush()
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        import sys
        sys.stderr.write("\n\n!!! Endpoint /close_communicator/ START !!!\n\n")
        sys.stderr.flush()
        
        kwargs = {"method": "close_communicator"}
        sys.stderr.write(f"\n\n!!! Endpoint /close_communicator/: Sending to {len(connections)} workers !!!\n\n")
        sys.stderr.flush()
        for i, connection in enumerate(connections):
            sys.stderr.write(f"\n\n!!! Endpoint /close_communicator/: Sending to worker {i} !!!\n\n")
            sys.stderr.flush()
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
            sys.stderr.write(f"\n\n!!! Endpoint /close_communicator/: Sent to worker {i} !!!\n\n")
            sys.stderr.flush()
            
        sys.stderr.write("\n\n!!! Endpoint /close_communicator/ COMPLETE !!!\n\n")
        sys.stderr.flush()
        return {"message": "Request received, closing communicator"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
