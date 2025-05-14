# My current take is that we dont need an HF wrapper for SGLang. Just launch it directly. SGLang client handles the weight syncing and stuff




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

# import os
# import sys
# import logging
# import asyncio
# import argparse
# from typing import Optional
# from dataclasses import dataclass, field

# import torch

# from trl.import_utils import is_sglang_available, is_fastapi_available, is_pydantic_available, is_uvicorn_available


# if is_sglang_available():
#     from sglang.srt.entrypoints.http_server import launch_server
#     from sglang.srt.server_args import prepare_server_args
#     from sglang.srt.utils import kill_process_tree


# logger = logging.getLogger(__name__)


# @dataclass
# class ScriptArguments:
#     r"""
#     Arguments for the script.

#     Args:
#         model (`str`):
#             Model name or path to load the model from.
#         tensor_parallel_size (`int`, *optional*, defaults to `1`):
#             Number of tensor parallel workers to use.
#         data_parallel_size (`int`, *optional*, defaults to `1`):
#             Number of data parallel workers to use.
#         host (`str`, *optional*, defaults to `"0.0.0.0"`):
#             Host address to run the server on.
#         port (`int`, *optional*, defaults to `8000`):
#             Port to run the server on.
#         gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
#             Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
#             device dedicated to generation powered by SGLang. Higher values will increase the KV cache size and thus
#             improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
#             during initialization.
#         dtype (`str`, *optional*, defaults to `"auto"`):
#             Data type to use for SGLang generation. If set to `"auto"`, the data type will be automatically determined
#             based on the model configuration. Find the supported values in the SGLang documentation.
#         max_model_len (`int` or `None`, *optional*, defaults to `None`):
#             If set, the `max_model_len` to use for SGLang. This can be useful when running with reduced
#             `gpu_memory_utilization`, leading to a reduced KV cache size. If not set, SGLang will use the model
#             context size, which might be much larger than the KV cache, leading to inefficiencies.
#         enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
#             Whether to enable prefix caching in SGLang. If set to `True`, ensure that the model and the hardware support
#             this feature.
#         tool_call_parser (`str`, *optional*, defaults to `"hermes"`):
#             The tool call parser to use. Only compatible with /v1/chat/completions endpoint.
#     """

#     model: str = field(metadata={"help": "Model name or path to load the model from."})
#     tensor_parallel_size: int = field(
#         default=1,
#         metadata={"help": "Number of tensor parallel workers to use."},
#     )
#     data_parallel_size: int = field(
#         default=1,
#         metadata={"help": "Number of data parallel workers to use."},
#     )
#     host: str = field(
#         default="0.0.0.0",
#         metadata={"help": "Host address to run the server on."},
#     )
#     port: int = field(
#         default=8000,
#         metadata={"help": "Port to run the server on."},
#     )
#     gpu_memory_utilization: float = field(
#         default=0.9,
#         metadata={
#             "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
#             "cache on the device dedicated to generation powered by SGLang. Higher values will increase the KV cache "
#             "size and thus improve the model's throughput. However, if the value is too high, it may cause "
#             "out-of-memory (OOM) errors during initialization."
#         },
#     )
#     dtype: str = field(
#         default="auto",
#         metadata={
#             "help": "Data type to use for SGLang generation. If set to 'auto', the data type will be automatically "
#             "determined based on the model configuration. Find the supported values in the SGLang documentation."
#         },
#     )
#     max_model_len: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": "If set, the `max_model_len` to use for SGLang. This can be useful when running with reduced "
#             "`gpu_memory_utilization`, leading to a reduced KV cache size. If not set, SGLang will use the model "
#             "context size, which might be much larger than the KV cache, leading to inefficiencies."
#         },
#     )
#     enable_prefix_caching: Optional[bool] = field(
#         default=None,
#         metadata={
#             "help": "Whether to enable prefix caching in SGLang. If set to `True`, ensure that the model and the "
#             "hardware support this feature."
#         },
#     )
#     tool_call_parser: str = field(
#         default="qwen25",
#         metadata={
#             "help": "The tool call parser to use. Only compatible with /v1/chat/completions endpoint."
#         },
#     )




#     # @app.get("/get_tensor_parallel_size/")
#     # async def get_tensor_parallel_size():
#     #     """
#     #     Retrieves the tensor parallel size from the LLM engine.

#     #     Returns:
#     #         `dict`:
#     #             A dictionary containing the tensor parallel size.

#     #     Example response:
#     #     ```json
#     #     {"tensor_parallel_size": 8}
#     #     ```
#     #     """
#     #     return {"tensor_parallel_size": args.tensor_parallel_size}

  

#     # @app.post("/reset_prefix_cache/")
#     # async def reset_prefix_cache():
#     #     """Drop the entire KV/prefix cache - matches vLLM `reset_prefix_cache`."""
#     #     async with engine_lock:
#     #         engine.reset_kv_cache()
#     #     return {"message": "prefix cache cleared"}



# def make_parser(subparsers: argparse._SubParsersAction = None):
#     if subparsers is not None:
#         parser = subparsers.add_parser("sglang-serve", help="Run the SGLang serve script", dataclass_types=ScriptArguments)
#     else:
#         parser = TrlParser(ScriptArguments)
#     return parser


# if __name__ == "__main__":
#     server_args = prepare_server_args(sys.argv[1:])

#     try:
#         launch_server(server_args)
#     finally:
#         kill_process_tree(os.getpid(), include_parent=False)


# # python -m sglang.launch_server --model-path qwen/qwen3-4b-awq --tool-call-parser qwen25 --quantization awq_marlin --reasoning-parser qwen3 --context-length 8096