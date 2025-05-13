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

import logging
import asyncio
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


from trl.import_utils import is_sglang_available, is_fastapi_available, is_pydantic_available, is_uvicorn_available


if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available(): 
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_sglang_available():
    from sglang import Engine


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
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
            device dedicated to generation powered by SGLang. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for SGLang generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the SGLang documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for SGLang. This can be useful when running with reduced
            `gpu_memory_utilization`, leading to a reduced KV cache size. If not set, SGLang will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in SGLang. If set to `True`, ensure that the model and the hardware support
            this feature.
        tool_call_parser (`str`, *optional*, defaults to `"hermes"`):
            The tool call parser to use. Only compatible with /v1/chat/completions endpoint.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
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
            "cache on the device dedicated to generation powered by SGLang. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for SGLang generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the SGLang documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for SGLang. This can be useful when running with reduced "
            "`gpu_memory_utilization`, leading to a reduced KV cache size. If not set, SGLang will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in SGLang. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    tool_call_parser: str = field(
        default="qwen25",
        metadata={
            "help": "The tool call parser to use. Only compatible with /v1/chat/completions endpoint."
        },
    )


def build_engine(args: ScriptArguments) -> Engine:
    """Instantiate SGLang engine once for the whole server lifetime."""
    engine = Engine(
        model_path=args.model,
        max_seq_len=args.max_seq_len,
        tp_size=args.tensor_parallel_size,
        dp_size=args.data_parallel_size,
        dtype=args.dtype,
        context_length=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        tool_call_parser=args.tool_call_parser,
    )
    logger.info("Loaded model %s", args.model)
    return engine


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI()
    reload_lock = asyncio.Lock()  # ensures only one weight‑reload at a time

    # class GenRequest(BaseModel):
    #     prompts: list[str]
    #     max_tokens: int = 32
    #     temperature: float = 0.8
    #     top_p: float = 1.0
    #     n: int = 1

    # class GenResponse(BaseModel):
    #     completions: list[list[int]]  # token ids

    # @app.post("/generate/", response_model=GenResponse)
    # async def generate(req: GenRequest):
    #     async with reload_lock:  # block if a reload is running
    #         outs = engine.generate(
    #             req.prompts,
    #             max_tokens=req.max_tokens,
    #             temperature=req.temperature,
    #             top_p=req.top_p,
    #             n=req.n,
    #         )
    #     ids = [[tok.id for tok in o] for o in outs]
    #     return {"completions": ids}

        # class UpdateReq(BaseModel):
        #     checkpoint_path: Path

    # @app.post("/update_model/")
    # async def update_model(req: UpdateReq):
    #     if not req.checkpoint_path.exists():
    #         raise HTTPException(status_code=404, detail="checkpoint file not found")
    #     async with reload_lock:
    #         logger.info("Reloading weights from %s", req.checkpoint_path)
    #         engine.update_weights_from_disk(str(req.checkpoint_path))
    #         engine.reset_kv_cache()
    #     return {"message": "weights reloaded"}

    # @app.post("/reset_kv_cache/")
    # async def reset_kv_cache():
    #     async with reload_lock:
    #         engine.reset_kv_cache()
    #     return {"message": "kv cache cleared"}

    return app

def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("sglang-serve", help="Run the SGLang serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser

def main():
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    engine = build_engine(script_args)
    app = create_app(engine)
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


if __name__ == "__main__":
    main()




# python -m sglang.launch_server --model-path qwen/qwen3-4b-awq --tool-call-parser qwen25 --quantization awq_marlin --reasoning-parser qwen3 --context-length 8096