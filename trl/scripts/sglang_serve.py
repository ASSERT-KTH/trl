"""Minimal FastAPI wrapper around SGLang Engine

Exposes three endpoints:
  • POST /generate            – basic text generation (OpenAI‑style is trivial to add later)
  • POST /update_model        – synchronous weight reload from a checkpoint path
  • POST /reset_kv_cache      – clear KV cache (optional but handy for evaluation)

Designed for a single‑node, multi‑GPU SGLang engine without extra data‑parallel plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from trl.import_utils import is_sglang_available, is_fastapi_available, is_pydantic_available, is_uvicorn_available

if is_fastapi_available():
    from fastapi import FastAPI, HTTPException

if is_pydantic_available(): 
    from pydantic import BaseModel

if is_uvicorn_available():
    import uvicorn

if is_sglang_available():
    import sglang as sg


logger = logging.getLogger(__name__)


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
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
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
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it mse the KV cache "
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



def build_engine(args: ScriptArguments) -> sg.Engine:
    """Instantiate SGLang engine once for the whole server lifetime."""
    engine = sg.Engine(
        model_path=args.model,
        max_seq_len=args.max_seq_len,
    )
    log.info("Loaded model %s", args.model)
    return engine


def create_app(engine: "sg.Engine") -> FastAPI:
    app = FastAPI()
    reload_lock = asyncio.Lock()  # ensures only one weight‑reload at a time

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    class GenRequest(BaseModel):
        prompts: List[str]
        max_tokens: int = 32
        temperature: float = 0.8
        top_p: float = 1.0
        n: int = 1

    class GenResponse(BaseModel):
        completions: List[List[int]]  # token ids

    @app.post("/generate/", response_model=GenResponse)
    async def generate(req: GenRequest):
        async with reload_lock:  # block if a reload is running
            outs = engine.generate(
                req.prompts,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                n=req.n,
            )
        ids = [[tok.id for tok in o] for o in outs]
        return {"completions": ids}

    class UpdateReq(BaseModel):
        checkpoint_path: Path

    @app.post("/update_model/")
    async def update_model(req: UpdateReq):
        if not req.checkpoint_path.exists():
            raise HTTPException(status_code=404, detail="checkpoint file not found")
        async with reload_lock:
            log.info("Reloading weights from %s", req.checkpoint_path)
            engine.update_weights_from_disk(str(req.checkpoint_path))
            engine.reset_kv_cache()
        return {"message": "weights reloaded"}

    @app.post("/reset_kv_cache/")
    async def reset_kv_cache():
        async with reload_lock:
            engine.reset_kv_cache()
        return {"message": "kv cache cleared"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve SGLang engine over FastAPI", prog="sglang-serve")
    parser.add_argument("model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--log-level", default="info")
    args = Args(**vars(parser.parse_args()))

    engine = build_engine(args)
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
