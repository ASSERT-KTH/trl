# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import time
import tempfile
import logging

import torch
from torch import nn

from vllm_client import Client, GenerationResult
from ..import_utils import is_requests_available

if is_requests_available():
    import requests
    from requests import ConnectionError

logger = logging.getLogger(__name__)


class SGLangClient(Client):
    def __init__(self, host: str, port: int, tp_size: int = 1):
        """
        Args:
            host (str): server hostname or IP
            port (int): server port
            tp_size (int): tensor-parallel size (for payload repetition)
        """
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.tp_size = tp_size

    def init_communicator(self, *args, **kwargs):
        # Not needed: using HTTP/disk-read based reloads
        return

    def close_communicator(self):
        # Not needed
        return

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Push a single tensor update (e.g. LoRA delta) to the server.
        Uses SGLang's `update_weights_from_tensor` endpoint.
        """
        # Serialize the tensor for HTTP transport
        # We send the same payload to each TP shard
        np_tensor = weights.detach().cpu().numpy()
        # Base64 or pickled serialization could be used; here we send raw list
        payload = {
            "named_tensors": [
                {"name": name, "data": np_tensor.tolist(), "dtype": str(weights.dtype), "shape": list(weights.shape)}
                for _ in range(self.tp_size)
            ],
            "flush_cache": False,  # this method might be called multiple times, we call reset_prefix_cache when done
        }
        url = f"{self.base_url}/update_weights_from_tensor"
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()
        logger.info("update_named_param: %s", resp.json())

    def update_model_params(self, model: nn.Module):
        """
        Save the full model to a temporary checkpoint then tell server to reload.
        """
        # Write to temp file
        fd, path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        # Using PyTorch save; convert to safetensors if needed
        torch.save(model.state_dict(), path)
        try:
            url = f"{self.base_url}/update_model"
            resp = self.session.post(url, json={"checkpoint_path": path})
            resp.raise_for_status()
            logger.info("update_model_params: %s", resp.json())
        finally:
            os.remove(path)

    def reset_prefix_cache(self):  # name kept for compatibility
        """
        Clear KV/prefix caches on the server (matching vLLM signature).
        """
        url = f"{self.base_url}/reset_kv_cache"
        resp = self.session.post(url)
        resp.raise_for_status()
        logger.info("reset_prefix_cache: %s", resp.json())