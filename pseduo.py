# I am making a PR with the behaviour marked as "new", we should make minimal changes to the old codebase with NO breaking changes
# code marked as "new" can be changed if we find a nicer way to do it

# The user either launches a synchronous-batched vllm server or an asynchrounous-dynamic-batching vllm server
# trl vllm-serve --model Qwen/Qwen3-8B-Instruct  # old
# trl vllm-serve-async --model Qwen/Qwen3-8B-Instruct  # new, fully featurer / openai compatible api server

# This class implements weight sybcing between the training and inference servers
# this should be used IF vllm-serve is launched
class VLLMClient:  # old
    # weight syncing stuff
    
    # would prefer that t
    def generate(self, data: list[dict], **kwargs) -> list[dict]:  # old, slightly changed
        ...
        
from abc import ABC, abstractmethod

# the async server in and of itself is not useful since it is slightly slower than the old server
# this should be used IF vllm-serve-async is launched
class AsyncVLLMClient(VLLMClient, ABC):  # new
    @abstractmethod
    def generate(self, data: list[dict], **kwargs) -> list[dict]:
        ...
    
# users can extend either class to customize the behaviour
# customizing AsyncVLLMClient is powerful, if you have any AI enabled product pipeline
# with some way to measure rewards, then you can directly do RL training with trl with minimal code.
# Currently only GRPO, PPO in the future could be cool.

# example class that extends AsyncVLLMClient, generates rollout of using the aider coding agent
# launches multiple aider agents in paralell, each making an arbitrary number of API calls to the async vllm server
# when it ultimately finishes, it returns a list of dicts which must contain "completion"
# all entries in this dict are fed to the reward function so you can use any information you want
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
class AiderClient(AsyncVLLMClient):
    def generate(self, data: list[dict], **kwargs) -> list[dict]:
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self.launch_one_agent, data)
        return results

from trl import GRPOTrainer
from datasets import load_dataset

trainer = GRPOTrainer(
    model="Qwen/Qwen3-8B-Instruct",
    dataset=load_dataset("princeton-nlp/SWE-Gym"),
    reward_funcs=["some reward function"],
    client=AiderClient(),  # new, overrides a few lines inside the trainer which would instantiate the old VLLMClient
)

trainer.train()  # now we are essentially training a coding agent on SWE-Gym, before this change, training such a model would be WAY harder