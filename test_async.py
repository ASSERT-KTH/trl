import requests
import torch
from transformers import AutoModelForCausalLM
import json
import time

# Server configuration
HOST = "0.0.0.0"
SERVER_PORT = 8000
GROUP_PORT = 51216

# Base URL
base_url = f"http://{HOST}:{SERVER_PORT}"

def test_health():
    """Test the health endpoint"""
    url = f"{base_url}/health/"
    response = requests.get(url)
    print(f"Health check: {response.status_code}")
    return response.ok

# def test_tensor_parallel_size():
#     """Test getting tensor parallel size"""
#     url = f"{base_url}/get_tensor_parallel_size/"
#     response = requests.get(url)
#     result = response.json()
#     print(f"Tensor parallel size: {result['tensor_parallel_size']}")
#     return result['tensor_parallel_size']

def test_get_world_size():
    """Test getting world size"""
    url = f"{base_url}/get_world_size/"
    response = requests.get(url)
    print(f"World size: {response.json()['world_size']}")
    return response.json()['world_size']

def test_init_communicator(tensor_parallel_size):
    """Test initializing the communicator"""
    url = f"{base_url}/init_communicator/"
    world_size = tensor_parallel_size + 1
    payload = {"host": "0.0.0.0", "port": GROUP_PORT, "world_size": world_size}
    response = requests.post(url, json=payload)
    print(f"Init communicator: {response.status_code}")
    return response.ok

def test_update_named_param():
    """Test updating a named parameter"""
    # Load a small model to get a parameter
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="cuda")
    
    # Get a sample parameter
    sample_param = next(iter(model.named_parameters()))
    name, weights = sample_param
    
    url = f"{base_url}/update_named_param/"
    payload = {
        "name": name,
        "dtype": str(weights.dtype),
        "shape": weights.shape
    }
    response = requests.post(url, json=payload)
    print(f"Update named param: {response.status_code}")
    return response.ok

def test_reset_prefix_cache():
    """Test resetting prefix cache"""
    url = f"{base_url}/reset_prefix_cache/"
    response = requests.post(url)
    print(response)
    print(f"Reset prefix cache: {response.status_code}")
    return response.ok

def test_close_communicator():
    """Test closing the communicator"""
    url = f"{base_url}/close_communicator/"
    response = requests.post(url)
    print(f"Close communicator: {response.status_code}")
    return response.ok

def test_chat_completions():
    """Test the chat completions endpoint for async client"""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about AI in 3 sentences."}
        ],
        "max_tokens": 100
    }
    response = requests.post(url, json=payload)
    print(f"Chat completions: {response.status_code}")
    if response.ok:
        print(json.dumps(response.json(), indent=2))
    return response.ok

def run_all_tests():
    """Run all tests sequentially"""
    print("Starting tests...")
    
    # Test server health
    if not test_health():
        print("Server health check failed! Make sure the server is running.")
        return False
    
    # if not (tp_size:=test_tensor_parallel_size()):
    #     print("Failed to get tensor parallel size!")
    #     return False

    if not (world_size:=test_get_world_size()):
        print("Failed to get world size!")
        return False
    
    # Test communicator initialization
    if not test_init_communicator(world_size):
        print("Failed to initialize communicator!")
        return False
    
    # Test parameter updates
    if not test_update_named_param():
        print("Failed to update named parameter!")
        return False
    
    # Test prefix cache reset
    if not test_reset_prefix_cache():
        print("Failed to reset prefix cache!")
        return False
    
    # Test generation endpoint
    test_chat_completions()
    
    # Test closing communicator
    if not test_close_communicator():
        print("Failed to close communicator!")
        return False
    
    print("\nAll tests completed!")
    return True

if __name__ == "__main__":
    # test_reset_prefix_cache()
    run_all_tests()
    # test_init_communicator(1)
    # test_chat_completions()