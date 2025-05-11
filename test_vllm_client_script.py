import time
from transformers import AutoModelForCausalLM
from trl.extras.vllm_client import VLLMClient

# Configuration
MODEL_ID = "Qwen/Qwen2.5-0.6B" # Make sure this model is compatible with your VLLM server setup

def run_vllm_client_operations():
    """
    Demonstrates various operations using VLLMClient.
    Assumes a VLLM server is already running.
    """
    print("Starting VLLM client operations...")

    # Initialize the client
    # Adjust connection_timeout as needed, similar to tests/test_vllm_client_server.py
    client = VLLMClient(connection_timeout=320)
    print("VLLMClient initialized.")

    try:
        # 1. Initialize communicator
        print("\nAttempting to initialize communicator...")
        client.init_communicator()
        print("Communicator initialized successfully.")
        # Small delay to ensure server is ready after communication init
        time.sleep(2)

        # 2. Update model parameters
        print(f"\nLoading model '{MODEL_ID}' for parameter update...")
        try:
            # Ensure the model is loaded on CPU to avoid conflicts if server uses GPUs.
            # The actual parameters are sent, device of local model doesn't dictate server device.
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID) #, device_map="cpu")
            print(f"Model '{MODEL_ID}' loaded.")
            print("Attempting to update model parameters on the server...")
            client.update_model_params(model)
            print("Model parameters updated successfully on the server.")
        except Exception as e:
            print(f"Could not load model or update parameters: {e}")
            print("Skipping model parameter update.")

        # 3. Generate text
        print("\nAttempting to generate text...")
        prompts = ["Hello, AI! Tell me a fun fact.", "What is the capital of France?"]
        try:
            outputs = client.generate(prompts, max_tokens=50, temperature=0.7)
            print("Text generation successful.")
            for i, output_seq in enumerate(outputs):
                # Assuming outputs are token IDs, we're not decoding them here
                # For actual text, you'd need a tokenizer to decode
                print(f"Prompt {i+1}: '{prompts[i]}' -> Generated (token IDs): {output_seq[:10]}... (first 10 tokens)")
        except Exception as e:
            print(f"Text generation failed: {e}")


        # 4. Generate text with more parameters
        print("\nAttempting to generate text with more parameters...")
        prompts_adv = ["Write a short poem about a robot."]
        try:
            outputs_adv = client.generate(
                prompts_adv,
                n=2,  # Number of output sequences to return for each prompt
                repetition_penalty=1.1,
                temperature=0.8,
                max_tokens=60
            )
            print("Advanced text generation successful.")
            for i, prompt_text in enumerate(prompts_adv):
                print(f"Prompt: '{prompt_text}'")
                # Each prompt will have 'n' completions
                start_idx = i * 2
                end_idx = start_idx + 2
                for j, output_seq in enumerate(outputs_adv[start_idx:end_idx]):
                    print(f"  Completion {j+1} (token IDs): {output_seq[:10]}... (first 10 tokens)")
        except Exception as e:
            print(f"Advanced text generation failed: {e}")


        # 5. Reset prefix cache
        print("\nAttempting to reset prefix cache...")
        try:
            client.reset_prefix_cache()
            print("Prefix cache reset successfully.")
        except Exception as e:
            print(f"Failed to reset prefix cache: {e}")

    except ConnectionRefusedError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("CONNECTION REFUSED: Could not connect to the VLLM server.")
        print("Please ensure the VLLM server is running and accessible.")
        print("You can start it with a command like:")
        print(f"  trl vllm-serve --model {MODEL_ID}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 6. Close communicator
        print("\nAttempting to close communicator...")
        client.close_communicator()
        print("Communicator closed.")

    print("\nVLLM client operations completed.")

if __name__ == "__main__":
    run_vllm_client_operations() 