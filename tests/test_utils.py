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

import textwrap
import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import ModelConfig
from trl.trainer import compute_accuracy
from trl.trainer.utils import (
    DataCollatorForChatML,
    batch_generation,
    decode_and_strip_padding,
    flush_left,
    generate_model_card,
    get_peft_config,
    mask_tool_response_tokens,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from .testing_utils import require_rich


if is_peft_available():
    from peft import LoraConfig


class TestPad(unittest.TestCase):
    def test_pad_1_dim_left(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_1_dim_right(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_left(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[0, 0], [5, 6]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right_multidim(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 0], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_1(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="right", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_2(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([6, 7, 8])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="right", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_side_left(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([6, 7, 8])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="left", pad_to_multiple_of=4)
        expected = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 6, 7, 8]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_to_multiple_of_no_extra_padding(self):
        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([5, 6, 7, 8])
        # Already multiple of 4
        output = pad((x, y), padding_value=0, padding_side="left", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertTrue(torch.equal(output, expected))


@require_peft
class TestGetPEFTConfig(unittest.TestCase):
    def test_create_peft_config_use_peft_false(self):
        """Test that when use_peft is False, the function returns None."""
        model_args = ModelConfig(use_peft=False)
        peft_config = get_peft_config(model_args)
        self.assertIsNone(peft_config)

    def test_create_peft_config_use_peft_true(self):
        """Test that when use_peft is True, the function returns a LoraConfig object."""
        # Provide non-default values to the model config for testing
        peft_kwargs = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_task_type": "SEQ_CLS",
            "use_rslora": True,
            "lora_target_modules": ["up_proj", "down_proj"],
            "lora_modules_to_save": ["up_proj"],
        }
        model_args = ModelConfig(use_peft=True, **peft_kwargs)
        peft_config = get_peft_config(model_args)
        self.assertTrue(isinstance(peft_config, LoraConfig))
        for arg, value in peft_kwargs.items():
            # Test that lists of modules are converted to sets
            if arg == "lora_target_modules":
                value = set(value)
            # Rename the argument to match the LoraConfig attribute name
            if arg in ["lora_r", "lora_task_type", "lora_target_modules", "lora_modules_to_save"]:
                arg = arg[len("lora_") :] if arg.startswith("lora_") else arg

            self.assertEqual(getattr(peft_config, arg), value)


class TestDecodeAndStripPadding(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

    def test_example_with_padding(self):
        inputs = self.tokenizer(["Hello world", "Hello"], padding=True, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello world", "Hello"])

    def test_example_without_padding(self):
        inputs = self.tokenizer(["Hello", "Hello"], padding=False, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello", "Hello"])


class TestGenerateModelCard(unittest.TestCase):
    def test_full(self):
        model_card = generate_model_card(
            base_model="username/my_base_model",
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name="username/my_dataset",
            tags=["trl", "trainer-tag"],
            wandb_url="https://wandb.ai/username/project_id/runs/abcd1234",
            comet_url="https://www.comet.com/username/project_id/experiment_id",
            trainer_name="My Trainer",
            trainer_citation="@article{my_trainer, ...}",
            paper_title="My Paper",
            paper_id="1234.56789",
        )
        card_text = str(model_card)
        self.assertIn("[username/my_base_model](https://huggingface.co/username/my_base_model)", card_text)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("datasets: username/my_dataset", card_text)
        self.assertIn("](https://wandb.ai/username/project_id/runs/abcd1234)", card_text)
        self.assertIn("](https://www.comet.com/username/project_id/experiment_id", card_text)
        self.assertIn("My Trainer", card_text)
        self.assertIn("```bibtex\n@article{my_trainer, ...}\n```", card_text)
        self.assertIn("[My Paper](https://huggingface.co/papers/1234.56789)", card_text)

    def test_val_none(self):
        model_card = generate_model_card(
            base_model=None,
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name=None,
            tags=[],
            wandb_url=None,
            comet_url=None,
            trainer_name="My Trainer",
            trainer_citation=None,
            paper_title=None,
            paper_id=None,
        )
        card_text = str(model_card)
        self.assertIn("my_model", card_text)
        self.assertIn('pipeline("text-generation", model="username/my_hub_model", device="cuda")', card_text)
        self.assertIn("My Trainer", card_text)


class TestDataCollatorForChatML(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        # Token ID for "true", the last assistant's response in the example:
        self.ignore_index = -100
        self.max_length = 1024
        self.messages_key = "messages"

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset.to_list()

        # Initialize the data collator
        self.collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

    def test_data_collator_for_chatml(self):
        # Process the data
        data = self.collator(self.examples)

        # Verify basic shapes and types
        self.assertIn("input_ids", data)
        self.assertIn("attention_mask", data)
        self.assertIn("labels", data)
        self.assertIn("prompts", data)
        self.assertIn("prompt_attention_mask", data)

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Get the last assistant's response for comparison
        last_message = self.examples[0][self.messages_key][-1]
        self.assertEqual(last_message["role"], "assistant", "Last message should be from assistant")
        last_assistant_response = last_message["content"]

        # Verify that input_ids contain both prompt and response
        decoded_input = self.tokenizer.decode(input_ids)
        self.assertIn(last_assistant_response, decoded_input, "Input should contain assistant's response")

        # Verify that prompts only contain the conversation up to the last response
        decoded_prompt = self.tokenizer.decode(prompt_only)
        self.assertNotIn(last_assistant_response, decoded_prompt, "Prompt should not contain assistant's response")

        # Verify labels are -100 for non-assistant parts
        prompt_length = len(prompt_only)
        self.assertTrue(
            all(label == self.ignore_index for label in labels[:prompt_length]),
            "Labels should be ignore_index for prompt tokens",
        )

        # Verify labels match assistant response after prompt
        # Add a filter to remove any trailing tokens after the first <|im_end|>
        last_assistant_response_with_end = last_assistant_response + self.tokenizer.eos_token
        last_assistant_response_tokens = self.tokenizer.encode(
            last_assistant_response_with_end, add_special_tokens=False
        )

        response_labels = []
        for label in labels[prompt_length:]:
            if label == self.ignore_index:
                continue
            response_labels.append(label)
            if label == self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
        self.assertEqual(
            response_labels,
            last_assistant_response_tokens,
            "Labels should match assistant response tokens",
        )

        # Verify there isn't a generation prompt at the end
        generation_prompt = "<|im_start|>assistant"
        self.assertFalse(
            decoded_input.strip().endswith(generation_prompt),
            f"Input should not end with generation prompt '{generation_prompt}'",
        )

        self.assertEqual(
            response_labels,
            last_assistant_response_tokens,
            "Labels should match assistant response tokens",
        )


class TestBatchGeneration(unittest.TestCase):
    def setUp(self):
        # Initialize the tokenizer
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.5,
            do_sample=True,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset["messages"]
        self.mini_batch_size = 3

    def test_mini_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"]
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, self.mini_batch_size, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))

    def test_single_batch_generation(self):
        batch = [
            self.tokenizer.apply_chat_template(example[:-1], add_generation_prompt=True, tokenize=False)
            for example in self.examples
        ]
        queries = self.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"]
        bs, context_length = queries.shape

        query_responses, logits = batch_generation(
            self.model, queries, bs, self.tokenizer.pad_token_id, self.generation_config
        )

        max_length_query = query_responses.shape[1]
        max_length_logits = max_length_query - context_length

        self.assertGreater(max_length_query, context_length)
        self.assertEqual(query_responses.shape, (bs, max_length_query))
        self.assertEqual(logits.shape, (bs, max_length_logits, self.model.config.vocab_size))


class TestComputeAccuracy(unittest.TestCase):
    def test_token_classification_task(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[0, 1], [1, 0]]),
        )
        expected_accuracy = 0.5  # 2 matches, 2 mismatches
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_token_classification_task_with_ignored_tokens_0(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[1, 0], [1, -100]]),
        )
        expected_accuracy = 1.0  # All non-ignored tokens match
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_token_classification_task_with_ignored_tokens_1(self):
        eval_pred = (
            np.array(
                [
                    [[0.1, 0.9], [0.8, 0.2]],  # Batch 1
                    [[0.3, 0.7], [0.6, 0.4]],  # Batch 2
                ]
            ),
            np.array([[1, 1], [0, -100]]),
        )
        expected_accuracy = 1 / 3  # 1 match, 2 mismatch, 1 ignored
        result = compute_accuracy(eval_pred)
        self.assertAlmostEqual(result["accuracy"], expected_accuracy)

    def test_rewards_comparison_task(self):
        eval_pred = (
            np.array(
                [
                    [0.9, 0.1],  # Batch 1
                    [0.6, 0.4],  # Batch 2
                    [0.5, 0.5],  # Batch 3 (equal)
                ]
            ),
            np.array([0, 1, 1]),
        )
        expected_accuracy = 0.5  # 1 match, 1 mismatch, 1 equal (ignored)

        with self.assertWarns(UserWarning) as cm:
            result = compute_accuracy(eval_pred)

        self.assertAlmostEqual(result["accuracy"], expected_accuracy)
        expected_warning = (
            "There are 1 out of 3 instances where the predictions for both options are equal. "
            "These instances are ignored in the accuracy computation."
        )
        self.assertEqual(str(cm.warning), expected_warning)


class TestFlushLeft(unittest.TestCase):
    def test_basic_case(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        tensor1 = torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 0, 0]])
        tensor2 = torch.tensor([[0, 0, 7, 8, 9], [0, 10, 11, 0, 0]])
        new_mask, new_tensor1, new_tensor2 = flush_left(mask, tensor1, tensor2)

        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        expected_tensor1 = torch.tensor([[2, 3, 4], [5, 6, 0]])
        expected_tensor2 = torch.tensor([[7, 8, 9], [10, 11, 0]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))
        self.assertTrue(torch.equal(new_tensor2, expected_tensor2))

    def test_single_row(self):
        mask = torch.tensor([[0, 0, 1, 1]])
        tensor1 = torch.tensor([[0, 0, 2, 3]])
        new_mask, new_tensor1 = flush_left(mask, tensor1)

        expected_mask = torch.tensor([[1, 1]])
        expected_tensor1 = torch.tensor([[2, 3]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))

    def test_no_shift_needed(self):
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
        tensor1 = torch.tensor([[5, 6, 0, 0], [7, 8, 0, 0]])
        new_mask, new_tensor1 = flush_left(mask, tensor1)

        expected_mask = torch.tensor([[1, 1], [1, 1]])
        expected_tensor1 = torch.tensor([[5, 6], [7, 8]])

        self.assertTrue(torch.equal(new_mask, expected_mask))
        self.assertTrue(torch.equal(new_tensor1, expected_tensor1))

    def test_no_tensors(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        new_mask = flush_left(mask)

        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        self.assertTrue(torch.equal(new_mask, expected_mask))


class TestSelectiveLogSoftmax(unittest.TestCase):
    @parameterized.expand([(torch.float64,), (torch.float32,), (torch.float16,), (torch.bfloat16,)])
    def test_selective_log_softmax(self, dtype):
        """Test selective_log_softmax with logits of different dtypes"""
        vocab_size = 1024
        batch_size = 4
        seq_len = 32

        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=dtype)

        expected_output = torch.gather(logits.log_softmax(-1), dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        actual_output = selective_log_softmax(logits, input_ids)

        if dtype in [torch.float16, torch.bfloat16]:
            # half-precision dtypes fall back to an exact method
            self.assertTrue(torch.equal(actual_output, expected_output))
        else:
            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)


@require_rich
class TestPrintPromptCompletionsSample(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_output(self, mock_stdout):
        prompts = ["The sky is", "The sun is"]
        completions = [" blue.", " in the sky."]
        rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
        step = 42

        print_prompt_completions_sample(prompts, completions, rewards, step)

        output = mock_stdout.getvalue()

        expected_output = textwrap.dedent("""\
        ╭────────────────────── Step 42 ───────────────────────╮
        │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓ │
        │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ │
        │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩ │
        │ │ The sky is │  blue.       │        0.12 │   0.79 │ │
        │ ├────────────┼──────────────┼─────────────┼────────┤ │
        │ │ The sun is │  in the sky. │        0.46 │   0.10 │ │
        │ └────────────┴──────────────┴─────────────┴────────┘ │
        ╰──────────────────────────────────────────────────────╯
        """)
        self.assertEqual(output, expected_output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_num_samples(self, mock_stdout):
        prompts = ["A", "B"]
        completions = ["1", "2"]
        rewards = {"Score": [0.1, 0.2]}
        step = 10

        print_prompt_completions_sample(prompts, completions, rewards, step, num_samples=1)
        output = mock_stdout.getvalue()

        possible_outputs = [
            textwrap.dedent("""\
                ╭──────────── Step 10 ────────────╮
                │ ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓ │
                │ ┃ Prompt ┃ Completion ┃ Score ┃ │
                │ ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩ │
                │ │ A      │ 1          │  0.10 │ │
                │ └────────┴────────────┴───────┘ │
                ╰─────────────────────────────────╯
                """),
            textwrap.dedent("""\
                ╭──────────── Step 10 ────────────╮
                │ ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓ │
                │ ┃ Prompt ┃ Completion ┃ Score ┃ │
                │ ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩ │
                │ │ B      │ 2          │  0.20 │ │
                │ └────────┴────────────┴───────┘ │
                ╰─────────────────────────────────╯
                """),
        ]
        self.assertIn(output, possible_outputs)


class TestMaskToolResponseTokens(unittest.TestCase):
    """Test the mask_tool_response_tokens utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
    def _create_test_data(self, tokenizer, text):
        """Helper to create test data for a given tokenizer and text."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        completion_ids = torch.tensor([tokens], device=self.device)
        completion_mask = torch.ones_like(completion_ids)
        return completion_ids, completion_mask, tokens

    @parameterized.expand([
        ("Qwen/Qwen2.5-0.5B-Instruct",),
        ("Qwen/Qwen3-0.6B",),
        ("meta-llama/Llama-3.1-8B",),
    ])
    def test_tool_response_masking_concrete(self, model_name):
        """Test concrete masking behavior with specific token verification."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Test text with clear tool response boundaries
        test_text = "I will help you<tool_response>file1.txt\nfile2.txt</tool_response>Done"
        completion_ids, completion_mask, tokens = self._create_test_data(tokenizer, test_text)
        
        # Apply masking
        result_mask = mask_tool_response_tokens(completion_ids, completion_mask, tokenizer)
        
        # Apply masking and verify results
        original_sum = completion_mask.sum().item()
        masked_sum = result_mask.sum().item()
        num_masked = original_sum - masked_sum
        
        # Should have masked some tokens since we have <tool_response> content
        self.assertGreater(num_masked, 0,
            f"Should have masked some tokens for {model_name}")
        
        # Should have masked at least a few tokens (content + tags)
        self.assertGreaterEqual(num_masked, 3, 
            f"Expected at least 3 masked tokens for {model_name}, got {num_masked}")
        
        # Verify which tokens were masked
        mask_diff = completion_mask - result_mask
        masked_positions = torch.nonzero(mask_diff[0]).flatten()
        
        # Check that masked tokens are related to tool response
        masked_content = [tokenizer.decode([tokens[pos.item()]]) for pos in masked_positions[:5]]
        tool_related = any("tool" in content.lower() or "response" in content.lower() 
                          or content in ["<", ">", "</"] for content in masked_content)
        self.assertTrue(tool_related, 
            f"Masked tokens should include tool-related content, got: {masked_content}")

    @parameterized.expand([
        ("Qwen/Qwen2.5-0.5B-Instruct",),
        ("Qwen/Qwen3-0.6B",),
        ("meta-llama/Llama-3.1-8B",),
    ])
    def test_conversation_tool_masking(self, model_name):
        """Test masking in conversation format with tool roles."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Create conversation with tool role
        messages = [
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": "I will list files"},
            {"role": "tool", "content": "file1.txt\nfile2.txt\nfile3.txt"}
        ]
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            completion_ids, completion_mask, tokens = self._create_test_data(tokenizer, text)
            
            # Apply masking
            result_mask = mask_tool_response_tokens(completion_ids, completion_mask, tokenizer)
            
            # Should mask tool content
            self.assertLessEqual(result_mask.sum().item(), completion_mask.sum().item())
            
            # Verify we can identify which tokens were masked
            mask_diff = completion_mask - result_mask
            masked_positions = torch.nonzero(mask_diff[0]).flatten()
            
            if len(masked_positions) > 0:
                # Verify that masked tokens are in tool response area
                for pos in masked_positions[:3]:  # Check first few masked positions
                    token_text = tokenizer.decode([tokens[pos.item()]])
                    # Masked tokens should be part of tool content or tags
                    self.assertTrue(
                        any(pattern in text for pattern in ["tool", "file", "txt"]),
                        f"Token '{token_text}' at position {pos} should be part of tool response"
                    )
        except Exception:
            # If chat template fails, skip this test for this model
            self.skipTest(f"Chat template not supported for {model_name}")

    @parameterized.expand([
        ("Qwen/Qwen2.5-0.5B-Instruct",),
        ("Qwen/Qwen3-0.6B",),
        ("meta-llama/Llama-3.1-8B",),
    ])
    def test_no_tool_response_unchanged(self, model_name):
        """Test that text without tool responses remains unchanged."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        test_text = "This is regular text without any tool responses or special patterns."
        completion_ids, completion_mask, _ = self._create_test_data(tokenizer, test_text)
        
        # Apply masking
        result_mask = mask_tool_response_tokens(completion_ids, completion_mask, tokenizer)
        
        # Verify mask is unchanged
        self.assertTrue(torch.equal(result_mask, completion_mask),
            f"Mask should be unchanged for text without tool responses ({model_name})")

    @parameterized.expand([
        ("Qwen/Qwen2.5-0.5B-Instruct",),
        ("Qwen/Qwen3-0.6B",),
        ("meta-llama/Llama-3.1-8B",),
    ])
    def test_batch_processing(self, model_name):
        """Test masking works correctly with batched inputs."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Create batch with mixed content
        text1 = "Regular text without tool responses"
        text2 = "Text with <tool_response>content</tool_response> inside"
        
        tokens1 = tokenizer.encode(text1, add_special_tokens=False)
        tokens2 = tokenizer.encode(text2, add_special_tokens=False)
        
        # Pad to same length
        max_len = max(len(tokens1), len(tokens2))
        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokens1.extend([pad_token] * (max_len - len(tokens1)))
        tokens2.extend([pad_token] * (max_len - len(tokens2)))
        
        completion_ids = torch.tensor([tokens1, tokens2], device=self.device)
        completion_mask = torch.ones_like(completion_ids)
        
        # Apply masking
        result_mask = mask_tool_response_tokens(completion_ids, completion_mask, tokenizer)
        
        # First sequence should be unchanged, second should have masked tokens (if patterns detected)
        self.assertTrue(torch.equal(result_mask[0], completion_mask[0]),
            f"First sequence without tool responses should be unchanged ({model_name})")
        self.assertLessEqual(result_mask[1].sum().item(), completion_mask[1].sum().item(),
            f"Second sequence should have equal or fewer tokens ({model_name})")

    def test_empty_input(self):
        """Test function handles empty inputs gracefully."""
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        
        completion_ids = torch.empty((0, 0), dtype=torch.long, device=self.device)
        completion_mask = torch.empty((0, 0), dtype=torch.long, device=self.device)
        
        # Apply masking
        result_mask = mask_tool_response_tokens(completion_ids, completion_mask, tokenizer)
        
        # Should return original mask unchanged
        self.assertTrue(torch.equal(result_mask, completion_mask))

    @parameterized.expand([
        ("Qwen/Qwen2.5-0.5B-Instruct",),
        ("Qwen/Qwen3-0.6B",),
        ("meta-llama/Llama-3.1-8B",),
    ])
    def test_pattern_detection_robustness(self, model_name):
        """Test that pattern detection handles edge cases correctly."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Text with partial patterns that shouldn't match
        test_text = "This has <tool but no closing tag and <tool_response without closing"
        completion_ids, completion_mask, _ = self._create_test_data(tokenizer, test_text)
        
        # Apply masking
        result_mask = mask_tool_response_tokens(completion_ids, completion_mask, tokenizer)
        
        # Should not mask anything since patterns are incomplete
        self.assertTrue(torch.equal(result_mask, completion_mask),
            f"Incomplete patterns should not trigger masking ({model_name})")
