"""Prompt-contract checks using synthetic questions and a mock tokenizer."""
import hashlib
import unittest

from fi_campaign.prompting import INSTRUCT_PROTOCOL, LLAMA2_BASE_PROTOCOL, freeze_final_prompt, render_task_prompt


class MockTokenizer:
    chat_template = "synthetic test template; not a model chat template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["role"] + ": " + message["content"] for message in messages) + "\nassistant:"


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.math = {"task": "gsm8k", "prompt": "synthetic source", "question": "What is 2 + 3?"}
        self.qa = {"task": "squad", "prompt": "synthetic source", "context": "The box is green.", "question": "What color is the box?"}
        self.base = {"prompt_protocol": LLAMA2_BASE_PROTOCOL, "prompt_format": "plain"}
        self.chat = {"prompt_protocol": INSTRUCT_PROTOCOL, "prompt_format": "chat",
                     "chat_system_prompt": "Synthetic deterministic system prompt.",
                     "chat_template_sha256": hashlib.sha256(MockTokenizer.chat_template.encode()).hexdigest()}

    def test_base_math_has_synthetic_demonstrations(self):
        output = freeze_final_prompt(self.base, self.math)
        self.assertIn("Nora has 12 apples", output["prompt"])
        self.assertIn("What is 2 + 3?", output["prompt"])
        self.assertTrue(output["prompt"].endswith("Reasoning:"))
        self.assertTrue(output["prompt_frozen"])
        self.assertEqual(output["rendered_prompt_sha256"], hashlib.sha256(output["prompt"].encode()).hexdigest())

    def test_base_qa_includes_abstention_demonstration(self):
        output = render_task_prompt(self.base, self.qa)["prompt"]
        self.assertIn("Final answer: UNANSWERABLE", output)
        self.assertIn("The box is green.", output)

    def test_chat_freezing_uses_explicit_system_message(self):
        output = freeze_final_prompt(self.chat, self.qa, MockTokenizer())
        self.assertTrue(output["prompt"].startswith("system: Synthetic deterministic system prompt."))
        self.assertIn("Do not explain.", output["prompt"])
        self.assertEqual(output["prompt_rendering"], "frozen_native_chat_template_v1")

    def test_changed_chat_template_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            freeze_final_prompt({**self.chat, "chat_template_sha256": "0" * 64}, self.math, MockTokenizer())

    def test_chat_requires_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            freeze_final_prompt(self.chat, self.math)

    def test_protocol_format_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            render_task_prompt({**self.base, "prompt_format": "chat"}, self.math)


if __name__ == "__main__":
    unittest.main()
