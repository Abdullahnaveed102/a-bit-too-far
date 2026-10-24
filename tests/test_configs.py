"""Check the reference grids and their interface to the released prompt code."""

import json
from pathlib import Path
import re
import unittest

from fi_campaign.prompting import native_chat_messages, protocol_metadata, render_task_prompt
from fi_campaign.scoring import SCORER_VERSION


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((CONFIGS / "controlled_protocol.json").read_text())
        cls.geometry = json.loads((CONFIGS / "fault_geometry.json").read_text())
        cls.protection = json.loads((CONFIGS / "protection.json").read_text())

    def test_model_revisions_and_prompt_protocols(self):
        item = {"task": "squad", "prompt": "Synthetic source prompt.",
                "question": "What color is the box?", "context": "The box is green."}
        for model in self.protocol["models"].values():
            with self.subTest(model=model["id"]):
                self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", model["revision"]))
                protocol_metadata(model)
                self.assertIn(item["context"], render_task_prompt(model, item)["prompt"])

    def test_granite_retains_fixed_system_message(self):
        model = self.protocol["models"]["granite33_8b"]
        messages = native_chat_messages(model, "Synthetic task.")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"],
                         "Knowledge Cutoff Date: April 2024.\n"
                         "You are Granite, developed by IBM. You are a helpful AI assistant.")

    def test_mistral_has_no_added_system_message(self):
        messages = native_chat_messages(self.protocol["models"]["mistral7b"], "Synthetic task.")
        self.assertEqual(messages, [{"role": "user", "content": "Synthetic task."}])

    def test_scorer_version(self):
        self.assertEqual(self.protocol["scoring"]["version"], SCORER_VERSION)

    def test_geometry_grid_counts(self):
        shared = 6 * self.geometry["questions_per_task"] * self.geometry["draws_per_question"]
        for key in ("protocol_controls", "equal_budget_geometry", "composition"):
            grid = self.geometry[key]
            self.assertEqual(grid["assigned_executions"], shared * grid["conditions_per_configuration"])
        grid = self.geometry["equal_budget_geometry"]
        self.assertEqual(grid["conditions_per_configuration"],
                         3 * len(grid["selected_bit_budgets"]) * len(grid["bits_per_site"]))
        for budget in grid["selected_bit_budgets"]:
            for multiplicity in grid["bits_per_site"]:
                self.assertEqual(budget % multiplicity, 0)

    def test_composition_counts(self):
        grid = self.geometry["composition"]
        arms = grid["arms"]
        self.assertEqual(arms["A"]["selected_bits"] + arms["B"]["selected_bits"],
                         arms["AB"]["selected_bits"])
        self.assertEqual(4 * grid["assigned_four_arm_blocks"], grid["assigned_executions"])

    def test_protection_grid_and_reuse(self):
        grid = self.protection["protection_intensity"]
        plans = (6 * grid["questions_per_task"] * grid["draws_per_question"]
                 * len(grid["fault_operations"]) * len(grid["selected_bit_budgets"])
                 * len(grid["bits_per_site"]))
        self.assertEqual(plans, grid["fault_plans"])
        self.assertEqual(plans * len(grid["policies"]), grid["assigned_policy_executions"])
        self.assertEqual(grid["new_executions"] + grid["reused_executions"],
                         grid["assigned_policy_executions"])

    def test_runtime_cohort_and_repetitions(self):
        grid = self.protection["healthy_runtime"]
        self.assertFalse(grid["independent_healthy_holdout"])
        self.assertEqual(6 * grid["questions_per_task"] * len(grid["arms"]) * grid["repetitions"],
                         grid["assigned_timed_requests"])


if __name__ == "__main__":
    unittest.main()
