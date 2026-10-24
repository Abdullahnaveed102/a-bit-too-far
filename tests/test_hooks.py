"""Hook planning and optional CPU tensor tests; no model downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from fi_campaign.hooks import ActivationHooks, module_catalog


def plan(**changes):
    condition = {"arm": "fault", "target": "activation", "stage": "prefill",
                 "placement": "within_tensor", "sites": 4, "bits_per_site": 4,
                 "budget": 16, "fault_mode": "flip", "component": "AB"}
    condition.update(changes)
    return {"seed": 42, "dtype": "float16", "condition": condition}


class Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class Module:
    def __init__(self):
        self.handles = []

    def register_forward_hook(self, hook):
        handle = Handle()
        self.handles.append(handle)
        return handle


class Model:
    def __init__(self):
        self.modules = [(f"model.layers.{i}.self_attn.q_proj", Module()) for i in range(3)]
        self.modules.append(("lm_head", Module()))

    def named_modules(self):
        return iter(self.modules)


class HookPlanTests(unittest.TestCase):
    def test_catalog_excludes_output_head(self):
        self.assertEqual(len(module_catalog(Model())), 3)

    def test_paired_components_use_same_parent_plan(self):
        a = ActivationHooks(Model(), plan(component="A"))
        b = ActivationHooks(Model(), plan(component="B"))
        ab = ActivationHooks(Model(), plan())
        self.assertEqual(a.assignments, b.assignments)
        self.assertEqual(a.assignments, ab.assignments)
        self.assertEqual(a.diagnostics()["planned_selected_bits"], 8)
        self.assertEqual(ab.diagnostics()["planned_selected_bits"], 16)

    def test_target_module_is_fixed_across_geometry(self):
        sparse = ActivationHooks(Model(), plan(sites=16, bits_per_site=1))
        concentrated = ActivationHooks(Model(), plan(sites=2, bits_per_site=8))
        self.assertEqual(sparse.assignments[0]["name"], concentrated.assignments[0]["name"])

    def test_unexecuted_fault_is_not_complete(self):
        hooks = ActivationHooks(Model(), plan())
        self.assertFalse(hooks.diagnostics()["delivery_complete"])
        self.assertEqual(hooks.diagnostics()["delivered_selected_bits"], 0)

    def test_handles_are_removed_on_exception(self):
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with ActivationHooks(model, plan()) as hooks:
                self.assertEqual(hooks.registered_hook_count, 3)
                raise RuntimeError("synthetic forward failure")
        self.assertEqual(hooks.handles, [])
        self.assertTrue(all(handle.removed for _, module in model.modules for handle in module.handles))

    def test_no_hook_control_registers_nothing(self):
        with ActivationHooks(Model(), plan(arm="no_hook")) as hooks:
            self.assertEqual(hooks.registered_hook_count, 0)
            self.assertEqual(hooks.diagnostics()["planned_selected_bits"], 0)

    def test_budget_must_match_geometry(self):
        with self.assertRaises(ValueError):
            ActivationHooks(Model(), plan(budget=32))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is not installed")
class TensorHookTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

        class TensorModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.q_proj = torch.nn.Identity()
                self.layers = torch.nn.ModuleList([layer])

            def forward(self, values):
                return self.layers[0].self_attn.q_proj(values)

        self.model = TensorModel()
        self.values = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]], dtype=torch.float16)

    def test_native_mutation_and_input_preservation(self):
        original = self.values.clone()
        with ActivationHooks(self.model, plan(sites=1, bits_per_site=16, fault_mode="stuck_at_1")) as hooks:
            result = self.model(self.values)
        self.assertEqual(int(self.torch.isnan(result).sum()), 1)
        self.assertTrue(self.torch.equal(self.values, original))
        self.assertTrue(hooks.diagnostics()["delivery_complete"])
        self.assertEqual(hooks.diagnostics()["delivered_selected_bits"], 16)
        self.assertEqual(len(hooks.events[0]["coordinate"]), 3)

    def test_identity_writes_preserve_tensor(self):
        with ActivationHooks(self.model, plan(arm="identity")) as hooks:
            result = self.model(self.values)
        self.assertTrue(self.torch.equal(result, self.values))
        self.assertEqual(hooks.diagnostics()["identity_site_writes"], 4)
        self.assertEqual(hooks.diagnostics()["actually_changed_bits"], 0)

    def test_nonfinite_repair_after_fault(self):
        with ActivationHooks(self.model, plan(sites=1, bits_per_site=16, fault_mode="stuck_at_1",
                                             policy={"kind": "nonfinite"})) as hooks:
            result = self.model(self.values)
        self.assertTrue(bool(self.torch.isfinite(result).all()))
        self.assertEqual(int((result == 0).sum()), 1)
        self.assertEqual(hooks.diagnostics()["protection_interventions"], 1)

    def test_finite_and_combined_repair_are_distinct(self):
        values = self.torch.tensor([[[float("nan"), float("inf"), -10., 10.]]], dtype=self.torch.float16)
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "synthetic_bounds.json"
            profile.write_text(json.dumps({"bounds": {"layers.0.self_attn.q_proj": [-2., 2.]}}))
            outputs = {}
            for kind in ("finite", "combined"):
                with ActivationHooks(self.model, plan(fault_enabled=False,
                        policy={"kind": kind, "profile_path": str(profile)})):
                    outputs[kind] = self.model(values)
        self.assertTrue(bool(self.torch.isnan(outputs["finite"][0, 0, 0])))
        self.assertTrue(bool(self.torch.isinf(outputs["finite"][0, 0, 1])))
        self.assertEqual(outputs["finite"][0, 0, 2:].tolist(), [-2., 2.])
        self.assertEqual(outputs["combined"].tolist(), [[[0., 2., -2., 2.]]])


if __name__ == "__main__":
    unittest.main()
