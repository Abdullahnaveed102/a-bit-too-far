"""Scoped activation hooks and native-word interventions.

Hooks are removed even after model exceptions. Sampling is independent of arm,
protection policy, and A/B component; exact coordinates are written to events.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path

from .words import choose_bits, component_bits, exposure_budget, mutate_word, stable_seed


LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
PROJECTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def layer_index(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def module_catalog(model) -> list[tuple[str, object, int, str]]:
    # Projection names are an explicit supported-model contract, not every Linear
    # in a model (which would inadvertently include the output vocabulary head).
    rows = []
    for name, module in model.named_modules():
        index = layer_index(name)
        suffix = name.rsplit(".", 1)[-1]
        if index is not None and suffix in PROJECTION_SUFFIXES:
            family = "attention" if suffix in PROJECTION_SUFFIXES[:4] else "mlp"
            rows.append((name, module, index, family))
    if not rows:
        raise ValueError("Unsupported decoder layout: no recognized decoder projection modules")
    return sorted(rows, key=lambda row: (row[2], row[0]))


def filter_catalog(catalog, depth="any", family="any"):
    layers = sorted({row[2] for row in catalog})
    if depth not in ("any", "early", "middle", "late"):
        raise ValueError(f"Unsupported layer stratum: {depth}")
    if family not in ("any", "attention", "mlp"):
        raise ValueError(f"Unsupported module stratum: {family}")
    selected_layers = set(layers)
    if depth != "any":
        thirds = {"early": 0, "middle": 1, "late": 2}
        selected_layers = {layer for rank, layer in enumerate(layers)
                           if min(2, 3 * rank // len(layers)) == thirds[depth]}
    selected = [row for row in catalog if row[2] in selected_layers and (family == "any" or row[3] == family)]
    if not selected:
        raise ValueError("The requested placement stratum contains no supported modules")
    return selected


def load_profile(path: str | None) -> tuple[dict, list[str]]:
    if not path:
        return {}, []
    with Path(path).open(encoding="utf-8") as handle:
        obj = json.load(handle)
    bounds = obj.get("bounds", obj)
    if not isinstance(bounds, dict):
        raise ValueError("Protection profile bounds must be a module-name mapping")
    result = {}
    for name, values in bounds.items():
        if name == "ranking" or not isinstance(values, (list, tuple)):
            continue
        if len(values) != 2:
            raise ValueError(f"Invalid profile bounds for {name}")
        lo, hi = map(float, values)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError(f"Profile bounds must be finite and increasing: {name}")
        result[name] = (lo, hi)
    ranking = obj.get("ranking", []) if "bounds" in obj else []
    return result, list(ranking)


class ActivationHooks:
    def __init__(self, model, plan: dict, *, trace_only: bool = False):
        self.model, self.plan = model, plan
        self.condition = plan.get("condition", {})
        self.trace_only = trace_only
        self.arm = "monitor" if trace_only else self.condition.get("arm", "fault")
        self.fault_enabled = (not trace_only and self.arm == "fault"
                              and bool(self.condition.get("fault_enabled", True))
                              and self.condition.get("component", "AB") != "none")
        self.catalog = module_catalog(model)
        self.events, self.traces, self.handles = [], [], []
        # Cumulative count for this context.  ``handles`` is cleared on exit,
        # so its final length cannot prove that instrumentation was installed.
        self.registered_hook_count = 0
        self.step = 0
        self.fired = set()
        self.selected_bits = self.changed_bits = self.interventions = 0
        self.identity_writes = 0
        self.detector_alarm = False
        self.detector_pre_repair_alarm = False
        self.detector_post_repair_alarm = False
        self.detect_phase = self.condition.get("detect_phase", "pre_repair")
        if self.detect_phase not in ("pre_repair", "post_repair", "both"):
            raise ValueError("detect_phase must be pre_repair, post_repair or both")
        self.assignments = self._assign()
        self.policy = {} if trace_only else self.condition.get("policy", {})
        self.profile_sha256 = None
        self.protected, self.bounds = self._protection()

    def _assign(self):
        c = self.condition
        if c.get("target", "activation") != "activation":
            return []
        sites, bps = int(c.get("sites", 1)), int(c.get("bits_per_site", 1))
        budget = int(c.get("parent_combined_budget_bits", c.get("budget", sites * bps)))
        if sites == 0 and budget == 0 and bps >= 0 and self.arm not in ("fault", "identity"):
            return []
        if sites < 1 or bps < 1 or budget != sites * bps:
            raise ValueError("Fault budget must equal positive sites × bits_per_site")
        seed = stable_seed(int(self.plan.get("seed", 0)), "activation")
        rng = random.Random(seed)
        pool = filter_catalog(self.catalog, c.get("depth", "any"), c.get("module", "any"))
        placement = c.get("placement", "within_tensor")
        if placement not in ("within_tensor", "cross_layer", "cross_step"):
            raise ValueError(f"Unsupported placement: {placement}")
        stage = c.get("stage", "prefill")
        if stage not in ("prefill", "decode"):
            raise ValueError("stage must be prefill or decode")
        first_step = 0 if stage == "prefill" else int(c.get("decode_step", 1))
        if stage == "decode" and first_step < 1:
            raise ValueError("decode_step is one-based")
        chosen = rng.choice(pool)
        if placement == "cross_layer":
            layers = sorted({row[2] for row in pool})
            if sites < 2 or len(layers) < 2:
                raise ValueError("cross_layer requires at least two sites and two eligible layers")
            rng.shuffle(layers)
            layer_rows = {layer: [row for row in pool if row[2] == layer] for layer in layers}
            modules = [rng.choice(layer_rows[layers[i % len(layers)]]) for i in range(sites)]
        else:
            modules = [chosen] * sites
        if placement == "cross_step" and stage != "decode":
            raise ValueError("cross_step must use decode stage")
        assignments = []
        for site, row in enumerate(modules):
            bits = choose_bits(self.plan["dtype"], c.get("bit_field", "all"), bps,
                               stable_seed(seed, "bits", site))
            assignments.append({"site": site, "name": row[0], "layer": row[2],
                                "step": first_step + site if placement == "cross_step" else first_step,
                                "bits": bits, "coordinate_seed": stable_seed(seed, "coordinate", site)})
        return assignments

    def _protection(self):
        kind = self.policy.get("kind", "none")
        allowed = ("none", "nonfinite", "finite", "combined", "selective")
        if kind not in allowed:
            raise ValueError(f"Unknown protection policy: {kind}")
        if kind == "none":
            return set(), {}
        bounds, ranking = load_profile(self.policy.get("profile_path"))
        if self.policy.get("profile_path"):
            self.profile_sha256 = hashlib.sha256(Path(self.policy["profile_path"]).read_bytes()).hexdigest()
        names = [row[0] for row in self.catalog]
        if kind in ("finite", "combined", "selective") and any(name not in bounds for name in names):
            missing = [name for name in names if name not in bounds]
            raise ValueError(f"Profile missing {len(missing)} projection modules; first={missing[0]}")
        if kind == "selective":
            coverage = float(self.policy.get("coverage", 1.0))
            if not 0 < coverage <= 1:
                raise ValueError("Selective coverage must be in (0,1]")
            count = max(1, math.ceil(len(names) * coverage))
            selection = self.policy.get("selection", "top")
            if selection == "random":
                names = random.Random(int(self.policy.get("mask_seed", 0))).sample(names, count)
            elif selection == "top":
                if not ranking or len(set(ranking)) != len(ranking) or set(ranking) != set(names):
                    raise ValueError("Top selection requires a complete, unique module ranking in the profile")
                names = ranking[:count]
            else:
                raise ValueError("Selective selection must be top or random")
        scale = float(self.policy.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Protection scale must be finite and positive")
        expanded = {}
        for name, (lo, hi) in bounds.items():
            center, radius = (lo + hi) / 2, (hi - lo) / 2
            expanded[name] = (center - scale * radius, center + scale * radius)
            if not all(math.isfinite(value) for value in expanded[name]):
                raise ValueError(f"Expanded protection bounds are not finite: {name}")
        return set(names), expanded

    def __enter__(self):
        if self.arm == "no_hook":
            return self
        if self.arm not in ("empty_hook", "identity", "fault", "monitor"):
            raise ValueError(f"Unknown instrumentation arm: {self.arm}")
        try:
            for name, module, _, _ in self.catalog:
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
                self.registered_hook_count += 1
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            if self.arm == "empty_hook":
                return None
            import torch
            if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1:
                raise ValueError(f"Unsupported projection output layout for {name}")
            result = output
            jobs = [a for a in self.assignments if a["name"] == name and a["step"] == self.step]
            if jobs and (self.fault_enabled or self.arm == "identity"):
                if output.dtype not in (torch.float16, torch.bfloat16):
                    raise ValueError("Refusing non-native FP16/BF16 mutation")
                expected = torch.float16 if self.plan["dtype"] == "float16" else torch.bfloat16
                if output.dtype != expected:
                    raise ValueError("Actual tensor dtype differs from planned representation")
                result = output.contiguous().clone()
                words = result.view(torch.int16).reshape(-1)
                # Assign all sites in one tensor jointly, without replacement.
                group = [a for a in self.assignments if a["name"] == name and a["step"] == self.step]
                if len(group) > words.numel():
                    raise ValueError("Requested distinct sites exceed the available tensor")
                indices = random.Random(stable_seed(int(self.plan.get("seed", 0)), name, self.step, "indices")).sample(
                    range(words.numel()), len(group))
                for assignment, index in zip(group, indices):
                    bits = component_bits(assignment["bits"], assignment["site"], int(self.condition.get("bits_per_site", 1)),
                                          int(self.condition.get("parent_combined_budget_bits", self.condition.get("budget", len(self.assignments)))),
                                          self.condition.get("component", "AB"))
                    if not bits:
                        continue
                    before = int(words[index].item()) & 0xFFFF
                    if self.arm == "identity":
                        after, changed = before, 0
                        self.identity_writes += 1
                    else:
                        after, changed = mutate_word(before, bits, self.condition.get("fault_mode", "flip"))
                        self.fired.add(assignment["site"])
                    words[index] = after if after < 32768 else after - 65536
                    self.selected_bits += len(bits) if self.arm == "fault" else 0
                    self.changed_bits += changed
                    coordinate = [0, index // result.shape[-1], index % result.shape[-1]]
                    self.events.append({"type": "mutation" if self.arm == "fault" else "identity_write", "target": "activation", "module": name,
                                        "layer": assignment["layer"], "forward_step": self.step, "site": assignment["site"],
                                        "coordinate": coordinate, "bits": list(bits), "before_word": before,
                                        "after_word": after, "selected_bits": len(bits) if self.arm == "fault" else 0, "changed_bits": changed})
            # Lean mode omits diagnostic counters, never the requested detector.
            detecting = self.condition.get("detect") == "nonfinite" or self.condition.get("detect_nonfinite")
            pre_repair = result
            pre_alarm = detecting and not bool(torch.isfinite(pre_repair).all().item())
            if pre_alarm:
                self.detector_pre_repair_alarm = True
                if self.detect_phase in ("pre_repair", "both"):
                    self.detector_alarm = True
            if name in self.protected and self.arm != "identity":
                result = self._repair(name, result)
            if self.condition.get("trace") and (self.step <= int(self.condition.get("trace_steps", 4))):
                self._trace(name, result)
            # Log both observation points; the declared phase controls triggering.
            # No second device synchronization is needed when no repair ran.
            post_alarm = (pre_alarm if result is pre_repair else
                          detecting and not bool(torch.isfinite(result).all().item()))
            if post_alarm:
                self.detector_post_repair_alarm = True
                if self.detect_phase in ("post_repair", "both"):
                    self.detector_alarm = True
            return result if result is not output else None
        return hook

    def _repair(self, name, tensor):
        import torch
        kind = self.policy.get("kind", "none")
        finite = torch.isfinite(tensor)
        modified = tensor
        native = torch.finfo(tensor.dtype)
        def representable(value):
            return min(native.max, max(native.min, value))
        if kind in ("nonfinite", "combined", "selective"):
            lo, hi = self.bounds.get(name, (torch.finfo(tensor.dtype).min, torch.finfo(tensor.dtype).max))
            lo, hi = representable(lo), representable(hi)
            # NaN -> zero; signed infinities -> finite bounds.
            modified = torch.nan_to_num(modified, nan=0.0, posinf=hi, neginf=lo)
        if kind in ("finite", "combined", "selective"):
            lo, hi = self.bounds[name]
            lo, hi = representable(lo), representable(hi)
            clipped = torch.clamp(modified, lo, hi)
            modified = torch.where(finite, clipped, modified) if kind == "finite" else clipped
        if not self.condition.get("lean", False):
            count = int((~((modified == tensor) | (torch.isnan(modified) & torch.isnan(tensor)))).sum().item())
            self.interventions += count
            if count:
                self.events.append({"type": "repair", "module": name, "forward_step": self.step, "modified_values": count})
        return modified

    def _trace(self, name, tensor):
        import torch
        # Sparse decoder boundaries only: down_proj and o_proj, not every projection.
        if not name.endswith(("down_proj", "o_proj")):
            return
        values = tensor.detach().float().reshape(-1)
        finite = torch.isfinite(values)
        safe = torch.where(finite, values, torch.zeros_like(values))
        indices = torch.linspace(0, safe.numel() - 1, min(16, safe.numel()), device=safe.device).long()
        self.traces.append({"module": name, "forward_step": self.step,
                            "finite_fraction": float(finite.float().mean().item()),
                            "rms_finite_zeroed": float(safe.double().square().mean().sqrt().item()),
                            "max_abs_finite_zeroed": float(safe.abs().max().item()),
                            "sample_flat_indices": indices.cpu().tolist(),
                            "sample": safe[indices].cpu().tolist()})

    def diagnostics(self):
        exposure = exposure_budget({**self.condition, "arm": self.arm,
                                    "fault_enabled": self.fault_enabled})
        assigned = []
        if self.fault_enabled and self.condition.get("target", "activation") == "activation":
            for assignment in self.assignments:
                bits = component_bits(assignment["bits"], assignment["site"],
                                      int(self.condition.get("bits_per_site", 1)),
                                      exposure["parent_combined_budget_bits"], self.condition.get("component", "AB"))
                if bits:
                    assigned.append({**assignment, "bits": bits})
        else:
            exposure.update({"component_assigned_budget_bits": 0, "planned_selected_bits": 0})
        expected = exposure["planned_selected_bits"]
        return {**exposure, "planned_sites": len(assigned), "fired_sites": len(self.fired),
                "registered_hook_count": self.registered_hook_count,
                "delivered_selected_bits": self.selected_bits,
                "actually_changed_bits": self.changed_bits, "delivery_complete": self.selected_bits == expected,
                "identity_site_writes": self.identity_writes,
                "protection_interventions": self.interventions,
                "protected_modules": sorted(self.protected), "protection_profile_sha256": self.profile_sha256,
                "protection_bound_scaling": "center +/- scale * half_range",
                "protection_native_bound_handling": "expanded bounds saturated to native finite limits before tensor operations",
                "intervention_count_collected": not self.condition.get("lean", False),
                "detector_alarm": self.detector_alarm, "trace": self.traces,
                "detector_pre_repair_alarm": self.detector_pre_repair_alarm,
                "detector_post_repair_alarm": self.detector_post_repair_alarm,
                "detector_trigger_phase": self.detect_phase,
                "detector_checks_collected": bool(self.condition.get("detect") == "nonfinite" or self.condition.get("detect_nonfinite")),
                "detector_observation": "pre/post-repair projections recorded; declared phase triggers, final logits always monitored when enabled",
                "fault_enabled": self.fault_enabled,
                "fault_assignments": assigned,
                "parent_fault_assignments": self.assignments,
                "parent_plan_sites": (len(self.assignments) if self.condition.get("target", "activation") == "activation"
                                      else int(self.condition.get("sites", 0))),
                "parent_bits_per_site": int(self.condition.get("bits_per_site", 0)),
                "coordinate_convention": "batch, token, channel; forward 0=prefill, 1=first decode"}
