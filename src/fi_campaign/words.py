"""Representation-level helpers. Bit 0 is the least significant native bit.

No torch import is needed here: these functions are independently testable on CPU.
"""
from __future__ import annotations

import hashlib
import random


FIELDS = {
    "float16": {"sign": (15,), "exponent": tuple(range(10, 15)), "fraction": tuple(range(10))},
    "bfloat16": {"sign": (15,), "exponent": tuple(range(7, 15)), "fraction": tuple(range(7))},
}


def native_bits(dtype: str, field: str = "all") -> tuple[int, ...]:
    if dtype not in FIELDS:
        raise ValueError("Only native float16 and bfloat16 are supported")
    if field == "all":
        return tuple(range(16))
    try:
        return FIELDS[dtype][field]
    except KeyError as exc:
        raise ValueError(f"Unsupported bit field: {field}") from exc


def stable_seed(seed: int, *parts: object) -> int:
    value = "\x1f".join([str(seed), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def choose_bits(dtype: str, field: str, count: int, seed: int) -> tuple[int, ...]:
    candidates = native_bits(dtype, field)
    if count < 1 or count > len(candidates):
        raise ValueError(f"Cannot select {count} distinct bits from {dtype}/{field}")
    return tuple(sorted(random.Random(seed).sample(candidates, count)))


def mutate_word(word: int, bits: tuple[int, ...] | list[int], mode: str) -> tuple[int, int]:
    if not 0 <= word <= 65535 or len(set(bits)) != len(bits) or any(b < 0 or b > 15 for b in bits):
        raise ValueError("Expected one unsigned 16-bit word and unique native bit indices")
    mask = sum(1 << b for b in bits)
    if mode == "flip":
        result = word ^ mask
    elif mode == "stuck_at_0":
        result = word & (~mask & 0xFFFF)
    elif mode == "stuck_at_1":
        result = word | mask
    else:
        raise ValueError(f"Unsupported fault mode: {mode}")
    return result, (word ^ result).bit_count()


def component_bits(bits: tuple[int, ...], site: int, bits_per_site: int, budget: int,
                   component: str) -> tuple[int, ...]:
    """Partition the *same* full sampled plan at bit granularity for A/B/AB."""
    if component == "none":
        return ()
    if component == "AB":
        return bits
    if component not in ("A", "B"):
        raise ValueError(f"Unknown component: {component}")
    cut = (budget + 1) // 2
    return tuple(bit for offset, bit in enumerate(bits)
                 if ((site * bits_per_site + offset < cut) == (component == "A")))


def exposure_budget(condition: dict) -> dict[str, int]:
    """Return assigned exposure, never confuse a paired parent with a mutation.

    Parent geometry remains available for sampling the same A/B/AB plan. The
    effective component budget is zero for controls and fault-disabled retries,
    even when their parent budget is nonzero. Delivery/change counts are runtime
    measurements and deliberately do not appear in this planning-only helper.
    """
    parent = int(condition.get("parent_combined_budget_bits", condition.get("budget", 0)))
    if parent < 0:
        raise ValueError("Parent combined budget must be nonnegative")
    component = condition.get("component", "AB")
    if component not in ("none", "A", "B", "AB"):
        raise ValueError(f"Unknown component: {component}")
    enabled = condition.get("arm", "fault") == "fault" and condition.get("fault_enabled", True)
    assigned = (0 if not enabled or component == "none" else
                (parent + 1) // 2 if component == "A" else parent // 2 if component == "B" else parent)
    return {"parent_combined_budget_bits": parent,
            "component_assigned_budget_bits": assigned,
            "planned_selected_bits": assigned}
