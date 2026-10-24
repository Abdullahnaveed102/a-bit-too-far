"""Representation-level checks; no tensor library or GPU is needed."""
import unittest

from fi_campaign.words import choose_bits, component_bits, exposure_budget, mutate_word, native_bits, stable_seed


class WordTests(unittest.TestCase):
    def test_flip_changes_every_selected_bit(self):
        word, changed = mutate_word(0x3C00, [0, 10, 15], "flip")
        self.assertEqual(word, 0x3C00 ^ ((1 << 0) | (1 << 10) | (1 << 15)))
        self.assertEqual(changed, 3)

    def test_forcing_selected_bits_need_not_change_them(self):
        self.assertEqual(mutate_word(0, [0, 1, 2], "stuck_at_0"), (0, 0))
        self.assertEqual(mutate_word(7, [0, 1, 2], "stuck_at_1"), (7, 0))
        self.assertEqual(mutate_word(0, [0, 1, 2], "stuck_at_1"), (7, 3))

    def test_all_bits(self):
        self.assertEqual(mutate_word(0x3C00, list(range(16)), "stuck_at_0")[0], 0)
        self.assertEqual(mutate_word(0x3C00, list(range(16)), "stuck_at_1")[0], 0xFFFF)

    def test_bit_field_layout(self):
        self.assertEqual(native_bits("float16", "sign"), (15,))
        self.assertEqual(native_bits("float16", "exponent"), tuple(range(10, 15)))
        self.assertEqual(native_bits("bfloat16", "exponent"), tuple(range(7, 15)))

    def test_sampling_is_repeatable_without_replacement(self):
        bits = choose_bits("float16", "all", 8, stable_seed(7, "example"))
        self.assertEqual(bits, choose_bits("float16", "all", 8, stable_seed(7, "example")))
        self.assertEqual(len(set(bits)), 8)

    def test_invalid_geometry(self):
        for bits in [[0, 0], [-1], [16]]:
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                mutate_word(0, bits, "flip")
        with self.assertRaises(ValueError):
            choose_bits("float16", "exponent", 6, 0)

    def test_components_form_disjoint_union(self):
        bits = (1, 2, 3, 4)
        for site in range(4):
            a = set(component_bits(bits, site, 4, 16, "A"))
            b = set(component_bits(bits, site, 4, 16, "B"))
            self.assertFalse(a & b)
            self.assertEqual(a | b, set(bits))

    def test_assigned_budget_distinguishes_controls(self):
        base = {"budget": 16, "component": "A", "arm": "fault"}
        self.assertEqual(exposure_budget(base)["planned_selected_bits"], 8)
        self.assertEqual(exposure_budget({**base, "component": "AB"})["planned_selected_bits"], 16)
        self.assertEqual(exposure_budget({**base, "fault_enabled": False})["planned_selected_bits"], 0)
        self.assertEqual(exposure_budget({**base, "arm": "identity"})["planned_selected_bits"], 0)


if __name__ == "__main__":
    unittest.main()
