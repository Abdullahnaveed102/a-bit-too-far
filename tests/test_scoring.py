"""Synthetic checks of the released answer-extraction contract."""
import unittest

from fi_campaign.scoring import score_output, task_answer_view


class ScoringTests(unittest.TestCase):
    def test_first_marked_line_wins(self):
        output = "Work: 6 * 7 = 42\nFinal answer: 42\nFinal answer: 99"
        self.assertEqual(task_answer_view(output)["task_answer_output"], "42")
        self.assertTrue(score_output(output, ["42"], "gsm8k")["exact"])

    def test_empty_marked_line_does_not_use_next_line(self):
        result = score_output("Final answer:\n42", ["42"], "gsm8k")
        self.assertFalse(result["valid"])
        self.assertIsNone(result["exact"])
        self.assertEqual(result["reason"], "empty_final_answer")

    def test_multiple_markers_on_one_line_are_unresolved(self):
        result = score_output("Final answer: 42 #### 42", ["42"], "gsm8k")
        self.assertFalse(result["valid"])

    def test_numeric_equivalence(self):
        for answer in ["1/2", "0.5", "5e-1"]:
            with self.subTest(answer=answer):
                self.assertTrue(score_output("Final answer: " + answer, ["0.5"], "gsm8k")["exact"])

    def test_percent_unit_is_not_silently_converted(self):
        result = score_output("Final answer: 50%", ["0.5"], "gsm8k")
        self.assertTrue(result["valid"])
        self.assertFalse(result["exact"])

    def test_terminal_equation_in_marked_answer(self):
        self.assertTrue(score_output("#### 6 * 7 = 42", ["42"], "gsm8k")["exact"])
        self.assertFalse(score_output("6 * 7 = 42", ["42"], "gsm8k")["valid"])

    def test_ambiguous_number_is_not_wrong_answer(self):
        result = score_output("Final answer: 4 or 5", ["4"], "gsm8k")
        self.assertFalse(result["valid"])
        self.assertIsNone(result["exact"])

    def test_valid_wrong_is_distinct(self):
        result = score_output("Final answer: 41", ["42"], "gsm8k")
        self.assertTrue(result["valid"])
        self.assertFalse(result["exact"])

    def test_unmarked_response_is_scored_whole(self):
        self.assertTrue(score_output("42", ["42"], "gsm8k")["exact"])
        self.assertFalse(score_output("I used 6 and obtained 42", ["42"], "gsm8k")["valid"])

    def test_qa_normalization(self):
        result = score_output("Final answer: The wooden drawer.", ["wooden drawer"], "squad")
        self.assertTrue(result["exact"])
        self.assertEqual(result["f1"], 1.0)

    def test_qa_explicit_abstention(self):
        self.assertTrue(score_output("Final answer: UNANSWERABLE", [""], "squad2")["exact"])
        self.assertFalse(score_output("Final answer: UNANSWERABLE", ["Paris"], "squad2")["exact"])
        self.assertFalse(score_output("", [""], "squad2")["valid"])

    def test_missing_gold_is_not_unanswerable(self):
        self.assertEqual(score_output("UNANSWERABLE", [], "squad")["reason"], "missing_gold_answers")

    def test_qa_lexical_f1(self):
        result = score_output("Final answer: wooden box", ["wooden drawer"], "squad")
        self.assertFalse(result["exact"])
        self.assertEqual(result["f1"], 0.5)

    def test_qa_length_limit(self):
        result = score_output("Final answer: " + "word " * 33, ["word"], "squad")
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
