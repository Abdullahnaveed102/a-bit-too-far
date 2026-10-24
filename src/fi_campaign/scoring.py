"""Conservative, dependency-free answer extraction and task scoring.

Scoring is lexical/task-specific, not a semantic oracle. Ambiguous extraction is
reported as undefined rather than silently marked correct or incorrect.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import re
import string
from typing import Any

SCORER_VERSION = "final_answer_v1.4"
_MARKER = re.compile(r"(?:\bFinal\s+answer\s*:|####)", re.IGNORECASE)
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
    r"(?=$|[\s,;:!?)}\]\"']|\.(?:$|[\s)}\]\"']))"
)
_NO_ANSWER = {"", "unanswerable", "no answer", "no answer available", "not answerable"}
_NUMERIC_UNCERTAINTY = re.compile(
    r"\b(?:not|isn't|isnt|cannot|either|or|maybe)\b", re.I
)
_APPROXIMATE = re.compile(r"\bapproximately\b", re.I)
_CONTRADICTION = re.compile(
    r"\bbut\s+actually\b|\bactually\s+not\b|\bcorrection\s*:", re.I
)
_TERMINAL_NUMERIC_SUFFIX = re.compile(
    r"\s*(?:(?!or\b|but\b|maybe\b|approximately\b)[A-Za-z][A-Za-z-]*\s*){0,8}"
    r"[.)!?]*\s*$",
    re.I,
)


def normalize_qa(text: str) -> str:
    """SQuAD-style lowercase, punctuation/article removal and whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def explicit_answer_marker_count(text: str) -> int:
    """Count explicit answer markers in retained generated text."""
    return len(_MARKER.findall(text)) if isinstance(text, str) else 0


def _invalid(reason: str, prediction: str = "") -> dict[str, Any]:
    return {"prediction": prediction, "exact": None, "f1": None,
            "valid": False, "reason": reason}


def task_answer_view(text: str) -> dict[str, Any]:
    """Return the bounded answer-bearing text without changing raw generation.

    A model can emit a complete answer and then continue into another example,
    especially under the intentionally uncensored fixed-horizon protocol.  When
    an explicit marker exists, the immediate first marked *line* is therefore
    the task-answer view. If that line has no answer, extraction fails rather
    than searching later markers. Later marked lines are retained in the
    raw/scoring stream for repetition diagnostics but cannot overwrite it.

    Unmarked text remains whole: there is no general first/last-number shortcut.
    Boundary validity is separate from task-score validity; an extracted line can
    still be numerically ambiguous or too long for the task evaluator.
    """
    if not isinstance(text, str):
        return {"task_answer_output": "", "task_answer_view": "invalid_non_string_output",
                "task_answer_boundary_valid": False,
                "task_answer_boundary_reason": "non_string_output"}
    if not text.strip():
        return {"task_answer_output": "", "task_answer_view": "empty_generated_output",
                "task_answer_boundary_valid": False,
                "task_answer_boundary_reason": "empty_output"}
    marker = _MARKER.search(text)
    if marker:
        # splitlines()[0] enforces the declared answer-line boundary in natural
        # and fixed regimes alike. Do not use a later marker if this one is empty.
        remainder = text[marker.end():]
        lines = remainder.splitlines()
        answer = (lines[0] if lines else "").strip().strip("`*").strip()
        if not answer:
            return {"task_answer_output": "",
                    "task_answer_view": "first_explicit_final_answer_line",
                    "task_answer_boundary_valid": False,
                    "task_answer_boundary_reason": "empty_final_answer"}
        # A second marker on the same physical line does not define two complete
        # answer lines and remains ambiguous. Markers on later lines are ignored.
        if _MARKER.search(answer):
            return {"task_answer_output": answer,
                    "task_answer_view": "first_explicit_final_answer_line",
                    "task_answer_boundary_valid": False,
                    "task_answer_boundary_reason": "multiple_markers_on_answer_line"}
        return {"task_answer_output": answer,
                "task_answer_view": "first_explicit_final_answer_line",
                "task_answer_boundary_valid": True,
                "task_answer_boundary_reason": "explicit_first_final_answer_line"}
    return {"task_answer_output": text.strip().strip("`*").strip(),
            "task_answer_view": "full_unmarked_output",
            "task_answer_boundary_valid": True,
            "task_answer_boundary_reason": "unmarked_answer"}


def _extract(text: str) -> tuple[str, bool, str]:
    view = task_answer_view(text)
    return (view["task_answer_output"], view["task_answer_boundary_valid"],
            view["task_answer_boundary_reason"])


def _terminal_equation_result(text: str, matches: list[re.Match[str]]) -> re.Match[str] | None:
    """Return the RHS number for a narrowly formed terminal arithmetic equality.

    This deliberately does not implement a general "last number wins" rule. The
    caller has already established that the text follows an explicit final-answer
    marker; here we additionally require one equality, at least two numeric
    operands and an arithmetic operator on the left, and only a number plus an
    optional short unit phrase on the right.
    """
    if text.count("=") != 1:
        return None
    lhs, rhs = text.split("=", 1)
    lhs_matches = list(_NUMBER.finditer(lhs))
    rhs_matches = list(_NUMBER.finditer(rhs))
    if len(lhs_matches) < 2 or len(rhs_matches) != 1:
        return None
    if not re.search(r"[+*/\u00d7\u00f7]|(?<=\s)-(?=\s)", lhs):
        return None
    result = rhs_matches[0]
    before = rhs[:result.start()]
    after = rhs[result.end():]
    if not re.fullmatch(r"\s*[$\u00a3\u20ac\u00a5]?\s*[([]?\s*", before):
        return None
    if not _TERMINAL_NUMERIC_SUFFIX.fullmatch(after):
        return None
    # Ensure the selected result is the final numeric token in the full answer.
    return matches[-1] if matches and matches[-1].group() == result.group() else None


def _without_trailing_parenthetical(text: str) -> str:
    """Remove explanatory trailing parentheses, never arbitrary middle prose."""
    result = text.strip()
    while True:
        match = re.search(r"\s*\([^()\n]*\)\s*[.!?]*\s*$", result)
        if match is None:
            return result
        result = result[:match.start()].rstrip()


def _numeric(text: str, *, explicit_final_line: bool = False) -> str | None:
    matches = list(_NUMBER.finditer(text))
    if len(matches) == 1:
        selected = matches[0]
    elif explicit_final_line:
        selected = _terminal_equation_result(text, matches)
        if selected is None:
            # A marked answer may state one result and then show supporting
            # arithmetic in a trailing parenthetical, e.g. "Jasper has 12 ...
            # (6 + 6)". Select the sole number in the declarative main clause;
            # this is deliberately not a last-number heuristic.
            main_clause = _without_trailing_parenthetical(text)
            main_matches = list(_NUMBER.finditer(main_clause))
            if len(main_matches) != 1:
                return None
            selected = main_matches[0]
    else:
        return None
    if _NUMERIC_UNCERTAINTY.search(text) or _CONTRADICTION.search(text) or (
            _APPROXIMATE.search(text) and not explicit_final_line):
        return None
    token = selected.group().replace(",", "").replace(" ", "")
    percent = token.endswith("%")
    token = token.rstrip("%")
    try:
        value = Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None
    # Percent notation is retained, not silently converted to another unit.
    return str(value) + ("%" if percent else "")


def score_output(output: str, gold_answers: list[str], task: str) -> dict[str, Any]:
    """Return prediction, exact/F1, extraction validity and an explicit reason.

    GSM8K requires one unambiguous numeric final answer. QA accepts an explicit
    final answer or a short, single-line answer. Golds must come from the frozen
    task manifest, and a missing gold is not treated as an unanswerable question.
    """
    if not isinstance(gold_answers, (list, tuple)) or not gold_answers:
        return _invalid("missing_gold_answers")
    answer, valid, reason = _extract(output)
    if not valid:
        return _invalid(reason)
    if task.lower() == "gsm8k":
        prediction = _numeric(
            answer, explicit_final_line=reason == "explicit_first_final_answer_line")
        if prediction is None:
            return _invalid("ambiguous_or_missing_numeric_answer", answer)
        golds = []
        for gold in gold_answers:
            gold_answer, gold_valid, _ = _extract(str(gold))
            parsed = (_numeric(gold_answer, explicit_final_line=True)
                      if gold_valid else None)
            if parsed is not None:
                golds.append(parsed)
        if not golds:
            return _invalid("invalid_numeric_gold", prediction)
        exact = prediction in golds
        return {"prediction": prediction, "exact": exact, "f1": float(exact),
                "valid": True, "reason": reason}
    if task.lower() not in {"squad", "squad2", "squad2.0"}:
        return _invalid("unsupported_task")
    # Avoid scoring a chain of thought or a multi-sentence answer as an extract.
    if ("\n" in answer or len(answer.split()) > 32 or len(answer) > 240
            or re.search(r"\b(?:because|therefore|however)\b", answer, re.I)):
        return _invalid("ambiguous_or_long_qa_answer", answer)
    if answer.lower().strip(" .!?") in _NO_ANSWER:
        prediction = ""
    else:
        if re.search(r"\b(?:not|isn't|isnt|cannot|maybe)\b|\bdon't know\b", answer, re.I):
            return _invalid("ambiguous_qa_negation_or_uncertainty", answer)
        prediction = answer
    normalized = normalize_qa(prediction)
    normalized_golds = ["" if str(g).lower().strip(" .!?") in _NO_ANSWER
                        else normalize_qa(str(g)) for g in gold_answers]
    exact = normalized in normalized_golds
    scores = []
    tokens = normalized.split()
    for gold in normalized_golds:
        gold_tokens = gold.split()
        if not tokens or not gold_tokens:
            scores.append(float(tokens == gold_tokens))
            continue
        overlap = sum((Counter(tokens) & Counter(gold_tokens)).values())
        if not overlap:
            scores.append(0.0)
            continue
        precision, recall = overlap / len(tokens), overlap / len(gold_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return {"prediction": prediction, "exact": exact, "f1": max(scores),
            "valid": True, "reason": reason}
