from __future__ import annotations

from typing import Any


ANSWERABILITY_VALUES = {
    "true": True,
    "yes": True,
    "1": True,
    "false": False,
    "no": False,
    "0": False,
}


def validate_evaluation_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[bool]]:
    """Validate the assignment's 10 answerable and 5 unanswerable questions."""
    errors: list[str] = []
    if len(rows) != 15:
        errors.append(f"The assignment requires exactly 15 questions; found {len(rows)}.")

    normalized: list[bool] = []
    for index, row in enumerate(rows, start=1):
        if not str(row.get("question", "")).strip():
            errors.append(f"Question {index} is blank.")
        value = str(row.get("answerable", "")).strip().lower()
        if value not in ANSWERABILITY_VALUES:
            errors.append(f"Question {index} answerable must be true/false, yes/no, or 1/0.")
            continue
        is_answerable = ANSWERABILITY_VALUES[value]
        normalized.append(is_answerable)
        if is_answerable and not str(row.get("expected_answer", "")).strip():
            errors.append(f"Question {index} needs an expected answer because it is answerable.")

    answerable_count = sum(normalized)
    unanswerable_count = len(normalized) - answerable_count
    if len(normalized) == 15 and (answerable_count != 10 or unanswerable_count != 5):
        errors.append(
            "Use exactly 10 answerable questions and 5 unanswerable questions "
            f"(found {answerable_count} and {unanswerable_count})."
        )
    return errors, normalized