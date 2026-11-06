from __future__ import annotations

import unittest

from langchain_core.documents import Document

from evaluation import validate_evaluation_rows
from rag_assistant import format_sources, split_notes


def valid_rows() -> list[dict[str, str]]:
    return [
        {
            "question": f"Question {index + 1}",
            "expected_answer": "Expected response" if index < 10 else "",
            "answerable": "true" if index < 10 else "false",
        }
        for index in range(15)
    ]


class EvaluationValidationTests(unittest.TestCase):
    def test_accepts_assignment_question_mix(self) -> None:
        errors, answerability = validate_evaluation_rows(valid_rows())

        self.assertEqual(errors, [])
        self.assertEqual(sum(answerability), 10)

    def test_rejects_wrong_answerability_mix(self) -> None:
        rows = valid_rows()
        rows[10]["answerable"] = "true"
        rows[10]["expected_answer"] = "Unexpected answer"

        errors, _ = validate_evaluation_rows(rows)

        self.assertTrue(any("exactly 10 answerable" in error for error in errors))

    def test_rejects_blank_question_and_missing_expected_answer(self) -> None:
        rows = valid_rows()
        rows[0]["question"] = " "
        rows[1]["expected_answer"] = ""

        errors, _ = validate_evaluation_rows(rows)

        self.assertTrue(any("Question 1 is blank" in error for error in errors))
        self.assertTrue(any("Question 2 needs an expected answer" in error for error in errors))

    def test_rejects_incomplete_dataset(self) -> None:
        errors, _ = validate_evaluation_rows(valid_rows()[:14])

        self.assertTrue(any("exactly 15" in error for error in errors))


class RagUtilityTests(unittest.TestCase):
    def test_splitter_preserves_source_metadata(self) -> None:
        source = Document(
            page_content="Course notes. " * 100,
            metadata={"source_file": "notes.pdf", "page": 2},
        )

        chunks = split_notes([source], chunk_size=100, chunk_overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["source_file"] == "notes.pdf" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page"] == 2 for chunk in chunks))

    def test_citations_include_file_and_page(self) -> None:
        source = Document(
            page_content="Evidence from the notes.",
            metadata={"source_file": "notes.pdf", "page": 2},
        )

        formatted = format_sources([source])

        self.assertIn("[Source 1: notes.pdf, page 3]", formatted)
        self.assertIn("Evidence from the notes.", formatted)


if __name__ == "__main__":
    unittest.main()