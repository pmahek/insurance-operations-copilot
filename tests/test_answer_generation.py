
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "semantic_pdf_search.py"
)


def load_generate_answer():
    """Load generate_answer without starting Streamlit or Ollama."""

    source = SOURCE_FILE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_answer"
    ]

    namespace = {"re": re}

    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(SOURCE_FILE),
            "exec",
        ),
        namespace,
    )

    return namespace["generate_answer"]


class TestAnswerGeneration(unittest.TestCase):

    def setUp(self):
        self.generate_answer = load_generate_answer()

        self.records = [
            {
                "document": "sample_policy.pdf",
                "page": 3,
                "chunk_index": 0,
                "section_code": "1",
                "section_title": "Hospitalization Benefits",
                "text": (
                    "Eligible inpatient hospital treatment "
                    "is covered subject to policy terms."
                ),
            },
            {
                "document": "sample_policy.pdf",
                "page": 5,
                "chunk_index": 1,
                "section_code": "2",
                "section_title": "Premium Payment",
                "text": "The premium is payable annually.",
            },
        ]

    def run_with_model_response(self, answer):
        """Replace the Ollama call with a predictable test response."""

        fake_response = SimpleNamespace(
            message=SimpleNamespace(content=answer)
        )

        with patch.dict(
            self.generate_answer.__globals__,
            {"ollama": SimpleNamespace(
                chat=lambda **kwargs: fake_response
            )},
        ):
            return self.generate_answer(
                "What does the policy cover?",
                self.records.copy(),
            )

    def test_assigns_source_ids_to_retrieved_records(self):
        self.run_with_model_response("Covered inpatient care [S1].")

        self.assertEqual(self.records[0]["source_id"], "S1")
        self.assertEqual(self.records[1]["source_id"], "S2")

    def test_prompt_includes_document_page_section_and_passage(self):
        captured = {}

        def fake_chat(**kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                message=SimpleNamespace(
                    content="Inpatient care is covered [S1]."
                )
            )

        with patch.dict(
            self.generate_answer.__globals__,
            {"ollama": SimpleNamespace(chat=fake_chat)},
        ):
            self.generate_answer(
                "What does the policy cover?",
                [record.copy() for record in self.records],
            )

        prompt = captured["messages"][1]["content"]

        self.assertIn("sample_policy.pdf", prompt)
        self.assertIn("Page: 3", prompt)
        self.assertIn("C.1 Hospitalization Benefits", prompt)
        self.assertIn(
            "Eligible inpatient hospital treatment",
            prompt,
        )
        self.assertIn("[S1]", prompt)
        self.assertIn("What does the policy cover?", prompt)

    def test_valid_citation_does_not_trigger_warning(self):
        answer = self.run_with_model_response(
            "Eligible inpatient care is covered [S1]."
        )

        self.assertNotIn("Citation warning", answer)

    def test_unknown_citation_triggers_warning(self):
        answer = self.run_with_model_response(
            "The policy covers this benefit [S9]."
        )

        self.assertIn("Citation warning", answer)
        self.assertIn("S9", answer)

    def test_missing_citation_triggers_warning(self):
        answer = self.run_with_model_response(
            "Eligible inpatient care is covered."
        )

        self.assertIn("Citation warning", answer)
        self.assertIn("No source-ID citations", answer)

    def test_multiple_valid_citations_are_accepted(self):
        answer = self.run_with_model_response(
            "See both passages [S1] [S2]."
        )

        self.assertNotIn("Citation warning", answer)

    def test_whitespace_is_removed_from_model_response(self):
        answer = self.run_with_model_response(
            "  Covered inpatient care [S1].  \n"
        )

        self.assertTrue(answer.startswith("Covered inpatient care"))
        self.assertFalse(answer.startswith(" "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
