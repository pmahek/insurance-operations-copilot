
import ast
import re
import unittest
from difflib import SequenceMatcher
from pathlib import Path


SOURCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "semantic_pdf_search.py"
)


def load_pure_functions():
    """Load selected functions without executing Streamlit or model setup."""
    source = SOURCE_FILE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    wanted = {
        "clean_text",
        "split_into_sentences",
        "create_chunks",
        "split_page_sections",
        "normalized_phrase",
        "section_targets_for_query",
        "section_target_for_query",
        "is_near_duplicate",
    }

    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in wanted
    ]

    namespace = {
        "re": re,
        "SequenceMatcher": SequenceMatcher,
        "SECTION_HEADING_RE": re.compile(
            r"(?im)^\s*C\.\s*(\d{1,2})\.\s*([^\r\n]{1,100}?)\s*$"
        ),
    }

    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(SOURCE_FILE),
            "exec",
        ),
        namespace,
    )

    return namespace


FUNCTIONS = load_pure_functions()

clean_text = FUNCTIONS["clean_text"]
split_into_sentences = FUNCTIONS["split_into_sentences"]
create_chunks = FUNCTIONS["create_chunks"]
normalized_phrase = FUNCTIONS["normalized_phrase"]
section_targets_for_query = FUNCTIONS["section_targets_for_query"]
section_target_for_query = FUNCTIONS["section_target_for_query"]
is_near_duplicate = FUNCTIONS["is_near_duplicate"]


class TestTextProcessing(unittest.TestCase):

    def test_clean_text_removes_extra_whitespace(self):
        result = clean_text("  Policy   terms \n and conditions  ")
        self.assertEqual(result, "Policy terms and conditions")

    def test_clean_text_joins_hyphenated_line_break(self):
        result = clean_text("hospitali-\nzation")
        self.assertEqual(result, "hospitalization")

    def test_clean_text_preserves_clause_punctuation(self):
        result = clean_text("Section C.1: Hospitalization.")
        self.assertIn("C.1:", result)
        self.assertTrue(result.endswith("."))

    def test_sentence_split_protects_abbreviations(self):
        result = split_into_sentences(
            "Dr. Rao reviewed the policy. It covers treatment."
        )
        self.assertEqual(
            result,
            ["Dr. Rao reviewed the policy.", "It covers treatment."],
        )

    def test_create_chunks_empty_text(self):
        self.assertEqual(create_chunks("   "), [])

    def test_create_chunks_preserves_content(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = create_chunks(text, chunk_size=25, overlap=5)

        self.assertTrue(chunks)
        combined = " ".join(chunks)

        for sentence in [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]:
            self.assertIn(sentence, combined)

    def test_normalized_phrase_ignores_case_and_punctuation(self):
        self.assertEqual(
            normalized_phrase("Section C.1 — Hospitalization!"),
            "section c 1 hospitalization",
        )


class TestSectionMatching(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "section_code": "1",
                "section_title": "Inpatient Hospitalization Treatment",
            },
            {
                "section_code": "6",
                "section_title": "Day Care Treatment",
            },
            {
                "section_code": "4",
                "section_title": "Pre Hospitalization",
            },
        ]

    def test_explicit_section_code(self):
        self.assertEqual(
            section_targets_for_query("Explain Section C.1", self.records),
            {"1"},
        )

    def test_section_code_with_spacing(self):
        self.assertEqual(
            section_targets_for_query("Explain C 6", self.records),
            {"6"},
        )

    def test_inpatient_alias(self):
        self.assertEqual(
            section_targets_for_query(
                "What is inpatient hospitalization treatment?",
                self.records,
            ),
            {"1"},
        )

    def test_day_care_alias(self):
        self.assertEqual(
            section_targets_for_query(
                "Explain day care treatment",
                self.records,
            ),
            {"6"},
        )

    def test_unmatched_section_is_ignored(self):
        self.assertEqual(
            section_targets_for_query("Explain Section C.9", self.records),
            set(),
        )

    def test_single_target_returns_code_and_title(self):
        self.assertEqual(
            section_target_for_query("Explain C.6", self.records),
            ("6", "Day Care Treatment"),
        )

    def test_no_target_returns_empty_result(self):
        self.assertEqual(
            section_target_for_query("Explain the policy", self.records),
            (None, ""),
        )


class TestPageSections(unittest.TestCase):

    def test_identifies_section_code_and_title(self):
        page_text = (
            "C.1. Inpatient Hospitalization Treatment\n"
            "The policy covers hospitalization."
        )

        segments, code, title = FUNCTIONS["split_page_sections"](page_text)

        self.assertEqual(code, "1")
        self.assertEqual(title, "Inpatient Hospitalization Treatment")
        self.assertEqual(len(segments), 1)

    def test_splits_multiple_sections(self):
        page_text = (
            "C.1. Inpatient Hospitalization Treatment\n"
            "Hospitalization must meet policy conditions.\n"
            "C.6. Day Care Treatment\n"
            "Certain shorter treatments are covered."
        )

        segments, code, title = FUNCTIONS["split_page_sections"](page_text)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][1], "1")
        self.assertEqual(segments[1][1], "6")
        self.assertEqual(code, "6")
        self.assertEqual(title, "Day Care Treatment")

    def test_preserves_text_before_first_heading(self):
        page_text = (
            "General policy introduction.\n"
            "C.1. Inpatient Hospitalization Treatment\n"
            "Hospitalization details."
        )

        segments, _, _ = FUNCTIONS["split_page_sections"](page_text)

        self.assertEqual(len(segments), 2)
        self.assertIn("General policy introduction.", segments[0][0])
        self.assertIsNone(segments[0][1])


class TestDuplicateDetection(unittest.TestCase):

    def test_identical_text_is_duplicate(self):
        self.assertTrue(
            is_near_duplicate(
                "The policy covers hospitalization.",
                ["The policy covers hospitalization."],
            )
        )

    def test_empty_text_is_duplicate(self):
        self.assertTrue(is_near_duplicate("   ", []))

    def test_different_text_is_not_duplicate(self):
        self.assertFalse(
            is_near_duplicate(
                "The policy covers hospitalization.",
                ["Premium payment is due annually."],
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
