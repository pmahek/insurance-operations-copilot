
import ast
import re
import unittest
from pathlib import Path

import pymupdf


SOURCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "semantic_pdf_search.py"
)


def load_extraction_functions():
    """Load extraction functions without starting Streamlit."""

    source = SOURCE_FILE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    wanted_functions = {
        "clean_text",
        "split_into_sentences",
        "create_chunks",
        "split_page_sections",
        "extract_pdf_chunks",
    }

    selected = []

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in wanted_functions
        ):
            selected.append(node)

        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "SECTION_HEADING_RE"
                for target in node.targets
            ):
                selected.append(node)

    class FakeStreamlit:
        @staticmethod
        def cache_data(*args, **kwargs):
            def decorator(function):
                return function
            return decorator

    namespace = {
        "re": re,
        "pymupdf": pymupdf,
        "st": FakeStreamlit(),
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


FUNCTIONS = load_extraction_functions()
extract_pdf_chunks = FUNCTIONS["extract_pdf_chunks"]


def create_test_pdf():
    """Create a small in-memory, two-page policy PDF."""

    pdf = pymupdf.open()

    first_page = pdf.new_page()
    first_page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 250),
        (
            "Policy overview.\n"
            "C. 1. Hospitalization Benefits\n"
            "Hospital treatment covers eligible inpatient care."
        ),
        fontsize=11,
    )

    second_page = pdf.new_page()
    second_page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 250),
        "Eligible treatment continues on the following page.",
        fontsize=11,
    )

    pdf_bytes = pdf.tobytes()
    pdf.close()

    return pdf_bytes


class TestPDFExtraction(unittest.TestCase):

    def setUp(self):
        self.pdf_bytes = create_test_pdf()

    def test_empty_input_returns_no_records(self):
        self.assertEqual(extract_pdf_chunks([]), [])

    def test_extracts_text_from_pdf(self):
        records = extract_pdf_chunks([
            ("sample_policy.pdf", self.pdf_bytes)
        ])

        self.assertTrue(records)

        combined_text = " ".join(
            record["text"] for record in records
        )

        self.assertIn("Hospital treatment", combined_text)
        self.assertIn("Eligible treatment continues", combined_text)

    def test_preserves_document_and_page_numbers(self):
        records = extract_pdf_chunks([
            ("sample_policy.pdf", self.pdf_bytes)
        ])

        self.assertTrue(
            all(
                record["document"] == "sample_policy.pdf"
                for record in records
            )
        )

        self.assertTrue(
            all(record["page"] in (1, 2) for record in records)
        )

        self.assertIn(1, [record["page"] for record in records])
        self.assertIn(2, [record["page"] for record in records])

    def test_preserves_section_metadata_across_pages(self):
        records = extract_pdf_chunks([
            ("sample_policy.pdf", self.pdf_bytes)
        ])

        section_records = [
            record
            for record in records
            if record["section_code"] == "1"
        ]

        self.assertTrue(section_records)

        self.assertTrue(
            all(
                record["section_title"]
                == "Hospitalization Benefits"
                for record in section_records
            )
        )

        self.assertTrue(
            any(record["page"] == 2 for record in section_records)
        )

    def test_every_record_has_required_metadata(self):
        records = extract_pdf_chunks([
            ("sample_policy.pdf", self.pdf_bytes)
        ])

        required_fields = {
            "document",
            "page",
            "chunk_index",
            "section_code",
            "section_title",
            "text",
        }

        self.assertTrue(records)

        for record in records:
            with self.subTest(record=record):
                self.assertTrue(
                    required_fields.issubset(record.keys())
                )
                self.assertTrue(record["text"].strip())
                self.assertIsInstance(record["page"], int)
                self.assertIsInstance(record["chunk_index"], int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
