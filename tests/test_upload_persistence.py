
import ast
import io
import unittest
from pathlib import Path


SOURCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "semantic_pdf_search.py"
)


class FakeSessionState(dict):
    """Dictionary-like substitute for Streamlit session state."""


def load_upload_functions():
    """Load the two functions without starting the Streamlit app."""

    source = SOURCE_FILE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    function_names = {
        "save_uploaded_documents",
        "clear_uploaded_documents",
    }

    selected_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in function_names
    ]

    fake_st = type(
        "FakeStreamlit",
        (),
        {"session_state": FakeSessionState()},
    )()

    namespace = {"st": fake_st}

    exec(
        compile(
            ast.Module(
                body=selected_functions,
                type_ignores=[],
            ),
            str(SOURCE_FILE),
            "exec",
        ),
        namespace,
    )

    return (
        namespace["save_uploaded_documents"],
        namespace["clear_uploaded_documents"],
        fake_st.session_state,
    )


class TestUploadPersistence(unittest.TestCase):

    def setUp(self):
        (
            self.save_documents,
            self.clear_documents,
            self.session_state,
        ) = load_upload_functions()

    def test_saves_uploaded_filename_and_bytes(self):
        pdf = io.BytesIO(b"sample PDF content")
        pdf.name = "policy.pdf"

        self.session_state["policy_upload"] = [pdf]

        self.save_documents()

        self.assertEqual(
            self.session_state["saved_pdf_items"],
            (("policy.pdf", b"sample PDF content"),),
        )

    def test_saves_multiple_uploaded_documents(self):
        first = io.BytesIO(b"first PDF")
        first.name = "policy1.pdf"

        second = io.BytesIO(b"second PDF")
        second.name = "policy2.pdf"

        self.session_state["policy_upload"] = [first, second]

        self.save_documents()

        saved = self.session_state["saved_pdf_items"]

        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0][0], "policy1.pdf")
        self.assertEqual(saved[1][0], "policy2.pdf")

    def test_empty_upload_does_not_overwrite_saved_documents(self):
        existing = (("existing.pdf", b"existing content"),)

        self.session_state["saved_pdf_items"] = existing
        self.session_state["policy_upload"] = []

        self.save_documents()

        self.assertEqual(
            self.session_state["saved_pdf_items"],
            existing,
        )

    def test_clear_removes_saved_documents(self):
        self.session_state["saved_pdf_items"] = (
            ("policy.pdf", b"sample content"),
        )
        self.session_state["policy_upload"] = ["uploaded file"]

        self.clear_documents()

        self.assertEqual(
            self.session_state["saved_pdf_items"],
            (),
        )

    def test_clear_resets_upload_selection(self):
        self.session_state["policy_upload"] = ["uploaded file"]

        self.clear_documents()

        self.assertIsNone(
            self.session_state["policy_upload"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
