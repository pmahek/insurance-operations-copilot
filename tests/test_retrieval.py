
import ast
import re
import unittest
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SOURCE_FILE = (
    Path(__file__).resolve().parents[1]
    / "semantic_pdf_search.py"
)


class FakeModel:
    """Small deterministic embedding model for isolated tests."""

    def encode(self, texts, convert_to_numpy=True):
        vocabulary = [
            "hospital",
            "premium",
            "daycare",
            "treatment",
        ]
        vectors = []

        for text in texts:
            words = set(re.findall(r"[a-z]+", text.casefold()))
            vectors.append([
                float(word in words)
                for word in vocabulary
            ])

        return np.array(vectors, dtype=float)


def load_retrieval_function():
    """Load retrieval code without starting Streamlit."""

    source = SOURCE_FILE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    wanted = {
        "retrieve_passages",
        "normalized_phrase",
        "section_targets_for_query",
        "is_near_duplicate",
    }

    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in wanted
    ]

    def build_test_tfidf_index(passages):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(list(passages))
        return vectorizer, matrix

    namespace = {
        "model": FakeModel(),
        "np": np,
        "re": re,
        "SequenceMatcher": SequenceMatcher,
        "cosine_similarity": cosine_similarity,
        "TfidfVectorizer": TfidfVectorizer,
        "build_tfidf_index": build_test_tfidf_index,
        "embed_passages": lambda passages: FakeModel().encode(
            list(passages),
            convert_to_numpy=True,
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


FUNCTIONS = load_retrieval_function()
retrieve_passages = FUNCTIONS["retrieve_passages"]


class TestPassageRetrieval(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "text": "Hospital treatment covers eligible inpatient care.",
                "section_code": "1",
                "section_title": "Inpatient Hospitalization Treatment",
            },
            {
                "text": "Premium payment must be made annually.",
                "section_code": "2",
                "section_title": "Premium Payment",
            },
            {
                "text": (
                    "Daycare treatment may cover procedures "
                    "not requiring overnight admission."
                ),
                "section_code": "6",
                "section_title": "Day Care Treatment",
            },
        ]

    def test_empty_records_return_empty_list(self):
        self.assertEqual(
            retrieve_passages("hospital treatment", []),
            [],
        )

    def test_top_k_limits_number_of_results(self):
        results = retrieve_passages(
            "hospital treatment premium daycare",
            self.records,
            top_k=2,
        )
        self.assertLessEqual(len(results), 2)

    def test_results_include_score_fields(self):
        results = retrieve_passages(
            "hospital treatment",
            self.records,
        )
        self.assertTrue(results)

        expected_fields = {
            "combined_score",
            "semantic_score",
            "keyword_score",
            "section_match_bonus",
        }
        self.assertTrue(expected_fields.issubset(results[0]))

    def test_relevant_passage_ranks_above_unrelated_passage(self):
        records = [
            {
                "text": "Premium payment must be made annually.",
                "section_code": "2",
                "section_title": "Premium Payment",
            },
            {
                "text": "Hospital treatment covers eligible inpatient care.",
                "section_code": "1",
                "section_title": "Inpatient Hospitalization Treatment",
            },
        ]

        results = retrieve_passages(
            "hospital treatment",
            records,
        )

        self.assertTrue(results)
        self.assertIn("Hospital treatment", results[0]["text"])

    def test_explicit_section_query_restricts_results(self):
        results = retrieve_passages(
            "Explain Section C.6",
            self.records,
        )
        self.assertTrue(results)
        self.assertTrue(
            all(
                str(record["section_code"]) == "6"
                for record in results
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
