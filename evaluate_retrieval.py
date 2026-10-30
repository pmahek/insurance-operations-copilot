
import ast
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "semantic_pdf_search.py"
PDF_FILE = BASE_DIR / "evaluation_policy.pdf"
QUESTIONS_FILE = BASE_DIR / "evaluation_questions.json"

TOP_K = 5


# 1. Load existing functions without running the Streamlit app.

source = APP_FILE.read_text(encoding="utf-8")
tree = ast.parse(source)

required_functions = {
    "clean_text",
    "split_into_sentences",
    "create_chunks",
    "split_page_sections",
    "extract_pdf_chunks",
    "embed_passages",
    "build_tfidf_index",
    "normalized_phrase",
    "section_targets_for_query",
    "is_near_duplicate",
    "retrieve_passages",
}

selected_nodes = []

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in required_functions:
        node.decorator_list = []
        selected_nodes.append(node)

    elif isinstance(node, ast.Assign):
        if any(
            isinstance(target, ast.Name)
            and target.id == "SECTION_HEADING_RE"
            for target in node.targets
        ):
            selected_nodes.append(node)

if not required_functions.issubset(
    {
        node.name
        for node in selected_nodes
        if isinstance(node, ast.FunctionDef)
    }
):
    raise RuntimeError("Could not find all required functions in the app.")

namespace = {
    "re": re,
    "SequenceMatcher": SequenceMatcher,
    "pymupdf": pymupdf,
    "SentenceTransformer": SentenceTransformer,
    "TfidfVectorizer": TfidfVectorizer,
    "cosine_similarity": cosine_similarity,
}

module = ast.Module(body=selected_nodes, type_ignores=[])
exec(compile(module, str(APP_FILE), "exec"), namespace)


# 2. Load the embedding model.

print("Loading embedding model...")
namespace["model"] = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Cache embeddings and TF-IDF during this evaluation run.

embedding_cache = {}
tfidf_cache = {}

original_embed = namespace["embed_passages"]
original_tfidf = namespace["build_tfidf_index"]


def cached_embed(passages):
    key = tuple(passages)

    if key not in embedding_cache:
        embedding_cache[key] = original_embed(key)

    return embedding_cache[key]


def cached_tfidf(passages):
    key = tuple(passages)

    if key not in tfidf_cache:
        tfidf_cache[key] = original_tfidf(key)

    return tfidf_cache[key]


namespace["embed_passages"] = cached_embed
namespace["build_tfidf_index"] = cached_tfidf


# 4. Extract passages from the test PDF.

if not PDF_FILE.exists():
    raise FileNotFoundError(f"Test PDF not found: {PDF_FILE}")

if not QUESTIONS_FILE.exists():
    raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")

print("\nExtracting policy passages...")

pdf_bytes = PDF_FILE.read_bytes()

records = namespace["extract_pdf_chunks"](
    ((PDF_FILE.name, pdf_bytes),)
)

if not records:
    raise RuntimeError("No passages could be extracted from the PDF.")

print(f"Extracted {len(records)} passages.")

section_counts = Counter(
    record["section_code"]
    for record in records
    if record.get("section_code")
)

print("\n========== EXTRACTED SECTIONS ==========")

for code in sorted(section_counts, key=int):
    titles = {
        record["section_title"]
        for record in records
        if record.get("section_code") == code
    }

    print(
        f"C.{code} | "
        f"{', '.join(titles)} | "
        f"{section_counts[code]} passages"
    )

print("========================================\n")


# 5. Load evaluation questions.

questions = json.loads(
    QUESTIONS_FILE.read_text(encoding="utf-8")
)


# 6. Evaluate retrieval.

results = []
reciprocal_ranks = []

print("\nRunning retrieval evaluation...\n")

for item in questions:
    question = item["question"]
    expected = {str(code) for code in item["expected_sections"]}

    retrieved = namespace["retrieve_passages"](
        question,
        records,
        top_k=TOP_K,
    )

    # Print detailed diagnostics ONLY for the two failing questions.

    if expected in ({"3"}, {"5"}):
        print("\n========== RETRIEVED PASSAGE DETAILS ==========")
        print(f"Question: {question}")
        print(f"Expected section(s): {sorted(expected)}")

        for rank, record in enumerate(retrieved, start=1):
            print(f"\nRank: {rank}")
            print(f"Section: C.{record.get('section_code')}")
            print(f"Title: {record.get('section_title')}")
            print(f"Combined score: {record['combined_score']:.4f}")
            print(f"Semantic score: {record['semantic_score']:.4f}")
            print(f"Keyword score: {record['keyword_score']:.4f}")
            print(f"Text: {record['text'][:400]}")

        print("\n===============================================\n")

    retrieved_sections = [
        str(record["section_code"])
        for record in retrieved
        if record.get("section_code") is not None
    ]

    relevant_ranks = [
        rank
        for rank, code in enumerate(retrieved_sections, start=1)
        if code in expected
    ]

    hit = bool(relevant_ranks)
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0

    reciprocal_ranks.append(reciprocal_rank)

    results.append({
        "question": question,
        "expected_sections": sorted(expected),
        "retrieved_sections": retrieved_sections,
        "hit_at_5": hit,
        "reciprocal_rank": reciprocal_rank,
    })

    print(f"Question: {question}")
    print(f"Expected sections: {sorted(expected)}")
    print(f"Retrieved sections: {retrieved_sections}")
    print(f"Hit@5: {'PASS' if hit else 'FAIL'}")
    print("-" * 60)


# 7. Calculate overall metrics.

hit_rate = sum(
    result["hit_at_5"] for result in results
) / len(results)

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n========== EVALUATION SUMMARY ==========")
print(f"Questions evaluated: {len(results)}")
print(f"Hit@5: {hit_rate:.2%}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
print("========================================")


# 8. Save detailed results.

output_file = BASE_DIR / "retrieval_evaluation_results.json"

output_file.write_text(
    json.dumps(results, indent=4),
    encoding="utf-8",
)

print(f"\nDetailed results saved to: {output_file}")
