
import re

import streamlit as st
import pymupdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Configure the Streamlit page
st.set_page_config(
    page_title="Insurance Semantic Search",
    page_icon="📄",
    layout="wide"
)

st.title("Insurance Operations Copilot")
st.subheader("Semantic Search Across Insurance PDFs")

st.write(
    "Upload insurance documents and ask a question "
    "to find the most relevant passages."
)


# 2. Load the embedding model once and reuse it
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# 3. Split page text into section-aware chunks
def create_chunks(text, chunk_size=700):
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Identify headings and group lines into sections.
    sections = []
    current_section = []

    for line in lines:
        is_heading = (
            len(line) <= 70
            and any(char.isalpha() for char in line)
            and line == line.upper()
        )

        if is_heading and current_section:
            sections.append(current_section)
            current_section = []

        current_section.append(line)

    if current_section:
        sections.append(current_section)

    chunks = []

    # Build chunks without crossing detected section boundaries.
    for section in sections:
        current_chunk = ""

        for line in section:
            if len(line) > chunk_size:
                units = re.split(r"(?<=[.!?])\s+", line)
            else:
                units = [line]

            for unit in units:
                unit = unit.strip()

                if not unit:
                    continue

                # Split any remaining oversized text into word groups.
                if len(unit) > chunk_size:
                    words = unit.split()
                    parts = []
                    part = ""

                    for word in words:
                        candidate = f"{part} {word}".strip()

                        if len(candidate) > chunk_size and part:
                            parts.append(part)
                            part = word
                        else:
                            part = candidate

                    if part:
                        parts.append(part)

                    units_to_add = parts
                else:
                    units_to_add = [unit]

                for piece in units_to_add:
                    candidate = f"{current_chunk} {piece}".strip()

                    if current_chunk and len(candidate) > chunk_size:
                        chunks.append(current_chunk)
                        current_chunk = piece
                    else:
                        current_chunk = candidate

        if current_chunk:
            chunks.append(current_chunk)

    return chunks


# 4. Extract PDF text and cache the resulting chunks
@st.cache_data(show_spinner=False)
def extract_pdf_chunks(file_items):
    records = []

    for filename, pdf_bytes in file_items:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                for chunk in create_chunks(page_text):
                    records.append({
                        "document": filename,
                        "page": page_number,
                        "text": chunk
                    })

    return records


# 5. Cache embeddings for document passages
@st.cache_data(show_spinner=False)
def embed_passages(passages):
    return model.encode(
        list(passages),
        convert_to_numpy=True
    )


# 6. Let the employee upload documents
uploaded_files = st.file_uploader(
    "Upload insurance policy or SOP PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# 7. Ask a question
with st.form("semantic_search_form"):
    query = st.text_input(
        "Enter your insurance operations question",
        placeholder="e.g. How soon must hospitalisation be reported?"
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=3
    )

    submitted = st.form_submit_button("Search documents")


# 8. Retrieve and rank matching passages
if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    elif not query.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Extracting and searching documents..."):

            # Convert uploaded files into cacheable filename-and-bytes pairs.
            file_items = tuple(
                (uploaded_file.name, uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            )

            # Reuse extracted chunks when the uploaded files are unchanged.
            records = extract_pdf_chunks(file_items)

            if not records:
                st.warning(
                    "No extractable text was found in these PDFs. "
                    "Scanned documents may require OCR."
                )

            else:
                passages = tuple(
                    record["text"] for record in records
                )

                # Reuse document embeddings for repeated searches.
                document_embeddings = embed_passages(passages)

                # Embed the new question each time.
                query_embedding = model.encode(
                    [query.strip()],
                    convert_to_numpy=True
                )

                # Calculate semantic similarity.
                scores = cosine_similarity(
                    query_embedding,
                    document_embeddings
                )[0]

                # Rank passages from highest to lowest score.
                ranked_indices = scores.argsort()[::-1][:top_k]

                st.success(
                    f"Found {len(ranked_indices)} matching passages."
                )

                for rank, index in enumerate(
                    ranked_indices, start=1
                ):
                    record = records[index]
                    score = scores[index]

                    st.markdown(f"### Result {rank}")

                    st.caption(
                        f"Source: {record['document']} | "
                        f"Page: {record['page']} | "
                        f"Similarity: {score:.4f}"
                    )

                    st.write(record["text"])
                    st.divider()
