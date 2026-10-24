
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


# 3. Split page text into overlapping chunks

def create_chunks(text, chunk_size=1000, overlap=150):
    # Step 1: Clean up unnecessary whitespace
    text = " ".join(text.split())

    # Step 2: Split the text into sentences
    sentences = text.replace("! ", ". ").replace("? ", ". ").split(". ")

    chunks = []
    current_chunk = ""

    # Step 3: Add complete sentences to each chunk
    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # Start a new chunk if adding this sentence exceeds the limit
        if (
            current_chunk
            and len(current_chunk) + len(sentence) + 1 > chunk_size
        ):
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # Add the sentence to the current chunk
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence

    # Step 4: Save the final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks



# 4. Extract text from uploaded PDFs
def extract_pdf_chunks(uploaded_files):
    records = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()

        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf:

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                for chunk in create_chunks(page_text):
                    records.append({
                        "document": uploaded_file.name,
                        "page": page_number,
                        "text": chunk
                    })

    return records


# 5. Let the employee upload documents
uploaded_files = st.file_uploader(
    "Upload insurance policy or SOP PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# 6. Ask a question
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


# 7. Retrieve and rank matching passages
if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    elif not query.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Extracting and searching documents..."):
            records = extract_pdf_chunks(uploaded_files)

            if not records:
                st.warning(
                    "No extractable text was found in these PDFs. "
                    "Scanned documents may require OCR."
                )

            else:
                passages = [
                    record["text"] for record in records
                ]

                document_embeddings = model.encode(
                    passages,
                    convert_to_numpy=True
                )

                query_embedding = model.encode(
                    [query],
                    convert_to_numpy=True
                )

                scores = cosine_similarity(
                    query_embedding,
                    document_embeddings
                )[0]

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
