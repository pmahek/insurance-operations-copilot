# Insurance Operations Copilot

An AI-powered assistant for searching insurance policy documents, retrieving relevant clauses, and generating evidence-grounded answers using semantic search, keyword retrieval, and a locally running language model.

## Problem Statement

Insurance policy documents contain detailed terms, conditions, exclusions, eligibility criteria, and coverage limits. Finding the correct clause manually can be time-consuming, particularly when information is spread across multiple pages and sections.

Insurance Operations Copilot helps users locate relevant policy information, inspect supporting passages, and generate answers grounded in the retrieved document content.

## Features

- Upload insurance policy and standard operating procedure (SOP) PDFs.
- Extract document text while preserving page and section information.
- Split documents into searchable passages with associated metadata.
- Generate semantic embeddings using Sentence Transformers.
- Combine semantic similarity with TF-IDF keyword retrieval.
- Generate evidence-grounded answers using Qwen 2.5 14B through Ollama.
- Display retrieved passages and source references for verification.
- Validate generated citations against available source IDs.
- Cache document extraction and embeddings to reduce repeated processing.
- Preserve uploaded document data during the application session.
- Clear uploaded documents through the application interface.
- Run automated tests for document processing, retrieval, answer generation, and upload persistence.

## Technology Stack

- **Language:** Python
- **Frontend:** Streamlit
- **PDF Processing:** PyMuPDF
- **Semantic Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Keyword Retrieval:** scikit-learn TF-IDF
- **Language Model Runtime:** Ollama
- **LLM:** Qwen 2.5 14B
- **Testing:** Python `unittest`

## Architecture

1. **Document ingestion:** Accept insurance policy and SOP PDFs through the Streamlit interface.
2. **Text extraction:** Extract text and page information using PyMuPDF.
3. **Chunking:** Split extracted text into searchable passages while retaining relevant metadata.
4. **Indexing:** Generate semantic embeddings and TF-IDF representations for document passages.
5. **Hybrid retrieval:** Rank passages using semantic similarity and keyword relevance, with section-aware retrieval where applicable.
6. **Answer generation:** Pass the user's question and retrieved evidence to Qwen through Ollama.
7. **Citation validation:** Check generated source references against the available source IDs and flag citation issues.
8. **Evidence display:** Present answers alongside supporting passages so users can verify the information against the original document.

## Setup and Installation

### Prerequisites

- Python 3.13
- Git
- Ollama installed and accessible locally
- Sufficient system memory and processing resources for the selected language model

### 1. Clone the repository

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual URL of your GitHub repository.

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd insurance-operations-copilot
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Download the language model

Ensure Ollama is installed and running, then download the model used by the application:

```powershell
ollama pull qwen2.5:14b
```

The model download can require substantial disk space. Runtime memory requirements depend on the model configuration and available hardware.

### 5. Run the application

From the project directory, run:

```powershell
python -m streamlit run semantic_pdf_search.py --server.fileWatcherType none
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Example Questions

Try asking questions such as:

- What are the eligibility conditions for inpatient hospitalization?
- What does Section C.1 say about hospitalization?
- How does Section C.1 differ from Section C.6?
- Which policy passages support the answer to my question?

The answers depend on the contents of the uploaded documents and the passages retrieved by the application.

## Testing

The project includes **42 automated tests** covering:

- Text processing and section matching.
- Page and section metadata handling.
- Duplicate-passage detection.
- Passage retrieval, relevance ranking, and section-aware filtering.
- PDF text extraction and document metadata.
- Answer generation and citation validation.
- Uploaded document persistence and clearing.

Run the complete test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests use controlled inputs and mocked dependencies where appropriate, allowing core functionality to be tested without launching the full Streamlit application or making live LLM requests.

## Limitations

- Answer quality depends on the extracted document text and the relevance of retrieved passages.
- Scanned PDFs without selectable text may require OCR, which is not currently implemented.
- Citation validation checks source references but does not independently prove that every generated claim is supported by its cited passage.
- Complex tables, unusual PDF layouts, and poorly formatted documents may affect extraction quality.
- The local language model requires sufficient system resources and may have higher response latency on less powerful hardware.
- Users should verify important coverage, exclusions, eligibility requirements, and other policy details against the original document and confirm interpretations with the insurer where necessary.

## Future Improvements

- Evaluate retrieval quality using a labelled dataset and ranking metrics such as Precision@K and Recall@K.
- Add OCR support for scanned insurance documents.
- Improve claim-level citation validation and evidence-grounding evaluation.
- Add structured checks for missing documents and policy information.
- Expand test coverage with edge cases and integration tests.
- Add Docker-based deployment and a production-ready API.

## Disclaimer

Insurance Operations Copilot is a portfolio project for document search and information assistance. It does not replace the original policy wording, insurer confirmation, or professional insurance advice. Generated answers may be incomplete or incorrect and should be verified before making insurance-related decisions.