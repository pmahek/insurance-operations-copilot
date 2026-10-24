
import streamlit as st
import pymupdf

# 1. Configure the web application
st.set_page_config(
    page_title="Insurance Operations Copilot",
    page_icon="📄",
    layout="wide"
)

# 2. Display the application heading
st.title("Insurance Operations Copilot")

st.write(
    "Upload insurance documents, extract their text, "
    "and prepare them for intelligent search."
)

# 3. Allow users to upload multiple PDF files
uploaded_files = st.file_uploader(
    "Upload insurance PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# 4. Process the uploaded documents
if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

    for uploaded_file in uploaded_files:
        with st.expander(uploaded_file.name):

            try:
                # Read the uploaded PDF into memory
                pdf_bytes = uploaded_file.getvalue()

                # Open the PDF using PyMuPDF
                with pymupdf.open(
                    stream=pdf_bytes,
                    filetype="pdf"
                ) as pdf:

                    # Extract text from every page
                    extracted_text = "\n".join(
                        page.get_text()
                        for page in pdf
                    )

                    # Display basic document information
                    st.write(f"**Pages:** {len(pdf)}")

                    if extracted_text.strip():
                        st.text_area(
                            "Extracted text",
                            value=extracted_text,
                            height=300,
                            key=uploaded_file.name
                        )
                    else:
                        st.warning(
                            "No selectable text found. "
                            "This PDF may be scanned."
                        )

            except Exception as error:
                st.error(
                    f"Could not process this PDF: {error}"
                )
