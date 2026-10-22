
import streamlit as st
from pypdf import PdfReader
from groq import Groq

# Configure Streamlit page
st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Summarizer")
st.write(
    "Upload your PDF and let Groq AI create "
    "a clear and easy-to-understand summary."
)

# Read API key from Streamlit secrets
try:
    api_key = st.secrets["GROQ_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "API key not found! Add GROQ_API_KEY "
        "to .streamlit/secrets.toml."
    )
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=api_key)
except Exception as error:
    st.error(f"Could not initialize Groq: {error}")
    st.stop()

MODEL = "openai/gpt-oss-20b"


def extract_pdf_text(pdf_file):
    """Extract readable text from a PDF."""
    reader = PdfReader(pdf_file)
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append(
                f"Page {page_number}:\n{page_text}"
            )

    return "\n\n".join(extracted_pages)


def split_text(text, chunk_size=10000):
    """Split long PDF text into smaller sections."""
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def summarize_text(text, instructions):
    """Send text to Groq and return the AI response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content or ""


def summarize_pdf(pdf_text):
    """Summarize PDF sections and combine summaries."""
    chunks = split_text(pdf_text)
    summaries = []

    progress = st.progress(0, text="Starting summary...")

    try:
        for index, chunk in enumerate(chunks):
            section_summary = summarize_text(
                chunk,
                "Summarize this PDF section accurately. "
                "Include main ideas, important facts, key findings, "
                "names, numbers, and conclusions. "
                "Do not invent information."
            )

            summaries.append(section_summary)

            progress.progress(
                (index + 1) / len(chunks),
                text=f"Summarizing section {index + 1} "
                     f"of {len(chunks)}"
            )
    finally:
        progress.empty()

    if len(summaries) == 1:
        return summaries[0]

    combined_summaries = "\n\n".join(summaries)

    return summarize_text(
        combined_summaries,
        "Create one coherent summary from these section summaries. "
        "Use clear headings and bullet points. Remove repetition "
        "while preserving important facts and conclusions. "
        "Do not add unsupported information."
    )


# PDF uploader
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("✨ Summarize PDF", type="primary"):
        try:
            with st.spinner("Extracting PDF text..."):
                pdf_text = extract_pdf_text(uploaded_file)

            if not pdf_text.strip():
                st.warning(
                    "No readable text was found in this PDF. "
                    "Scanned PDFs require OCR."
                )
            else:
                st.info(
                    f"Extracted {len(pdf_text):,} characters."
                )

                with st.spinner("Groq AI is preparing your summary..."):
                    summary = summarize_pdf(pdf_text)

                st.subheader("📝 Your PDF Summary")
                st.markdown(summary)

                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name="pdf_summary.txt",
                    mime="text/plain"
                )

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.caption(
                "Check your API key, internet connection, "
                "model availability, and API usage limits."
            )
