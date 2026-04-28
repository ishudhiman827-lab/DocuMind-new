import nltk
nltk.download('punkt')
import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import re

# Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Chat ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF Report
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- FUNCTIONS ---------------- #

def extract_text_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_text_image(file):
    return pytesseract.image_to_string(Image.open(file))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text.lower()

# -------- Summary -------- #

def get_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)
    return " ".join([str(sentence) for sentence in summary])

# -------- Chat -------- #

def split_text(text):
    return [text[i:i+300] for i in range(0, len(text), 300)]

def get_answer(query, chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    index = similarity.argmax()
    return chunks[index]

# -------- PDF Report -------- #

def create_pdf(summary, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("DocuMind AI Report", styles['Title']))
    content.append(Paragraph("<br/><b>Summary:</b><br/>" + summary, styles['Normal']))

    doc.build(content)
    return filename

# ---------------- UI ---------------- #

st.set_page_config(page_title="DocuMind AI", layout="wide")

st.markdown("<h1 style='text-align:center; color:cyan;'>📄 DocuMind AI</h1>", unsafe_allow_html=True)
st.markdown("Upload document → Summary + Chat + Download Report 🔥")
st.markdown("---")

file = st.file_uploader("Upload PDF / Image", type=["pdf", "png", "jpg", "jpeg"])

if file:
    st.success("File uploaded successfully ✅")

    # Extract text
    if file.type == "application/pdf":
        text = extract_text_pdf(file)
    else:
        text = extract_text_image(file)

    if text:
        st.subheader("📃 Extracted Text")
        st.write(text[:1000])

        clean = clean_text(text)

        st.markdown("---")

        # -------- Summary -------- #
        st.subheader("🧠 Summary")
        summary = get_summary(clean)
        st.write(summary)

        # -------- Download PDF -------- #
        pdf_file = create_pdf(summary)

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📥 Download Report",
                data=f,
                file_name="DocuMind_Report.pdf",
                mime="application/pdf"
            )

        st.markdown("---")

        # -------- Chat -------- #
        st.subheader("💬 Chat with Document")

        chunks = split_text(text)
        query = st.text_input("Ask something from document:")

        if query:
            answer = get_answer(query, chunks)
            st.write("📌 Answer:")
            st.write(answer)

    else:
        st.error("Text extract nahi ho paaya ❌")