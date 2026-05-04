import streamlit as st
import nltk
from collections import Counter

# NLTK FIX
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
# utils
from utils.extractor import extract_text_pdf, extract_text_image
from utils.summarizer import clean_text
from utils.report import create_pdf

# UI
st.set_page_config(page_title="DocuMind AI", layout="wide")

st.markdown("""
<style>
.stApp {background-color: #0E1117; color: white;}
h1 {text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("📄 DocuMind AI (Ultimate Pro)")

# simple summary
def get_summary(text):
    return text[:300]

# keywords
def get_keywords(text):
    words = text.split()
    return Counter(words).most_common(5)

# upload
file = st.file_uploader("Upload PDF/Image", type=["pdf","png","jpg","jpeg"])

if file:
    if file.type == "application/pdf":
        text = extract_text_pdf(file)
    else:
        text = extract_text_image(file)

    clean = clean_text(text)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📃 Text","🧠 Summary","🔥 Keywords","🤖 AI Chat","📥 Report"]
    )

    with tab1:
        st.write(text[:1500])

    with tab2:
        summary = get_summary(clean)
        st.write(summary)

    with tab3:
        st.write(get_keywords(clean))

    with tab4:
        if "history" not in st.session_state:
            st.session_state.history = []

        query = st.text_input("Ask anything:")

        if query:
            st.session_state.history.append((query, ans))

        for q,a in st.session_state.history:
            st.write("🧑", q)
            st.write("🤖", a)

    with tab5:
        pdf = create_pdf(summary)
        with open(pdf, "rb") as f:
            st.download_button("Download Report", f)