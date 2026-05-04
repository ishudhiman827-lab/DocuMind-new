import streamlit as st
import nltk
import joblib
import re
import spacy
import os
from collections import Counter

# NLTK fix
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# spaCy model load
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# load ML model
model = joblib.load("model.pkl")

# utils
from utils.extractor import extract_text_pdf, extract_text_image
from utils.summarizer import clean_text
from utils.report import create_pdf

st.set_page_config(page_title="DocuMind AI", layout="wide")
st.title("📄 DocuMind AI - ML Powered")

# ================= FUNCTIONS ================= #

def get_summary(text):
    return text[:300]

def get_keywords(text):
    return Counter(text.split()).most_common(5)

# 🔥 ML classification
def classify_document(text):
    return model.predict([text])[0]

# 🔥 Confidence
def get_confidence(text):
    probs = model.predict_proba([text])[0]
    return round(max(probs)*100, 2)

# 🔥 REAL NER
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# 🔥 Sensitive detection
def detect_sensitive(text):
    patterns = {
        "PAN": r"[A-Z]{5}[0-9]{4}[A-Z]",
        "Aadhaar": r"\b\d{12}\b",
        "Phone": r"\b\d{10}\b"
    }
    results = {}
    for k,v in patterns.items():
        matches = re.findall(v, text)
        if matches:
            results[k] = matches
    return results

# ================= UI ================= #

file = st.file_uploader("Upload PDF/Image", type=["pdf","png","jpg","jpeg"])

if file:
    if file.type == "application/pdf":
        text = extract_text_pdf(file)
    else:
        text = extract_text_image(file)

    clean = clean_text(text)

    category = classify_document(clean)
    confidence = get_confidence(clean)
    entities = extract_entities(clean)
    sensitive = detect_sensitive(clean)

    st.success(f"📂 Category: {category}")
    st.info(f"📊 Confidence: {confidence}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Text","Summary","Keywords","Entities","Report"]
    )

    with tab1:
        st.write(text[:1500])

    with tab2:
        summary = get_summary(clean)
        st.write(summary)

    with tab3:
        st.write(get_keywords(clean))

    with tab4:
        st.subheader("Named Entities")
        st.write(entities)

        st.subheader("Sensitive Data")
        st.write(sensitive if sensitive else "None")

    with tab5:
        report = f"""
        Category: {category}
        Confidence: {confidence}

        Summary:
        {summary}

        Entities:
        {entities}

        Sensitive:
        {sensitive}
        """

        pdf = create_pdf(report)

        with open(pdf, "rb") as f:
            st.download_button("Download Report", f)