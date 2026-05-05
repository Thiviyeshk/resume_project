import streamlit as st
import fitz
import pickle
import re

model = pickle.load(open("resume_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("AI Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    return text

if uploaded_file is not None:

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    st.subheader("Predicted Job Role:")
    st.success(prediction[0])