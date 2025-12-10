import streamlit as st
from src.extract_resume import extract_text_from_pdf
from src.extract_jd import extract_jd_text
from src.nlp_pipeline import (
    extract_keywords,
    calculate_similarity,
    find_missing_skills,
    ats_score,
)

st.title("ATS Resume Checker + Keyword Matcher")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description")

if uploaded_file and jd_text:

    st.subheader("Processing...")

    resume_text = extract_text_from_pdf(uploaded_file)
    jd_text_clean = extract_jd_text(jd_text)

    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text_clean)

    similarity = calculate_similarity(resume_text, jd_text_clean)
    missing = find_missing_skills(resume_keywords, jd_keywords)
    score = ats_score(similarity, len(missing))

    st.subheader("ATS Score")
    st.metric("Score", f"{score}/100")

    st.subheader("Missing Skills")
    st.write(missing)
