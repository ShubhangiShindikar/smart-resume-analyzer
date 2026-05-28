import streamlit as st
import joblib
import re
import numpy as np
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from fpdf import FPDF

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("🧠 AI Resume Intelligence System")
st.subheader("Upload Resume + Job Description → Get AI Analysis, ATS Score & Skills")

st.divider()

# ---------------- LOAD MODEL ----------------

try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("❌ Model files not found. Run train_model.py first.")
    st.stop()

# ---------------- SKILLS LIST ----------------

SKILLS = [
    "python", "java", "c++", "sql",
    "machine learning", "deep learning",
    "tensorflow", "pytorch", "nlp",
    "pandas", "numpy", "django", "flask",
    "html", "css", "javascript"
]

# ---------------- FUNCTIONS ----------------

def clean_text(text):
    return re.sub(r'[^a-zA-Z]', ' ', text.lower())

def extract_skills(text):
    found = []
    text = text.lower()
    for skill in SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found.append(skill)
    return found

def ats_score(resume_text, jd_text):
    if not jd_text.strip():
        return 0
    vecs = vectorizer.transform([resume_text, jd_text])
    score = cosine_similarity(vecs[0], vecs[1])[0][0]
    return round(score * 100, 2)

def skill_match_score(skills, jd_text):
    jd_text = jd_text.lower()
    if not skills:
        return 0
    matched = sum(1 for s in skills if s in jd_text)
    return round((matched / len(skills)) * 100, 2)

def predict_role(text):
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    idx = np.argmax(probs)
    return model.classes_[idx], round(probs[idx] * 100, 2)

def show_chart(skills):
    if not skills:
        return
    df = {"Skills": skills, "Count": [1] * len(skills)}
    fig = px.bar(df, x="Skills", y="Count", title="Extracted Skills")
    st.plotly_chart(fig, use_container_width=True)

def generate_report(role, confidence, ats, skills):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="AI Resume Intelligence Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Predicted Role: {role}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"ATS Score: {ats:.2f}%", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Skills:", ln=True)
    pdf.multi_cell(200, 10, txt=", ".join(skills))

    pdf.output("resume_report.pdf")

# ---------------- INPUT UI ----------------

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("📌 Paste Job Description")

st.divider()

# ---------------- PROCESS ----------------

if resume_file:

    with open("temp.pdf", "wb") as f:
        f.write(resume_file.read())

    raw_text = extract_text("temp.pdf")
    cleaned_text = clean_text(raw_text)

    role, confidence = predict_role(cleaned_text)
    skills = extract_skills(cleaned_text)

    ats1 = ats_score(cleaned_text, jd_text)
    ats2 = skill_match_score(skills, jd_text)
    ats = round((ats1 + ats2) / 2, 2)

    # ---------------- RESULTS ----------------

    st.subheader("📊 Analysis Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("🎯 Predicted Role", role)
    col2.metric("📊 Confidence", f"{confidence:.2f}%")
    col3.metric("📈 ATS Score", f"{ats:.2f}%")

    st.divider()

    st.subheader("🧠 Extracted Skills")

    if skills:
        for i, skill in enumerate(skills, 1):
            st.write(f"{i}. {skill}")
    else:
        st.write("No skills detected")

    st.subheader("📊 Skill Visualization")
    show_chart(skills)

    st.divider()

    if st.button("📥 Generate Report"):
        generate_report(role, confidence, ats, skills)

        with open("resume_report.pdf", "rb") as f:
            st.download_button(
                "⬇ Download Report",
                f,
                file_name="AI_Resume_Report.pdf"
            )