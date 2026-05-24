import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re


st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.title("📊 Dashboard")
    st.markdown("### Instructions")
    st.write("""
    1. Upload resumes  
    2. Paste job description  
    3. Click analyze  
    """)
    
    st.markdown("---")
    st.info("🔒 Secure AI Screening System")

# -----------------------
# Header
# -----------------------
st.markdown("""
# 🚀 AI Resume Screening System
### Smart Hiring using AI + NLP
""")
st.markdown("---")

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return model, nlp

model, nlp = load_models()

# -----------------------
# Input Layout
# -----------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(
        "📂 Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

with col2:
    job_description = st.text_area("📝 Paste Job Description Here", height=200)

st.markdown("###")
analyze = st.button("🔍 Analyze Candidates")

# -----------------------
# FIXED Extract Text Function
# -----------------------
def extract_text_from_pdf(file):
    text = ""

    try:
        file.seek(0)  # 🔥 reset pointer
        file_bytes = file.read()

        # handle empty file
        if not file_bytes:
            return "EMPTY_FILE"

        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        for page in pdf:
            text += page.get_text()

    except Exception as e:
        return f"ERROR: {str(e)}"

    return text

# -----------------------
# Known Skills
# -----------------------
KNOWN_SKILLS = [
    "Python","SQL","Machine Learning","Deep Learning","NLP",
    "Data Analysis","Excel","Power BI","Tableau","TensorFlow","PyTorch"
]

# -----------------------
# Resume Entity Extraction
# -----------------------
def extract_resume_entities(text):
    doc = nlp(text)

    skills = [skill for skill in KNOWN_SKILLS if skill.lower() in text.lower()]

    education_keywords = ["B.Sc","B.Tech","M.Sc","M.Tech","MBA","Bachelor","Master"]
    education = []

    for keyword in education_keywords:
        matches = re.findall(rf"{keyword}.*", text)
        education.extend(matches)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            education.append(ent.text)

    experience = []
    lines = text.split("\n")

    for line in lines:
        if re.search(r"experience|worked|intern", line, re.I):
            experience.append(line.strip())

    return {
        "skills": list(set(skills)),
        "education": education if education else ["N/A"],
        "experience": experience if experience else ["N/A"]
    }

# -----------------------
# Main Logic
# -----------------------
if analyze:

    if not uploaded_files or not job_description:
        st.warning("⚠️ Please upload resumes and enter job description")

    else:

        resumes = []
        names = []
        entities_list = []

        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):

            # 🚨 extra safety check
            if file.size == 0:
                st.warning(f"{file.name} is empty!")
                continue

            with st.spinner(f"Processing {file.name}..."):
                text = extract_text_from_pdf(file)

                # 🚨 handle errors safely
                if text == "EMPTY_FILE":
                    st.error(f"{file.name} is empty or corrupted ❌")
                    continue

                if text.startswith("ERROR"):
                    st.error(f"Error reading {file.name}: {text}")
                    continue

                entities = extract_resume_entities(text)

            resumes.append(text)
            names.append(file.name)
            entities_list.append(entities)

            progress_bar.progress((i+1)/len(uploaded_files))

        # 🚨 prevent crash if all files failed
        if len(resumes) == 0:
            st.error("No valid resumes processed!")
            st.stop()

        # -----------------------
        # Semantic Similarity
        # -----------------------
        with st.spinner("Computing similarity..."):
            job_embedding = model.encode([job_description])
            resume_embeddings = model.encode(resumes)

            similarity_scores = cosine_similarity(job_embedding, resume_embeddings)[0]
            semantic_scores = [round(score*100,2) for score in similarity_scores]

        # -----------------------
        # Skill Matching
        # -----------------------
        job_skills = [s for s in KNOWN_SKILLS if s.lower() in job_description.lower()]
        skill_scores = []

        for r in entities_list:
            if len(job_skills) == 0:
                skill_scores.append(0)
            else:
                matched = len(set(r["skills"]).intersection(set(job_skills)))
                score = round((matched/len(job_skills))*100,2)
                skill_scores.append(score)

        # -----------------------
        # Final Score
        # -----------------------
        final_scores = [
            round(0.7*s + 0.3*k ,2)
            for s,k in zip(semantic_scores,skill_scores)
        ]

        # -----------------------
        # Results Table
        # -----------------------
        results = pd.DataFrame({
            "Candidate": names,
            "Final Score (%)": final_scores,
            "Semantic Score (%)": semantic_scores,
            "Skill Match (%)": skill_scores,
            "Skills": [", ".join(r["skills"]) for r in entities_list],
            "Education": [", ".join(r["education"]) for r in entities_list],
            "Experience": [", ".join(r["experience"]) for r in entities_list]
        }).sort_values(by="Final Score (%)", ascending=False)

        # -----------------------
        # Display Results
        # -----------------------
        st.markdown("## 🏆 Candidate Ranking")
        st.dataframe(results, use_container_width=True)

        # -----------------------
        # Top Candidate
        # -----------------------
        top = results.iloc[0]

        st.markdown("## 🌟 Top Candidate")

        st.markdown(f"""
        <div style="background-color:#d4edda;padding:20px;border-radius:10px">
            <h3>{top['Candidate']}</h3>
            <p><b>Final Score:</b> {top['Final Score (%)']}%</p>
            <p><b>Skills:</b> {top['Skills']}</p>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------
        # Score Breakdown
        # -----------------------
        st.markdown("## 📊 Score Breakdown")

        for _, row in results.iterrows():
            st.write(f"**{row['Candidate']}**")
            st.progress(int(row["Final Score (%)"]))

        # -----------------------
        # Download
        # -----------------------
        st.markdown("## ⬇️ Export Results")

        st.download_button(
            "Download CSV",
            data=results.to_csv(index=False),
            file_name="candidate_ranking.csv"
        )