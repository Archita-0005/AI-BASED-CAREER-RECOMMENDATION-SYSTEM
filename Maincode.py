import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

st.set_page_config(page_title="Advanced Career AI", layout="wide")

# -----------------------------
# STEP 1: STRONG DATASET
# -----------------------------

career_data = {
    "Software Engineer": ["python java c++ dsa algorithms coding system design"],
    "Frontend Developer": ["html css javascript react ui ux frontend"],
    "Backend Developer": ["nodejs express api database server backend"],
    "Full Stack Developer": ["html css js node react fullstack"],
    "Data Scientist": ["python statistics machine learning data analysis pandas numpy"],
    "AI Engineer": ["deep learning tensorflow pytorch ai neural networks"],
    "Data Analyst": ["excel sql powerbi data visualization analysis"],
    "Cloud Engineer": ["aws azure gcp cloud docker kubernetes devops"],
    "Cybersecurity Analyst": ["networking security ethical hacking penetration testing"],
    "DevOps Engineer": ["ci cd docker kubernetes automation cloud"],
    "Graphic Designer": ["photoshop illustrator design creativity ui"],
    "Content Writer": ["writing editing blogging content seo"],
    "Marketing Manager": ["marketing branding sales communication strategy"],
    "Accountant": ["accounting finance taxation tally auditing"],
}

# Expand dataset
dataset = []
for career, skills_list in career_data.items():
    for _ in range(200):
        skills = skills_list[0].split()
        random_skills = random.sample(skills, min(3, len(skills)))
        dataset.append({
            "skills": " ".join(skills + random_skills),
            "career": career
        })

df = pd.DataFrame(dataset)

# -----------------------------
# STEP 2: ADVANCED VECTORIZER
# -----------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),   # improves accuracy
    stop_words="english"
)

X = vectorizer.fit_transform(df["skills"])
y = df["career"]

model = MultinomialNB(alpha=0.1)  # smoothing tuned
model.fit(X, y)

# -----------------------------
# UI & DESIGN SYSTEM
# -----------------------------

# Inject custom CSS for a modern, sleek appearance
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #4A90E2;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .main-header {
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
        background: -webkit-linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 10px;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("<h1 class='main-header'>🚀 Career AI Navigator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Discover your optimal career path through intelligent NLP analysis</p>", unsafe_allow_html=True)

st.divider()

# Input Section with Column Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    skills_input = st.text_area(
        "✨ Describe your skills, tools, and interests:", 
        placeholder="e.g., Python, machine learning, data visualization, problem solving...",
        height=130
    )
    
    predict_clicked = st.button("Analyze My Profile", use_container_width=True)
# -----------------------------
# PREDICTION & RESULTS
# -----------------------------

if predict_clicked:
    if not skills_input.strip():
        st.warning("⚠️ Please enter your skills to get a valid prediction.")
    else:
        with st.spinner("🤖 Analyzing your profile against industry standards..."):
            user_vec = vectorizer.transform([skills_input.lower()])

            probs = model.predict_proba(user_vec)[0]
            classes = model.classes_

            results = list(zip(classes, probs))
            results = sorted(results, key=lambda x: x[1], reverse=True)

            best = results[0][0]

            st.divider()
            
            # Display results dynamically with Columns
            res_col1, res_col2 = st.columns([3, 2], gap="large")
            
            with res_col1:
                st.markdown("### 🎯 Recommended Career Paths")
                
                # Highlight Best Match in a success banner
                st.success(f"### 🏆 Top Match: {best}\n\n**{round(results[0][1]*100, 1)}%** confidence level based on your skillset.")
                
                st.markdown("#### 🌟 Alternative Strong Matches")
                # Show top alternatives using Progress Bars
                for career, prob in results[1:5]:
                    if prob > 0.05:  # filter very weak predictions
                        st.markdown(f"**{career}**")
                        prog_col1, prog_col2 = st.columns([5, 1])
                        with prog_col1:
                            st.progress(float(prob))
                        with prog_col2:
                            st.write(f"**{round(prob*100, 1)}%**")
            
            with res_col2:
                # -----------------------------
                # EXPLANATION FEATURE
                # -----------------------------
                with st.expander("🧠 Why this primary career?", expanded=True):
                    feature_names = vectorizer.get_feature_names_out()
                    log_probs = model.feature_log_prob_

                    class_index = list(classes).index(best)
                    top_features = sorted(
                        zip(feature_names, log_probs[class_index]),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    st.markdown(f"Your input highlighted these **key matching keywords** for **{best}**:")
                    for word, score in top_features:
                        st.markdown(f"- ✅ **{word.title()}**")
                    
                    st.info("💡 **Tip:** Adding more specific keywords or technologies you are familiar with will improve prediction accuracy!")

# -----------------------------
# FOOTER
# -----------------------------

st.divider()
st.markdown("<p style='text-align: center; color: #888; font-size: 0.9em;'>💡 <b>Example inputs:</b> <i>python machine learning, html css react, accounting finance tax tally</i></p>", unsafe_allow_html=True)