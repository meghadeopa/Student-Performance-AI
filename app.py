"""
Student Performance Predictor — Streamlit Web Application
Project: End-to-End AI Framework for Student Performance Analysis
Student: Megha Deopa | PRN: 2405020011520 | MBA AI/ML July 2024

HOW TO RUN:
    1. Open terminal in this project folder
    2. Install streamlit: pip install streamlit
    3. Run: streamlit run app.py
    4. Browser opens at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model   import LinearRegression
from sklearn.preprocessing  import OneHotEncoder, StandardScaler
from sklearn.compose        import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title = "Student Score Predictor",
    page_icon  = "🎓",
    layout     = "wide"
)

# ── Title Section ──────────────────────────────────────────────
st.title("🎓 Student Math Score Predictor")
st.markdown("**MBA Final Year Project — AI/ML | OMBP 405**")
st.markdown("*Dr. D. Y. Patil Vidyapeeth, Centre for Online Learning, Pune*")
st.markdown("**Student:** Megha Deopa | PRN: 2405020011520")
st.divider()
st.markdown("Fill in the student details on the left to predict their **Mathematics Score**.")

# ── Train and Cache Model ──────────────────────────────────────
@st.cache_resource
def load_and_train():
    """Load dataset and train Linear Regression model once"""
    data = pd.read_csv('data/stud.csv')
    X    = data.drop(columns=['math_score'])
    y    = data['math_score']

    cat = X.select_dtypes(include='object').columns
    num = X.select_dtypes(exclude='object').columns

    pre = ColumnTransformer([
        ('ohe',    OneHotEncoder(handle_unknown='ignore'), cat),
        ('scaler', StandardScaler(),                       num)
    ])

    Xp = pre.fit_transform(X)
    m  = LinearRegression().fit(Xp, y)

    from sklearn.metrics import r2_score
    train_r2 = r2_score(y, m.predict(Xp)) * 100

    return m, pre, X.columns.tolist(), round(train_r2, 2)

model, preprocessor, feature_cols, train_r2 = load_and_train()

# ── Sidebar Inputs ─────────────────────────────────────────────
st.sidebar.header("📋 Enter Student Details")
st.sidebar.markdown("---")

gender = st.sidebar.selectbox(
    "Gender",
    options=["female", "male"],
    help="Student's gender"
)

race = st.sidebar.selectbox(
    "Race / Ethnicity",
    options=["group A", "group B", "group C", "group D", "group E"],
    index=2,
    help="Student's ethnic group"
)

parent_edu = st.sidebar.selectbox(
    "Parental Level of Education",
    options=["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"],
    index=2,
    help="Highest education level of student's parent"
)

lunch = st.sidebar.selectbox(
    "Lunch Type",
    options=["standard", "free/reduced"],
    help="Type of lunch the student receives"
)

test_prep = st.sidebar.selectbox(
    "Test Preparation Course",
    options=["none", "completed"],
    help="Whether the student completed the test preparation course"
)

st.sidebar.markdown("---")
reading_score = st.sidebar.slider(
    "Reading Score",
    min_value=0, max_value=100, value=70,
    help="Student's reading score (0–100)"
)

writing_score = st.sidebar.slider(
    "Writing Score",
    min_value=0, max_value=100, value=70,
    help="Student's writing score (0–100)"
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Model:** Linear Regression\n\n**Training R²:** {train_r2}%\n\n**Dataset:** 1000 students")

# ── Predict Button ─────────────────────────────────────────────
predict_btn = st.button(
    "🔮 Predict Math Score",
    type="primary",
    use_container_width=True
)

if predict_btn:
    # Prepare input
    input_data = pd.DataFrame(
        [[gender, race, parent_edu, lunch, test_prep, reading_score, writing_score]],
        columns=feature_cols
    )

    # Predict
    pred = float(np.clip(model.predict(preprocessor.transform(input_data))[0], 0, 100))
    avg  = (pred + reading_score + writing_score) / 3

    # ── Results ────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("📐 Predicted Math Score", f"{pred:.1f} / 100")
    col2.metric("📈 Estimated Average",    f"{avg:.1f} / 100")
    col3.metric("📖 Reading Score",        f"{reading_score} / 100")

    # Grade and colour feedback
    st.markdown("### 🏅 Grade Assessment")
    if pred >= 90:
        st.success(f"🏆 **Grade A — Outstanding!** Score: {pred:.1f}")
        st.balloons()
    elif pred >= 75:
        st.success(f"🌟 **Grade B — Good Performance!** Score: {pred:.1f}")
    elif pred >= 60:
        st.info(f"✅ **Grade C — Average Performance.** Score: {pred:.1f}")
    elif pred >= 40:
        st.warning(f"⚠️ **Grade D — Below Average. Needs Improvement.** Score: {pred:.1f}")
    else:
        st.error(f"❌ **Grade F — At Risk of Failing. Immediate Support Needed.** Score: {pred:.1f}")

    # ── Personalised Recommendations ──────────────────────────
    st.divider()
    st.subheader("💡 Personalised Improvement Recommendations")

    recs = []
    if test_prep == "none":
        recs.append(("HIGH",   "📚 Enrol in the Test Preparation Course",
                     "Research shows students who complete test prep score 5–8 points higher on average."))
    if lunch == "free/reduced":
        recs.append(("HIGH",   "🍱 Apply for Standard Meal Program",
                     "Students with standard lunch score 8–10 points higher. Nutritional support matters."))
    if reading_score < 60:
        recs.append(("MEDIUM", "📖 Join a Reading Improvement Program",
                     "Reading score has a 0.82 correlation with math. Better reading → better math."))
    if writing_score < 60:
        recs.append(("MEDIUM", "✍️ Attend Writing Support Sessions",
                     "Reading and writing have r=0.95 correlation — improving both together is effective."))
    if race in ["group A", "group B"]:
        recs.append(("MEDIUM", "🎯 Access Targeted Academic Support",
                     "Group A and B students benefit most from additional tutoring and mentoring."))
    if parent_edu in ["some high school", "high school"]:
        recs.append(("LOW",    "👨‍👩‍👧 Encourage Parental Engagement",
                     "Students with more educated parents tend to score higher. Community programs help."))

    if pred >= 75:
        recs.append(("INFO",   "🏆 Strong performer!",
                     "Consider advanced coursework or scholarship opportunities to maximise potential."))
    elif pred >= 50:
        recs.append(("INFO",   "📈 Room to improve",
                     "Consistent study habits and test prep completion will help reach higher grades."))
    else:
        recs.append(("URGENT", "⚠️ At-Risk student",
                     "Immediate academic intervention and counselling is strongly recommended."))

    for priority, title, detail in recs:
        if priority == "HIGH" or priority == "URGENT":
            st.error(f"**[{priority}] {title}**\n\n{detail}")
        elif priority == "MEDIUM":
            st.warning(f"**[{priority}] {title}**\n\n{detail}")
        else:
            st.info(f"**[{priority}] {title}**\n\n{detail}")

# ── Model Information ──────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About This Application"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Model Details:**
- Algorithm: Linear Regression
- Training R² Score: ~88%
- Dataset: 1000 students × 8 variables
- Source: Kaggle — Students Performance in Exams
        """)
    with col2:
        st.markdown("""
**Key Findings:**
- Lunch type is the strongest predictor (+8–10 pts)
- Test prep completion improves all subjects
- Females score higher overall; males slightly better in Math
- Group E scores highest; Group A scores lowest
        """)

st.caption("Megha Deopa | PRN: 2405020011520 | MBA Online AI/ML July 2024 | OMBP 405 | Dr. D.Y. Patil Vidyapeeth COL")
