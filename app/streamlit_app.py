import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Streamlit UI Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Load Model Artifacts
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.joblib"
PIPELINE_PATH = BASE_DIR / "outputs" / "models" / "best_pipeline.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

model, pipeline = load_artifacts()

# ------------------------------------------------------------
# Title + Description
# ------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2C3E50;'>📘 Student Academic Performance Predictor</h1>
    <p style='text-align:center; font-size:18px;'>
        Predict the academic performance category of a student based on key academic and demographic factors.
    </p>
    <hr style="border:1px solid #bbb;">
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Input Form
# ------------------------------------------------------------
st.header("📝 Student Information")

with st.form("prediction_form"):
    st.subheader("Demographic & Background Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        Age_at_enrollment = st.number_input("Age at Enrollment", 14, 100, 18)
        Mothers_occupation = st.number_input("Mother's Occupation", 0, 100)
        Fathers_occupation = st.number_input("Father's Occupation", 0, 100)

    with col2:
        Admission_grade = st.number_input("Admission Grade", 0.0, 200.0)
        Tuition_fees_up_to_date = st.selectbox("Tuition Fees Up-to-Date?", [0, 1])
        Previous_qualification_grade = st.number_input("Previous Qualification Grade", 0.0, 20.0)

    with col3:
        Course = st.number_input("Course ID", 171, 9119)
        Curricular_units_1st_sem_enrolled = st.number_input("1st Sem Units Enrolled", 0, 40)
        Curricular_units_1st_sem_approved = st.number_input("1st Sem Units Approved", 0, 40)

    st.markdown("---")
    st.subheader("🎓 Academic Performance Inputs")

    col4, col5 = st.columns(2)

    with col4:
        Curricular_units_1st_sem_evaluations = st.number_input("1st Sem Evaluations", 0, 100)
        Curricular_units_1st_sem_grade = st.number_input("1st Sem Grade", 0.0, 20.0)
        Curricular_units_2nd_sem_enrolled = st.number_input("2nd Sem Units Enrolled", 0, 40)

    with col5:
        Curricular_units_2nd_sem_approved = st.number_input("2nd Sem Units Approved", 0, 40)
        Curricular_units_2nd_sem_evaluations = st.number_input("2nd Sem Evaluations", 0, 100)
        Curricular_units_2st_sem_grade = st.number_input("2nd Sem Grade", 0.0, 20.0)

    submitted = st.form_submit_button("🔍 Predict Performance", use_container_width=True)

# ------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Curricular_units_2nd_sem_approved": Curricular_units_2nd_sem_approved,
        "Tuition_fees_up_to_date": Tuition_fees_up_to_date,
        "Curricular_units_2st_sem_grade": Curricular_units_2st_sem_grade,
        "Admission_grade": Admission_grade,
        "Previous_qualification_grade": Previous_qualification_grade,
        "Curricular_units_1st_sem_grade": Curricular_units_1st_sem_grade,
        "Curricular_units_2nd_sem_enrolled": Curricular_units_2nd_sem_enrolled,
        "Course": Course,
        "Age_at_enrollment": Age_at_enrollment,
        "Curricular_units_1st_sem_evaluations": Curricular_units_1st_sem_evaluations,
        "Curricular_units_2nd_sem_evaluations": Curricular_units_2nd_sem_evaluations,
        "Curricular_units_1st_sem_approved": Curricular_units_1st_sem_approved,
        "Curricular_units_1st_sem_enrolled": Curricular_units_1st_sem_enrolled,
        "Fathers_occupation": Fathers_occupation,
        "Mothers_occupation": Mothers_occupation        
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()

    label_map = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}
    prediction_label = label_map.get(prediction, "Unknown")

    # ------------------------------------------------------------
    # Display Results
    # ------------------------------------------------------------
    st.markdown("---")
    st.header("📊 Prediction Results")

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            label="Predicted Category",
            value=prediction_label,
        )

    with colB:
        st.metric(
            label="Confidence Score",
            value=f"{probability:.2f}"
        )

    st.success(
        f"🎯 The student is predicted to **{prediction_label}** with a confidence of **{probability:.2f}**."
    )

    st.info("Modify the inputs above and try again to explore different scenarios.")
