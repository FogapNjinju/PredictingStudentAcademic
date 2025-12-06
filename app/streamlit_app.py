import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="📊",
    layout="centered"
)


# ------------------------------------------------------------
# Load trained model and preprocessing pipeline
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


st.title("📘 Student Academic Performance Prediction App")
st.write(
    "This application predicts the academic performance category of a student based on their demographics and study habits."
)

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
st.header("Student Information Input")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age_at_enrollment = st.number_input("Age at enrollment", min_value=14, max_value=100, step=1)
        Mothers_occupation = st.number_input("Mother's occupation", min_value=0, max_value=100, step=1)
        Fathers_occupation = st.number_input("Father's occupation", min_value=0, max_value=100, step=1)
        Admission_grade = st.number_input("Admission_grade", min_value=0.0, max_value=200.0, step=0.1)
        Tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", options=[0, 1])
        Previous_qualification_grade = st.number_input("Previous qualification (grade)", min_value=0.0, max_value=20.0, step=0.1)

    with col2:
        Course = st.number_input("Course", min_value=171, max_value=9119, step=1)
        Curricular_units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem (enrolled)", min_value=0, max_value=40, step=1)
        Curricular_units_1st_sem_approved = st.number_input("Curricular Units 1st Sem (approved)", min_value=0, max_value=40, step=1)
        Curricular_units_1st_sem_evaluations = st.number_input("Curricular Units 1st Sem (evaluations)", min_value=0, max_value=100, step=1)
        Curricular_units_1st_sem_grade = st.number_input("Curricular Units 1st Sem (grade)", min_value=0.0, max_value=20.0, step=0.1)

        Curricular_units_2st_sem_enrolled = st.number_input("Curricular Units 2st Sem (enrolled)", min_value=0, max_value=40, step=1)
        Curricular_units_2st_sem_approved = st.number_input("Curricular Units 2st Sem (approved)", min_value=0, max_value=40, step=1)
        Curricular_units_2st_sem_evaluations = st.number_input("Curricular Units 2st Sem (evaluations)", min_value=0, max_value=100, step=1)
        Curricular_units_2st_sem_grade = st.number_input("Curricular Units 2st Sem (grade)", min_value=0.0, max_value=20.0, step=0.1)
         

    submitted = st.form_submit_button("Predict Performance")

# ------------------------------------------------------------
# Prediction Logic
# ------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([
        {
            "Age at enrollment": Age_at_enrollment,
            "Mothers_occupation": Mothers_occupation,
            "Fathers_occupation": Fathers_occupation,
            "Admission_grade": Admission_grade,
            "Tuition fees up to date": Tuition_fees_up_to_date, 
            "Previous qualification (grade)": Previous_qualification_grade,

            "Course": Course,
            "Curricular_units_1st_sem_enrolled": Curricular_units_1st_sem_enrolled,
            "Curricular_units_1st_sem_evaluations": Curricular_units_1st_sem_evaluations,
            "Curricular_units_1st_sem_grade": Curricular_units_1st_sem_grade,
            "Curricular_units_1st_sem_approved": Curricular_units_1st_sem_approved,
            "Curricular_units_2st_sem_enrolled": Curricular_units_2st_sem_enrolled,
            "Curricular_units_2st_sem_evaluations": Curricular_units_2st_sem_evaluations,
            "Curricular_units_2st_sem_grade": Curricular_units_2st_sem_grade,
            "Curricular_units_2st_sem_approved": Curricular_units_2st_sem_approved  
            
        }
    ])

    # Preprocess and predict
    processed = pipeline.transform(input_data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed).max()

    # ------------------------------------------------------------
    # Display Results
    # ------------------------------------------------------------
    st.subheader("Prediction Result")

    st.success(f"🎯 Predicted Performance Category: **{prediction}**")
    st.write(f"Prediction Confidence: **{probability:.2f}**")

    st.info("You can modify the inputs and try again to see how changes affect prediction results.")
