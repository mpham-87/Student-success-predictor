from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
 
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs" / "student_support_model.joblib"
st.title("Student Success Prediction System")
st.caption("Educational demonstration using a historical public dataset; not a real-school decision tool.")
if not MODEL.exists():
    st.error("No trained model found. Run: python src/train.py")
    st.stop()
model = joblib.load(MODEL)  # Load only the file your own training script created.
 
with st.form("student"):
    studytime = st.selectbox("Weekly study-time category", [1, 2, 3, 4])
    absences = st.number_input("Absences", min_value=0, max_value=100, value=3)
    failures = st.number_input("Previous failures", min_value=0, max_value=4, value=0)
    schoolsup = st.selectbox("Extra school support", ["yes", "no"])
    famsup = st.selectbox("Family educational support", ["yes", "no"])
    activities = st.selectbox("Extracurricular activities", ["yes", "no"])
    higher = st.selectbox("Intends higher education", ["yes", "no"])
    submitted = st.form_submit_button("Predict")
 
if submitted:
    example = pd.DataFrame([{
        "studytime": studytime, "absences": absences,
        "failures": failures, "schoolsup": schoolsup,
        "famsup": famsup, "activities": activities, "higher": higher,
    }])
    prediction = int(model.predict(example)[0])
    if prediction == 1:
        st.warning("Model result: may need additional academic support")
    else:
        st.success("Model result: not flagged by this demonstration")
    st.caption("Predictions can be wrong. A teacher should never use this alone for decisions.")
