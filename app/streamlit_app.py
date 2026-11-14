import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "mental_health_pipeline.joblib"
metadata_path = root / "models" / "model_metadata.json"
report_dir = root / "reports"

st.set_page_config(page_title="Student Mental Health", layout="wide")
st.title("Student Mental Health")
st.info("Educational demo using synthetic data. Predictions are not clinical assessments.")

if not model_path.exists() or not metadata_path.exists():
    st.warning("Run the mental_health_pipeline DAG in Airflow to create the model first.")
    st.stop()

model = joblib.load(model_path)
metadata = json.loads(metadata_path.read_text())
results_tab, prediction_tab = st.tabs(["Model results", "Try a prediction"])

with results_tab:
    st.subheader(metadata["model"])
    metrics = metadata.get("test_metrics", {})
    if metrics:
        columns = st.columns(4)
        for column, metric in zip(columns, [
            "Accuracy", "Weighted precision", "Weighted recall", "Weighted F1"
        ]):
            column.metric(metric, f"{metrics[metric]:.1%}")
        st.caption(f"Test ROC-AUC: {metrics['Weighted ROC-AUC']:.3f}")
    else:
        st.info("Evaluation is pending. Refresh this page when the Airflow run finishes.")

    left, right = st.columns(2)
    with left:
        st.subheader("Test confusion matrix")
        image_path = report_dir / "confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path))
        else:
            st.info("The evaluation task will create this chart.")
    with right:
        st.subheader("Baseline model comparison")
        comparison_path = report_dir / "model_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path).set_index("Model")
            st.bar_chart(comparison[["F1"]])
            st.caption("Five-fold cross-validation weighted F1 on the training data.")
        else:
            st.info("The training task will create this comparison.")

with prediction_tab:
    st.write("Enter a sample student's features to try the saved model.")
    with st.form("student_form"):
        left, middle, right = st.columns(3)
        with left:
            age = st.number_input("Age", 17, 80, 22)
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
            year = st.selectbox("Year of study", [1, 2, 3, 4, 5], index=1)
            sleep = st.number_input("Sleep hours per night", 0.0, 24.0, 7.0, step=0.1)
            study = st.number_input("Study hours per week", 0.0, 168.0, 20.0, step=0.5)
        with middle:
            gpa = st.number_input("GPA (0–4)", 0.0, 4.0, 2.7, step=0.1)
            social = st.number_input("Social media hours per day", 0.0, 24.0, 5.0, step=0.1)
            screen = st.number_input("Total screen hours per day", 0.0, 24.0, 8.0, step=0.1)
            activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"], index=1)
            financial = st.slider("Financial stress", 1.0, 10.0, 5.0, step=0.1)
        with right:
            family = st.slider("Family support", 1.0, 10.0, 7.0, step=0.1)
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, step=0.5)
            job = st.selectbox("Part-time job", ["No", "Yes"])
            diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"], index=1)
            history = st.selectbox("Previous mental health history", ["No", "Yes"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        if social > screen:
            st.warning("Social media hours cannot exceed total screen hours.")
        else:
            student = pd.DataFrame([{
                "Age": age, "Gender": gender, "Year_of_Study": year,
                "Sleep_Hours": sleep, "Study_Hours_Per_Week": study, "GPA": gpa,
                "Social_Media_Hours": social, "Screen_Time_Hours": screen,
                "Physical_Activity": activity, "Financial_Stress": financial,
                "Family_Support": family, "Attendance_Rate": attendance,
                "Part_Time_Job": job, "Diet_Quality": diet,
                "Previous_Mental_Health_History": history
            }])
            student = student[metadata["feature_columns"]]
            prediction = model.predict(student)[0]
            probabilities = model.predict_proba(student)[0]
            st.success(f"Predicted category: {prediction}")
            st.bar_chart(pd.DataFrame(
                {"Model probability": probabilities}, index=model.classes_
            ).reindex(["Low", "Medium", "High"]))
            st.caption("These are model outputs for an educational example, not medical probabilities.")
