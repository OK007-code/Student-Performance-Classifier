import streamlit as st
import pandas as pd
import joblib


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------------
# Load Model
# -----------------------------------

model = joblib.load(
    "models/student_performance_model.pkl"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("🎓 Student Performance Classifier")

st.write(
    "Predict student performance as High, Average, or Low "
    "using academic and related factors."
)


# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_student(student_data):

    prediction = model.predict(student_data)[0]

    probabilities = model.predict_proba(student_data)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# -----------------------------------
# Improvement Suggestions
# -----------------------------------

def get_suggestions(data):

    suggestions = []

    if data["Hours_Studied"].iloc[0] < 15:
        suggestions.append(
            "Increase study time and follow a consistent study schedule."
        )

    if data["Attendance"].iloc[0] < 75:
        suggestions.append(
            "Improve class attendance to avoid missing important lessons."
        )

    if data["Previous_Scores"].iloc[0] < 60:
        suggestions.append(
            "Revise weak academic topics and strengthen previous concepts."
        )

    if data["Sleep_Hours"].iloc[0] < 6:
        suggestions.append(
            "Aim for 6–8 hours of sleep for better concentration."
        )

    if data["Tutoring_Sessions"].iloc[0] == 0:
        suggestions.append(
            "Consider tutoring or additional academic support."
        )

    if data["Physical_Activity"].iloc[0] < 3:
        suggestions.append(
            "Include regular physical activity in your routine."
        )

    if data["Motivation_Level"].iloc[0] == "Low":
        suggestions.append(
            "Set small academic goals and maintain a regular study routine."
        )

    if data["Access_to_Resources"].iloc[0] == "Low":
        suggestions.append(
            "Seek additional learning resources or academic support."
        )

    if data["Parental_Involvement"].iloc[0] == "Low":
        suggestions.append(
            "Encourage greater academic support and communication at home."
        )

    if data["Peer_Influence"].iloc[0] == "Negative":
        suggestions.append(
            "Build a supportive study environment and spend more time with positive academic peers."
        )

    if not suggestions:
        suggestions.append(
            "No major weak factors detected. Continue maintaining consistent academic habits."
        )

    return suggestions

# -----------------------------------
# Tabs
# -----------------------------------

tab1, tab2 = st.tabs([
    "👤 Individual Prediction",
    "📂 CSV Batch Prediction"
])


# ===================================
# INDIVIDUAL PREDICTION
# ===================================

with tab1:

    st.subheader("Enter Student Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        hours = st.number_input(
            "Hours Studied",
            min_value=0,
            max_value=50,
            value=20
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=80
        )

        parental = st.selectbox(
            "Parental Involvement",
            ["Low", "Medium", "High"]
        )

        resources = st.selectbox(
            "Access to Resources",
            ["Low", "Medium", "High"]
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["No", "Yes"]
        )

        sleep = st.number_input(
            "Sleep Hours",
            min_value=0,
            max_value=12,
            value=7
        )

        previous = st.number_input(
            "Previous Scores",
            min_value=0,
            max_value=100,
            value=70
        )

    with col2:

        motivation = st.selectbox(
            "Motivation Level",
            ["Low", "Medium", "High"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["No", "Yes"]
        )

        tutoring = st.number_input(
            "Tutoring Sessions",
            min_value=0,
            max_value=10,
            value=2
        )

        income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"]
        )

        teacher = st.selectbox(
            "Teacher Quality",
            ["Low", "Medium", "High"]
        )

        school = st.selectbox(
            "School Type",
            ["Public", "Private"]
        )

        peer = st.selectbox(
            "Peer Influence",
            ["Negative", "Neutral", "Positive"]
        )

    with col3:

        physical = st.number_input(
            "Physical Activity (hours/week)",
            min_value=0,
            max_value=20,
            value=3
        )

        disability = st.selectbox(
            "Learning Disabilities",
            ["No", "Yes"]
        )

        education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"]
        )

        distance = st.selectbox(
            "Distance from Home",
            ["Near", "Moderate", "Far"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    if st.button("🔍 Predict Performance"):

        student = pd.DataFrame([{
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Parental_Involvement": parental,
            "Access_to_Resources": resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep,
            "Previous_Scores": previous,
            "Motivation_Level": motivation,
            "Internet_Access": internet,
            "Tutoring_Sessions": tutoring,
            "Family_Income": income,
            "Teacher_Quality": teacher,
            "School_Type": school,
            "Peer_Influence": peer,
            "Physical_Activity": physical,
            "Learning_Disabilities": disability,
            "Parental_Education_Level": education,
            "Distance_from_Home": distance,
            "Gender": gender
        }])

        prediction, confidence = predict_student(student)

        st.subheader("Prediction Result")

        if prediction == "High":
            st.success(f"Performance: {prediction}")

        elif prediction == "Average":
            st.warning(f"Performance: {prediction}")

        else:
            st.error(f"Performance: {prediction}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # At-risk flag
        if prediction == "Low":
            st.error("⚠️ At-Risk Student: Immediate academic support is recommended.")

        elif prediction == "Average":
            st.warning("⚠️ Student may benefit from additional academic support.")

        else:
            st.success("✅ Student is currently not flagged as at-risk.")

        # Suggestions
        st.subheader("💡 Improvement Suggestions")

        suggestions = get_suggestions(student)

        for suggestion in suggestions:
            st.write("•", suggestion)


# ===================================
# CSV BATCH PREDICTION
# ===================================

with tab2:

    st.subheader("Upload Student CSV")

    st.write(
        "Upload a CSV containing the same student input columns "
        "used by the model."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file:

        batch_data = pd.read_csv(uploaded_file)

        # Remove target columns if present
        batch_data = batch_data.drop(
            columns=["Exam_Score", "Performance"],
            errors="ignore"
        )

        predictions = model.predict(batch_data)
        probabilities = model.predict_proba(batch_data)

        batch_data["Predicted_Performance"] = predictions

        batch_data["Confidence"] = (
            probabilities.max(axis=1) * 100
        ).round(2)

        batch_data["At_Risk"] = (
            batch_data["Predicted_Performance"] == "Low"
        )

        st.success(
            f"{len(batch_data)} students processed successfully."
        )

        st.dataframe(batch_data)

        # Download results
        csv = batch_data.to_csv(index=False)

        st.download_button(
            "⬇️ Download Predictions",
            csv,
            "student_predictions.csv",
            "text/csv"
        )