import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ------------------- LOAD MODEL -------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------- LOAD ENV & LLM -------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please set your GROQ_API_KEY in a .env file!")
else:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=groq_api_key,
        temperature=0.4
    )

# ------------------- SESSION STATE -------------------
if "page" not in st.session_state:
    st.session_state.page = "registration"
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}

# ------------------- PAGE 1: OP REGISTRATION -------------------
if st.session_state.page == "registration":
    st.set_page_config(page_title="🏥 Diabetes OP Registration", layout="centered")
    st.title("Diabetes Hospital OP Registration")

    with st.form("OP_Registration"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", "")
            patient_name = st.text_input("Patient Name", "")
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        with col2:
            age = st.number_input("Age", 1, 120, 35)
            contact = st.text_input("Contact Number", "")
            if contact and len(contact) > 10:
                st.warning("Contact number cannot exceed 10 digits!")
            address = st.text_area("Address", "")
        register_submit = st.form_submit_button("✅ Register Patient")

    if register_submit:
        if not patient_id or not patient_name or not sex or len(contact) != 10:
            st.error("Please fill all required fields correctly! Contact must be 10 digits.")
        else:
            st.session_state.patient_info = {
                "Patient ID": patient_id,
                "Patient Name": patient_name,
                "Sex": sex,
                "Age": age,
                "Contact": contact,
                "Address": address
            }
            st.success(f"Patient {patient_name} registered successfully! 🎉")
            st.session_state.page = "prediction"  # move to next page automatically

# ------------------- PAGE 2: Diabetes Prediction & Doctor Report -------------------
elif st.session_state.page == "prediction":
    st.set_page_config(page_title="🏥 Diabetes Prediction & Doctor Report", layout="centered")
    st.title("🩺 Diabetes Prediction & AI Doctor Report")

    patient_info = st.session_state.patient_info

    # Display patient info in readable format
    st.subheader("Patient Details:")
    st.markdown(f"""
**Patient ID:** {patient_info['Patient ID']}  
**Patient Name:** {patient_info['Patient Name']}  
**Sex:** {patient_info['Sex']}  
**Age:** {patient_info['Age']}  
**Contact:** {patient_info['Contact']}  
**Address:** {patient_info['Address']}  
""")

    with st.form("Patient_Form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            preg = st.number_input("Pregnancies", 0, 20, 2)
            glucose = st.number_input("Glucose", 0, 300, 100)
            bp = st.number_input("Blood Pressure", 0, 150, 70)
        with col2:
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = patient_info["Age"]  # already collected in registration
            st.text_input("Age (auto-filled)", value=age, disabled=True)
        pred_submit = st.form_submit_button("🔍 Predict")

    if pred_submit:
        # Ensure feature order matches model training columns:
        # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_data = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
        prediction = model.predict(user_data)[0]

        st.subheader("📊 Prediction Result:")
        result_text = "🚨 High Risk of Diabetes Detected 😟" if prediction == 1 else "😃🎉 Low Risk / No Diabetes Detected"
        st.markdown(f"### {result_text}")

        # ------------------- CREATE DOCTOR REPORT -------------------
        patient_summary = f"""
Patient Report:
- Patient ID: {patient_info['Patient ID']}
- Patient Name: {patient_info['Patient Name']}
- Sex: {patient_info['Sex']}
- Age: {patient_info['Age']}
- Contact: {patient_info['Contact']}
- Address: {patient_info['Address']}
- Pregnancies: {preg}
- Glucose Level: {glucose}
- Blood Pressure: {bp}
- Skin Thickness: {skin}
- Insulin: {insulin}
- BMI: {bmi}
- Diabetes Pedigree Function: {dpf}
- Model Prediction: {"Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)"}
"""

        full_prompt = f"""
You are Dr. Ramesh Kumar, a Senior Diabetologist with 25 years of experience.
Prepare a detailed, structured hospital-style diagnosis report for the following patient.

Include:
1️⃣ Patient Overview  
2️⃣ Risk Assessment  
3️⃣ Possible Causes  
4️⃣ Recommended Medical Tests  
5️⃣ Lifestyle & Diet Advice  
6️⃣ Follow-up Plan  

Keep the tone professional, warm, and medically appropriate.

{patient_summary}
"""

        response = llm.invoke(full_prompt)

        st.divider()
        st.subheader("🏥 Doctor’s Report (Dr. Ramesh Kumar, Senior Diabetologist):")
        st.markdown(response.content)

        # Option to download report as text
        st.download_button(
            label="📄 Download Doctor Report",
            data=response.content,
            file_name=f"{patient_info['Patient Name']}_Diabetes_Report.txt",
            mime="text/plain"
        )

        st.success("✅ Doctor Report generated successfully!")

    st.divider()
    if st.button("🔄 Register New Patient"):
        st.session_state.page = "registration"
        st.session_state.patient_info = {}
