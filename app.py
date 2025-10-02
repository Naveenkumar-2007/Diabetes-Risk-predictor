import streamlit as st
from crew import diabetes_agent
import pickle
from crewai import Crew, Task


st.set_page_config(page_title="Diabetes Risk Prediction ", page_icon="🩺",layout='centered')

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter patient data to predict diabetes risk")
with open('model.pkl',"rb") as f:
    model=pickle.load(f)

    with st.form("Patient Form"):
        col1,col2,col3=st.columns(3)
        with col1:
            preg = st.number_input("Pregnancies", min_value=0, max_value=200,value=2)
            glucose = st.number_input("Glucose", min_value=0, max_value=300,value=1)
            bp = st.number_input("Blood Pressure",min_value= 0, max_value=150,value=0)
        with col2:
            skin = st.number_input("Skin Thickness", min_value=0, max_value=100,value=5)
            insulin = st.number_input("Insulin",min_value=0,max_value= 900,value=0)
            bmi = st.number_input("BMI", min_value=0.0,max_value= 70.0,value=22.5)
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function",min_value= 0.0, max_value=3.0,value=0.031)
            age = st.number_input("Age", min_value=1, max_value=120,value=18)
        submit=st.form_submit_button("Predict")

           


if submit:
    user_data = {
    "Pregnancies": preg,
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}
    
    user=[[preg,glucose,bp,skin,insulin,bmi,dpf,age]]
    # Call the first tool of the agent
    response1=model.predict(user)
    st.subheader("🔎 Diabetes Assistant Response")
    st.subheader("🚨⚠️ **High Risk of Diabetes Predicted** 😟" if response1[0] == 1 else "✅🟢 **No Diabetes Predicted** 😃🎉")
    
    # Create a task for the agent
    task = Task(
        description=f"""Analyze this patient's health data using AI and machine learning: {user_data}. 
        
        First use the diabetes prediction tool to get the risk assessment. Then provide:
        1. Clear explanation of the diabetes risk prediction
        2. Personalized health recommendations based on their specific data
        3. Lifestyle advice and prevention strategies
        4. Follow-up recommendations
        
        Format the response in a patient-friendly way and mention that this analysis was powered by CrewAI technology.""",
        expected_output="A comprehensive patient report with AI-powered diabetes risk assessment and personalized health recommendations",
        agent=diabetes_agent
    )
    
    # Create a crew and execute the task
    crew = Crew(
        agents=[diabetes_agent],
        tasks=[task],
        verbose=False  # Reduce verbose output to avoid JSON clutter
    )
    
    