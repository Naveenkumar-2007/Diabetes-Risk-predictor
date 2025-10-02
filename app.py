import streamlit as st

import pickle



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
    
    