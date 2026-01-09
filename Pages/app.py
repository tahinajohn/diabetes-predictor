import streamlit as st
import joblib
import numpy as np

diabetes_model = joblib.load("../xg_diabetes_model.pkl")

st.title("Welcome to my portfolio Page")
st.write("This is the main application page.")

st.header("Diabetes Prediction")
st.write("Enter the patient data to predict diabetes:")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)


input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
]])

if st.button("Predict Diabetes"):
    prediction = diabetes_model.predict(input_data)
    st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
    proba = diabetes_model.predict_proba(input_data)
    st.write(f"Risk of being diabetic: {proba[0][1]*100:.2f}%")


st.header("About Me")
st.write("""
         



Hello! I'm Tahina Johnson, a data scientist with expertise in machine learning and web development.
I have worked on various projects involving predictive modeling, data analysis, and building interactive web applications.
Feel free to explore my portfolio and reach out if you'd like to collaborate!
""")
