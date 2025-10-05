import streamlit as st
import pandas as pd
import joblib

# App Title
st.title(" Stroke Prediction ")
st.markdown("Predict the likelihood of a stroke based on health and lifestyle information.")

# Load model and preprocessing tools
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')
l_gen = joblib.load('l_gen.pkl')
l_evrmd = joblib.load('l_evrmd.pkl')
l_restype = joblib.load('l_restype.pkl')
l_worktype = joblib.load('l_worktype.pkl')
l_smoking = joblib.load('l_smoking.pkl')

# User input section
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
age = st.number_input("Age")
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Predict button
if st.button("Predict Stroke Risk"):
    # Create DataFrame from input
    input_df = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }])

    # Encode categorical values
    input_df['gender'] = l_gen.transform(input_df['gender'])
    input_df['ever_married'] = l_evrmd.transform(input_df['ever_married'])
    input_df['Residence_type'] = l_restype.transform(input_df['Residence_type'])
    input_df['work_type'] = l_worktype.transform(input_df['work_type'])
    input_df['smoking_status'] = l_smoking.transform(input_df['smoking_status'])

    # Scale numerical values
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    

    # Display result
    if prediction == 1:
        st.error("The person is likely to have a stroke")
    else:
        st.success(" The person is not likely to have a stroke")