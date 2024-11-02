# Diabetes Readmission Prediction App
# -----------------------------------
# Instructions to Run the App:
# 1. Ensure you have Python installed (Python 3.7 or later recommended).
# 2. Install required packages by running:
#    pip install -r requirements.txt
#    (If you don’t have a requirements.txt, manually install key libraries:
#     pip install streamlit pandas numpy joblib scikit-learn xgboost)
# 3. Start the app by running:
#    streamlit run app.py
# 4. Open the local URL provided in the terminal (usually http://localhost:8501).

# This app allows you to input patient data and predict the likelihood of early readmission.
# Additionally, it shows the relationship between key features (like age and number of medications)
# and readmission rates based on the entire dataset.


import streamlit as st
import pandas as pd
import joblib  # For loading the model and scaler
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Load the trained model and scaler
try:
    model = joblib.load('retrained_model_important_features.pkl')  # Ensure this path is correct
    scaler = joblib.load('scaler_important_features.pkl')        # Ensure this path is correct
except FileNotFoundError as e:
    st.write(f"Error loading model or scaler: {e}")
    scaler = StandardScaler()  # Initialize an empty scaler if loading fails

# Page title
st.title("Diabetes Readmission Prediction App")
st.write("Enter patient data to predict the likelihood of early readmission.")

# Define the main 7 features for user input
input_data = {
    'age': st.number_input("Age (Years)", min_value=1, max_value=100, value=50),
    'time_in_hospital': st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=5),
    'num_lab_procedures': st.number_input("Number of Lab Procedures", min_value=1, value=40),
    'num_medications': st.number_input("Number of Medications", min_value=1, value=10),
    'number_emergency': st.number_input("Number of Emergency Visits", min_value=0, value=0),
    'number_inpatient': st.number_input("Number of Inpatient Visits", min_value=0, value=0),
    'number_diagnoses': st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=5),
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict"):
    try:
        # Preprocess input (e.g., scaling)
        try:
            input_df_scaled = scaler.transform(input_df)
        except NotFittedError:
            scaler.fit(input_df)  # Fit scaler if not loaded
            input_df_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_df_scaled)
        prediction_proba = model.predict_proba(input_df_scaled)

        # Display prediction
        if prediction[0] == 1:
            st.write("### Prediction: High Risk of Early Readmission")
        else:
            st.write("### Prediction: Low Risk of Early Readmission")
        
        st.write(f"Prediction Probability: {prediction_proba[0][1]*100:.2f}% chance of readmission within 30 days.")

    except Exception as e:
        st.write(f"An error occurred during prediction: {e}")

# Section for analysis of specific features and their impact on readmission rates
st.write("## Analysis of Features and Their Relationship with Early Readmission")

# Load dataset
try:
    diabetic_data = pd.read_csv("PRJ-002/diabetic_data.csv")  # Update path if necessary
except FileNotFoundError:
    st.write("Dataset not found. Please ensure the dataset is in the correct location.")
else:
    # Convert categorical 'age' ranges to numeric midpoints
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }
    diabetic_data['age'] = diabetic_data['age'].map(age_mapping)

    # Ensure 'readmitted' is binary
    diabetic_data['readmitted'] = diabetic_data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # Analysis for each key feature

    # Age vs. Readmission Rate
    st.write("### Age and Readmission Rate")
    age_readmission = diabetic_data.groupby('age')['readmitted'].mean()
    st.line_chart(age_readmission)
    st.write("This chart shows the relationship between age and the likelihood of early readmission.")

    # Number of Medications vs. Readmission Rate
    st.write("### Number of Medications and Readmission Rate")
    # Grouping medications into ranges for better visualization
    diabetic_data['medication_range'] = pd.cut(diabetic_data['num_medications'], bins=[0, 5, 10, 15, 20, 50], 
                                               labels=['0-5', '6-10', '11-15', '16-20', '20+'])
    medication_readmission = diabetic_data.groupby('medication_range')['readmitted'].mean()
    st.bar_chart(medication_readmission)
    st.write("Higher numbers of medications are associated with an increased likelihood of early readmission.")

    # Number of Diagnoses vs. Readmission Rate
    st.write("### Number of Diagnoses and Readmission Rate")
    diagnosis_readmission = diabetic_data.groupby('number_diagnoses')['readmitted'].mean()
    st.line_chart(diagnosis_readmission)
    st.write("This chart shows how the number of diagnoses impacts the readmission rate.")

    # Time in Hospital vs. Readmission Rate
    st.write("### Time in Hospital and Readmission Rate")
    time_in_hospital_readmission = diabetic_data.groupby('time_in_hospital')['readmitted'].mean()
    st.line_chart(time_in_hospital_readmission)
    st.write("Longer stays in the hospital may be correlated with a higher probability of readmission.")
