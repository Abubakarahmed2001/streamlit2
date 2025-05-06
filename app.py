import streamlit as st
import pandas as pd
import numpy as np
import joblib
df = pd.read_csv("Financial_inclusion_dataset.csv")
# Load model + expected columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Title
st.title("Financial Inclusion Predictor")
st.markdown("Enter demographic features to predict bank account ownership.")

# --- Input fields based on your original features ---
age = st.number_input("Age of respondent", min_value=18, max_value=100, value=30)
household_size = st.number_input("Household size", min_value=1, max_value=20, value=3)

country = st.selectbox("Country", df['country'].unique())
location_type = st.selectbox("Location type", df['location_type'].unique())
cellphone_access = st.selectbox("Cellphone access", df['cellphone_access'].unique())
gender = st.selectbox("Gender", df['gender_of_respondent'].unique())
relationship = st.selectbox("Relationship with head", df['relationship_with_head'].unique())
marital = st.selectbox("Marital status", df['marital_status'].unique())
education = st.selectbox("Education level", df['education_level'].unique())
job = st.selectbox("Job type", df['job_type'].unique())

# Build a DataFrame of zeros with all model columns
input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

# Fill numeric features
input_df['age_of_respondent'] = age
input_df['household_size'] = household_size

# Set the appropriate dummy variable to 1 for each categorical
def set_dummy(col_prefix, value):
    col_name = f"{col_prefix}_{value}"
    if col_name in input_df.columns:
        input_df[col_name] = 1

set_dummy('country', country)
set_dummy('location_type', location_type)
set_dummy('cellphone_access', cellphone_access)
set_dummy('gender_of_respondent', gender)
set_dummy('relationship_with_head', relationship)
set_dummy('marital_status', marital)
set_dummy('education_level', education)
set_dummy('job_type', job)

# Prediction button
if st.button("Predict"):
    proba = model.predict_proba(input_df)[0,1]
    pred = model.predict(input_df)[0]
    st.write(f"**Probability of having a bank account:** {proba:.2%}")
    st.write("**Prediction:**", "Has bank account" if pred == 1 else "No bank account")
