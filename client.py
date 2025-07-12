
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained XGBoost model and feature list
model = joblib.load("xgboost_salary_predictor.pkl")
feature_columns = joblib.load("model_features.pkl")

# Page configuration
st.set_page_config(page_title="💼 DS Salary Predictor", page_icon="💸", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f7f7f7;
    }
    h1 {
        color: #4B8BBE;
        text-align: center;
        font-size: 3em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stSelectbox>div>div>div>div {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💼 Data Science Salary Predictor")
st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
        Enter your job and company details to estimate your expected salary 💰.<br>
        Powered by a trained XGBoost regression model.
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Input fields with icons
col1, col2 = st.columns(2)
with col1:
    experience = st.selectbox("👨‍💻 Experience Level", ['EN', 'MI', 'SE', 'EX'])
    employment = st.selectbox("💼 Employment Type", ['FT', 'PT', 'CT', 'FL'])
    company_size = st.selectbox("🏢 Company Size", ['S', 'M', 'L'])

with col2:
    job_title = st.selectbox("🧠 Job Title", ['Data Scientist', 'ML Engineer', 'Data Analyst', 'Other'])
    company_continent = st.selectbox("🌍 Company Continent", ['North America', 'Europe', 'Asia', 'South America', 'Oceania', 'Other'])
    remote_flag = st.selectbox("🧑‍💻 Remote Work", ['Remote', 'Not Remote'])

year = st.slider("📅 Work Year", 2020, 2023, 2023)

# Prepare input data for model
input_dict = {
    'work_year': year,
    'remote_flag': 1 if remote_flag == 'Remote' else 0
}

# One-hot encoding based on training features
for col in feature_columns:
    if col.startswith('experience_level_'):
        input_dict[col] = 1 if col.endswith(experience) else 0
    elif col.startswith('employment_type_'):
        input_dict[col] = 1 if col.endswith(employment) else 0
    elif col.startswith('job_title_'):
        input_dict[col] = 1 if col.endswith(job_title) else 0
    elif col.startswith('company_size_'):
        input_dict[col] = 1 if col.endswith(company_size) else 0
    elif col.startswith('company_continent_'):
        input_dict[col] = 1 if col.endswith(company_continent) else 0

# Ensure all features used during training are present
for col in feature_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Create DataFrame for prediction
X_input = pd.DataFrame([input_dict])

# Predict button
if st.button("🚀 Predict Salary"):
    prediction = model.predict(X_input)[0]
    st.success(f"💰 Estimated Salary (USD): ${prediction:,.2f}")
    st.balloons()
