import streamlit as st
import pickle
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Predictor", page_icon="💰")

# --- LOAD PICKLE FILES ---
def load_assets():
    with open('poly_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('poly_transformer.pkl', 'rb') as f:
        poly_transformer = pickle.load(f)
    return model, poly_transformer

try:
    model, poly_transformer = load_assets()
except FileNotFoundError:
    st.error("Error: .pkl files not found. Please run the saving script first!")

# --- APP UI ---
st.title("💰 Position Salary Predictor")
st.write("This app uses **Polynomial Regression (Degree 4)** to predict salaries based on organization levels.")

# User Input
level = st.slider("Select Position Level:", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

if st.button("Predict Salary"):
    # 1. Convert input to 2D array
    level_arr = np.array([[level]])
    
    # 2. Transform the level into polynomial terms
    level_poly = poly_transformer.transform(level_arr)
    
    # 3. Predict
    prediction = model.predict(level_poly)
    
    # 4. Display Result
    st.success(f"### The predicted salary for Level {level} is: ${prediction[0]:,.2f}")
    
    # Fun visualization
    st.metric(label="Estimated Annual Pay", value=f"${prediction[0]:,.2f}")

st.divider()
st.info("Level 1 = Junior | Level 5 = Manager | Level 10 = CEO")
