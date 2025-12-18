import streamlit as st
import pickle
import numpy as np

# Set page title
st.set_page_config(page_title="Salary Predictor")

st.title("Salary Prediction App")
st.write("This app predicts salary based on Position Level using the Polynomial Regression model from the notebook.")

# Load the saved model and the polynomial transformer
try:
    model = pickle.load(open('poly_reg.pkl', 'rb'))
    poly = pickle.load(open('poly_transformer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'poly_reg.pkl' and 'poly_transformer.pkl' are in the same folder as this script.")
    st.stop()

# User Input
level = st.number_input("Enter Position Level (e.g., 1 to 10):", min_value=1.0, max_value=12.0, value=6.5, step=0.1)

if st.button("Predict Salary"):
    # 1. Reshape input to 2D array
    val = np.array([[level]])
    
    # 2. Transform the input using the loaded PolynomialFeatures object
    val_poly = poly.transform(val)
    
    # 3. Make prediction
    prediction = model.predict(val_poly)
    
    # 4. Display Result
    st.success(f"The predicted salary for level {level} is: **${prediction[0]:,.2f}**")

# Optional: Add information about the dataset
st.info("""
**Reference Levels:**
1: Business Analyst, 
4: Manager, 
5: Country Manager, 
8: Region Manager, 
10: CEO
""")
