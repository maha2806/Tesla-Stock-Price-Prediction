import streamlit as st
import joblib
import numpy as np

st.title("Tesla Stock Price Prediction")

st.markdown("""
### Project Overview
This machine learning application predicts Tesla's stock closing price based on historical market indicators.

**Model Used:** Linear Regression  
**Features Used:** Open Price, High Price, Low Price, and Volume  
**Dataset Source:** Yahoo Finance Historical Tesla Stock Data
""")

st.write(
    "This application predicts Tesla stock closing prices using a Linear Regression model trained on historical stock data."
)

model = joblib.load("tesla_stock_prediction_model.pkl")

st.write("Enter stock values below:")

# Input fields
open_price = st.number_input("Open Price", value=300.0)
high_price = st.number_input("High Price", value=305.0)
low_price = st.number_input("Low Price", value=295.0)
volume = st.number_input("Volume", value=50000000.0)

if st.button("Predict Closing Price"):

    features = np.array([[
        open_price,
        high_price,
        low_price,
        volume
    ]])

    prediction = model.predict(features)

    st.success(f"Predicted Close Price: ${prediction[0]:.2f}")
