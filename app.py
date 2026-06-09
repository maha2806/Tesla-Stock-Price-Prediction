import streamlit as st
import joblib
import numpy as np

st.title("Tesla Stock Price Prediction")

model = joblib.load("tesla_stock_prediction_model.pkl")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
volume = st.number_input("Volume")

if st.button("Predict Closing Price"):

    features = np.array([[
        open_price,
        high_price,
        low_price,
        volume
    ]])

    prediction = model.predict(features)

    st.success(f"Predicted Close Price: ${prediction[0]:.2f}")
