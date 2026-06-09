import streamlit as st
import joblib
import numpy as np

st.title("Tesla Stock Price Prediction")

model = joblib.load("tesla_stock_prediction_model.pkl")

st.write("Enter stock values")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
adj_close = st.number_input("Adj Close")
volume = st.number_input("Volume")

price_range = high_price - low_price

daily_return = st.number_input("Daily Return")
ma7 = st.number_input("MA 7")
ma30 = st.number_input("MA 30")

if st.button("Predict Closing Price"):

    features = np.array([[
        open_price,
        high_price,
        low_price,
        adj_close,
        volume,
        price_range,
        daily_return,
        ma7,
        ma30
    ]])

    prediction = model.predict(features)

    st.success(f"Predicted Close Price: ${prediction[0]:.2f}")
