import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/tesla_stock_model.pkl")

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Tesla Stock Price Prediction")
st.write("Predict Tesla's Closing Stock Price using Machine Learning")

st.subheader("Enter Stock Information")

open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=100.0)
low_price = st.number_input("Low Price", value=100.0)
adj_close = st.number_input("Adj Close Price", value=100.0)
volume = st.number_input("Volume", value=1000000.0)

price_range = st.number_input("Price Range", value=5.0)
daily_return = st.number_input("Daily Return", value=0.0)

ma_7 = st.number_input("7-Day Moving Average", value=100.0)
ma_30 = st.number_input("30-Day Moving Average", value=100.0)

if st.button("Predict Closing Price"):

    features = np.array([[
        open_price,
        high_price,
        low_price,
        adj_close,
        volume,
        price_range,
        daily_return,
        ma_7,
        ma_30
    ]])

    prediction = model.predict(features)

    st.success(
        f"Predicted Tesla Closing Price: ${prediction[0]:.2f}"
    )
