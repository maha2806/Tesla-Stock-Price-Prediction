import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
}

h1 {
    text-align: center;
    color: #e11d48;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.stButton > button {
    width: 100%;
    background-color: #dc2626;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #b91c1c;
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #111827;
    border-radius: 10px;
    padding: 15px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("📊 Project Information")

st.sidebar.info("""
**Tesla Stock Price Prediction**

🎓 Machine Learning Project

📈 Model Used:
- Linear Regression

📂 Dataset:
- Yahoo Finance Historical Tesla Data

🔍 Features:
- Open Price
- High Price
- Low Price
- Volume
""")

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("tesla_stock_prediction_model.pkl")

# ---------------- HEADER ---------------- #

st.markdown(
    "<h1>🚗 Tesla Stock Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("""
<div style="
background-color:#111827;
padding:20px;
border-radius:15px;
margin-bottom:20px;
">

### 📈 Project Overview

This Machine Learning application predicts Tesla's stock closing price using historical stock market indicators.

**Model Used:** Linear Regression

**Features Used:** Open Price, High Price, Low Price, Volume

**Dataset Source:** Yahoo Finance Historical Tesla Stock Data

</div>
""", unsafe_allow_html=True)

# ---------------- INPUT SECTION ---------------- #

st.subheader("📥 Enter Stock Values")

open_price = st.number_input(
    "Open Price",
    min_value=0.0,
    value=300.0
)

high_price = st.number_input(
    "High Price",
    min_value=0.0,
    value=305.0
)

low_price = st.number_input(
    "Low Price",
    min_value=0.0,
    value=295.0
)

volume = st.number_input(
    "Volume",
    min_value=0.0,
    value=50000000.0
)

# ---------------- PREDICTION ---------------- #

if st.button("🚀 Predict Closing Price"):

    features = np.array([[
        open_price,
        high_price,
        low_price,
        volume
    ]])

    prediction = model.predict(features)

    st.success("Prediction Generated Successfully!")

    st.metric(
        label="💰 Predicted Tesla Closing Price",
        value=f"${prediction[0]:.2f}"
    )

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption(
    "Built with Streamlit | Tesla Stock Prediction using Machine Learning"
)
