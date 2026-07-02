import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below to predict the house price.")

# Load the trained model
try:
    with open("model_pickle", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User inputs
area = st.number_input(
    "Area (sq.ft)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=10
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Prediction button
if st.button("Predict Price"):
    data = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(data)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")

st.markdown("---")
st.write("Developed using Streamlit & Scikit-learn")
