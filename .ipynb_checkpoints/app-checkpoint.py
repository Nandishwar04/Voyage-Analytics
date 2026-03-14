import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open(r"C:\Users\kisho\Downloads\Voyage_Analytics\Models\flight_price_model.pkl", "rb"))

# Title
st.title("Flight Price Prediction App")

st.write("Predict flight ticket prices using machine learning")

# Sidebar Inputs
st.sidebar.header("Enter Flight Details")

distance = st.sidebar.number_input("Distance (km)", min_value=100, max_value=10000)

time = st.sidebar.number_input("Travel Time (minutes)", min_value=30, max_value=1000)

flight_type = st.sidebar.selectbox(
    "Flight Type",
    ["economy", "business"]
)

agency = st.sidebar.selectbox(
    "Booking Agency",
    ["Airline", "Travel Agent"]
)

month = st.sidebar.slider("Month", 1, 12)

day = st.sidebar.slider("Day", 1, 31)

# Convert inputs to dataframe
input_data = pd.DataFrame({
    "distance": [distance],
    "time": [time],
    "month": [month],
    "day": [day]
})

# Prediction
if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.success(f"Estimated Flight Price: ${prediction[0]:.2f}")
