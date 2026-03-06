import streamlit as st
import requests


st.title("Gender Classification System")

st.write("Predict gender using name and age")


# Input fields
name = st.text_input("Enter Name")

age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)


# Predict button
if st.button("Predict Gender"):

    url = "http://127.0.0.1:5000/predict_gender"

    data = {
        "name": name,
        "age": age
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:

        result = response.json()

        st.success(f"Predicted Gender: {result['predicted_gender']}")

    else:

        st.error("API Error. Check if Flask server is running.")