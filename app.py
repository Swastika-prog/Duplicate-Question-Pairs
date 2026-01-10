import streamlit as st
import requests

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Duplicate Question Detection",
    layout="centered"
)

st.title("Duplicate Question Detection")
st.write("Check whether two questions are duplicates.")

# -----------------------------
# Input fields
# -----------------------------
q1 = st.text_area("Question 1", height=100)
q2 = st.text_area("Question 2", height=100)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Check"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions.")
    else:
        payload = {
            "question1": q1,
            "question2": q2
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()

                prob = result["probability"]
                pred = result["duplicate"]

                if pred == 1:
                    st.success(f"Duplicate ✅ (Probability: {prob})")
                else:
                    st.error(f"Not Duplicate ❌ (Probability: {prob})")

            else:
                st.error("API Error. Check backend logs.")

        except Exception as e:
            st.error(f"Connection error: {e}")