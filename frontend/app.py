import streamlit as st
import requests

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Duplicate Question Detector",
    layout="centered"
)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🧠 Duplicate Question Detector")
st.write("Enter two questions to check if they mean the same thing.")

q1 = st.text_area("Question 1", height=100)
q2 = st.text_area("Question 2", height=100)

model_choice = st.radio(
    "Choose model",
    ["SVM (Fast)", "XGBoost (With Probability)"]
)

# --------------------------------------------------
# ACTION
# --------------------------------------------------
if st.button("Predict"):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")
    else:
        model = "svm" if model_choice.startswith("SVM") else "xgb"

        payload = {
    "question1": q1,
    "question2": q2,
    "model": model
}


        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code != 200:
                st.error(f"Backend error: {response.text}")
            else:
                result = response.json()
                st.subheader("Result:")
                st.write(result)


                if result["is_duplicate"] == 1:
                    if model == "xgb":
                        st.success(
                            f"✔ Duplicate ({result['confidence']:.2f} confidence)"
                        )
                    else:
                        st.success("✔ Duplicate")
                else:
                    if model == "xgb":
                        st.error(
                            f"✘ Not Duplicate ({result['confidence']:.2f} confidence)"
                        )
                    else:
                        st.error("✘ Not Duplicate")

        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
