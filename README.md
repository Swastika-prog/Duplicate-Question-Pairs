# Duplicate Question Pair Detection
A machine learning system that detects whether two questions are semantically duplicate, implemented using a hybrid feature pipeline and deployed via a FastAPI backend with a Streamlit frontend.
---
## Problem Statement
Online Q&A platforms often contain multiple questions that ask the same thing using different wording.  
Identifying duplicate questions helps reduce redundancy, improve answer quality, and enhance user experience.

This project aims to automatically determine whether two questions convey the same meaning.
---
## Architecture
The project follows a clean, production-style separation of concerns:

- Model training and experimentation are done in notebooks
- Inference and feature recreation are handled by a FastAPI backend
- User interaction is provided through a Streamlit frontend
- Trained models are stored as serialized artifacts

High-level flow:
User → Streamlit Frontend → FastAPI Backend → ML Models

---

## Project Structure
```
DUPLICATE-QUESTION/
│
├── artifacts/ # Trained ML models & vectorizers
│ ├── tfidf_q1.pkl
│ ├── tfidf_q2.pkl
│ ├── svm_model.pkl
│ └── xgb_model.pkl
│
├── backend/ # Inference service
│ └── app.py # FastAPI application
│
├── frontend/ # User interface
│ └── app.py # Streamlit application
│
├── data/ # Dataset
│ └── train.csv
│
├── notebooks/ # Experiments & training
│ └── model3.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```
---
## Feature Engineering & Models
### Feature Engineering
The model uses a hybrid feature representation that combines semantic and lexical similarity:

- TF-IDF vectors for Question 1
- TF-IDF vectors for Question 2
- Lexical overlap features
  - common word count
  - word overlap ratio
  - length difference
- Fuzzy string similarity metrics
  - QRatio
  - Partial Ratio
  - Token Sort Ratio
  - Token Set Ratio
- Cosine similarity between TF-IDF vectors

### Models Used
- Linear SVM (fast and strong baseline)
- XGBoost Classifier (with probability estimates)

---

## API Contract

### Endpoint
```
POST /predict
```
### Request Body
```
{
  "question1": "Which city is the capital of France?",
  "question2": "What is the capital of France?",
  "model": "svm"
}
```
Response (SVM)
```
{
  "model": "SVM",
  "is_duplicate": 1
}
```
Response (XGBoost)
```
{
  "model": "XGBoost",
  "is_duplicate": 1,
  "confidence": 0.94
}
```
How to Run
1. Install dependencies
```
pip install -r requirements.txt
```
2. Start the backend (FastAPI)
```
uvicorn backend.app:app --reload
```
API documentation will be available at:
```
http://127.0.0.1:8000/docs
```
3. Start the frontend (Streamlit)
```
streamlit run frontend/app.py
```
Example Output

Input:
```
Which city is the capital of France?
What is the capital of France?
```
Output:
```
Duplicate
```
---

## Features

- Duplicate question detection using semantic similarity
- Hybrid feature engineering pipeline
- SVM for fast inference
- XGBoost with probability confidence
- RESTful FastAPI backend
- Interactive Streamlit UI
- Modular and scalable architecture

---

## Feature Engineering & Models

### Feature Engineering
- TF-IDF vectors for both questions
- Lexical overlap features
- Fuzzy string similarity metrics
- Cosine similarity between TF-IDF vectors

### Models
- Linear SVM
- XGBoost Classifier

---

## Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit

---

## Future Improvements
- Dockerization
- CI/CD pipeline
- Cloud deployment
- Model calibration
- Real-time inference API
---








