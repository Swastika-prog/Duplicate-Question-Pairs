# Duplicate Question Pairs Detection
A Machine Learning–based application that detects whether two questions are semantically duplicate using Natural Language Processing (NLP) techniques.  
The system is built using FastAPI for the backend and Streamlit for the frontend, forming an end-to-end ML deployment pipeline.

## Live Deployment
- 🔗 **Frontend (Streamlit App):**
  ```
  https://duplicate-question-pairs-quora.streamlit.app/
  ```
- 🔗 **Backend API (FastAPI):**
  ```
  https://duplicate-question-pairs-quora.onrender.com/docs
  ```

The Streamlit frontend communicates with a FastAPI backend deployed on Render,
forming a production-style ML inference pipeline.
---

## About the Project
This project demonstrates how an NLP-based Machine Learning model can be deployed as a REST API and accessed through a simple web interface.  
It focuses on semantic similarity detection using TF-IDF vectorization and cosine similarity to identify redundant or duplicate questions commonly found on Q&A platforms.

## Features
- Duplicate question detection using NLP techniques  
- TF-IDF vectorization for text representation  
- Feature-based semantic similarity using TF-IDF and fuzzy matching  
- Supervised classification using SVM and XGBoost  
- FastAPI backend for real-time inference  
- Streamlit-based interactive frontend  
- Model and vectorizer loading using saved artifacts (.pkl)  
- Clean and modular project structure
  
## Tech Stack
- **Language:** Python  
- **Machine Learning:** Scikit-learn  
- **NLP:** TF-IDF, Cosine Similarity  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Data Processing:** Pandas, NumPy  

## Project Structure
```
Duplicate-Question-Pairs/
│
├── backend/
│   └── app.py                  (FastAPI backend)
│
├── frontend/
│   └── streamlit_app.py        (Streamlit frontend)
│
├── data/
│   └── train.csv               (Training dataset)
│
├── model.ipynb                 (Model training notebook)
├── model.pkl                   (Trained similarity model)
├── tfidf.pkl                   (TF-IDF vectorizer)
├── threshold.pkl               (Similarity threshold)
├── requirements.txt            (Project dependencies)
└── .gitignore
```
## How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the FastAPI backend
```bash
uvicorn backend.app:app --reload
```

#### Backend URL
```
http://127.0.0.1:8000
```

#### Swagger UI
```
http://127.0.0.1:8000/docs
```

### Step 3: Run the Streamlit frontend
```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints
### POST `/predict`
Predicts whether two input questions are duplicate or not.


## Dataset
The dataset consists of question pairs used to train a semantic similarity model.  
It enables the system to learn linguistic patterns and contextual similarity between questions.

## Purpose
This project is intended for learning and demonstrating:

- NLP-based semantic similarity techniques  
- Machine Learning model deployment  
- REST API development using FastAPI  
- Frontend–backend integration with Streamlit  
- End-to-end ML project structuring and deployment
