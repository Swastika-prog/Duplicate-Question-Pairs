print("MAIN FILE LOADING")

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load trained artifacts
# -----------------------------
with open("duplicate_question_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

pipeline = artifacts["pipeline"]
tfidf = artifacts["tfidf_word"]
char_tfidf = artifacts["tfidf_char"]
feature_cols = artifacts["feature_cols"]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Duplicate Question Detection API")

# -----------------------------
# Request schema
# -----------------------------
class QuestionPair(BaseModel):
    question1: str
    question2: str

# -----------------------------
# Helper functions (SAME AS NOTEBOOK)
# -----------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    return text

def extract_features(q1: str, q2: str) -> pd.DataFrame:
    q1_clean = preprocess(q1)
    q2_clean = preprocess(q2)

    q1_len = len(q1_clean)
    q2_len = len(q2_clean)
    len_diff = abs(q1_len - q2_len)

    first_word_eq = int(q1_clean.split()[0] == q2_clean.split()[0])
    common_words = len(set(q1_clean.split()) & set(q2_clean.split()))

    # TF-IDF cosine
    q1_tfidf = tfidf.transform([q1_clean])
    q2_tfidf = tfidf.transform([q2_clean])
    tfidf_cosine = (q1_tfidf.multiply(q2_tfidf)).sum()

    # Char TF-IDF cosine
    q1_char = char_tfidf.transform([q1_clean])
    q2_char = char_tfidf.transform([q2_clean])
    char_tfidf_cosine = (q1_char.multiply(q2_char)).sum()

    data = {
        "q1_len": q1_len,
        "q2_len": q2_len,
        "len_diff": len_diff,
        "first_word_eq": first_word_eq,
        "common_words": common_words,
        "fuzz_ratio": 100,        # same defaults you used
        "fuzz_partial": 100,
        "tfidf_cosine": float(tfidf_cosine),
        "char_tfidf_cosine": float(char_tfidf_cosine),
    }

    return pd.DataFrame([data])[feature_cols]

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: QuestionPair):
    X = extract_features(data.question1, data.question2)
    prob = pipeline.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    return {
        "duplicate": pred,
        "probability": round(float(prob), 4)
    }
