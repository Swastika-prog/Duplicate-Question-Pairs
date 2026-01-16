from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np
import string
import scipy.sparse as sp
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ---------------- LOAD ARTIFACTS ----------------
with open(os.path.join(ARTIFACTS_DIR, "tfidf_q1.pkl"), "rb") as f:
    tfidf_q1 = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, "tfidf_q2.pkl"), "rb") as f:
    tfidf_q2 = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, "svm_model.pkl"), "rb") as f:
    svm_model = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

# ---------------- TEXT CLEANING ----------------
punct = str.maketrans("", "", string.punctuation)

def fast_clean(text: str) -> str:
    text = str(text).lower().translate(punct)
    return " ".join(w for w in text.split() if w.isalpha())

# ---------------- FEATURE ENGINEERING ----------------
def hybrid_features(q1: str, q2: str) -> np.ndarray:
    s1, s2 = set(q1.split()), set(q2.split())
    common = len(s1 & s2)
    total = max(len(s1 | s2), 1)
    overlap = common / total
    len_diff = abs(len(q1.split()) - len(q2.split()))

    f1 = fuzz.QRatio(q1, q2)
    f2 = fuzz.partial_ratio(q1, q2)
    f3 = fuzz.token_sort_ratio(q1, q2)
    f4 = fuzz.token_set_ratio(q1, q2)

    return np.array([[common, overlap, len_diff, f1, f2, f3, f4]])

def cosine_sim(A, B) -> np.ndarray:
    return np.array([[cosine_similarity(A, B)[0][0]]])

# ---------------- FASTAPI ----------------
app = FastAPI(title="Duplicate Question Detector API")

class Query(BaseModel):
    question1: str
    question2: str
    model: str = "svm"   # svm | xgb

@app.post("/predict")
def predict(query: Query):
    try:
        q1 = fast_clean(query.question1)
        q2 = fast_clean(query.question2)

        # TF-IDF
        X_q1 = tfidf_q1.transform([q1])
        X_q2 = tfidf_q2.transform([q2])

        # Extra features
        eng = hybrid_features(q1, q2)
        cos = cosine_sim(X_q1, X_q2)

        # Final feature vector
        X_final = sp.hstack([X_q1, X_q2, eng, cos])

        if query.model.lower() == "svm":
            pred = int(svm_model.predict(X_final)[0])
            return {
                "model": "SVM",
                "is_duplicate": pred
            }

        elif query.model.lower() == "xgb":
            pred = int(xgb_model.predict(X_final)[0])
            prob = float(xgb_model.predict_proba(X_final)[0][1])
            return {
                "model": "XGBoost",
                "is_duplicate": pred,
                "confidence": prob
            }

        else:
            raise HTTPException(status_code=400, detail="Model must be svm or xgb")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
