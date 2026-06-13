from fastapi import APIRouter, Depends
from app.utils.auth_utils import get_current_user
from app.models.user_model import User
import numpy as np
import joblib
import pandas as pd
import os

from app.schemas.career_schema import CareerInput
from app.services.ai_report import generate_ai_report
from app.routes.chatbot import store_report  # ✅ IMPORTANT

router = APIRouter()

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../ml/models/career_model.pkl")
)

ENCODER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../ml/models/career_encoders.pkl")
)

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

label_map = {
    0: "Low",
    1: "Moderate",
    2: "High"
}


# -----------------------------
# PREPROCESS INPUT
# -----------------------------
def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    return df


# -----------------------------
# PREDICT CAREER API
# -----------------------------
@router.post("/predict/career")
def predict_career(input_data: CareerInput, current_user: User = Depends(get_current_user)):

    data_dict = input_data.dict()

    # Preprocess
    processed = preprocess_input(data_dict)

    # Prediction
    pred = model.predict(processed)[0]
    probs = model.predict_proba(processed)[0]

    prediction = label_map[int(pred)]
    confidence = float(np.max(probs) * 100)

    # -----------------------------
    # AI REPORT
    # -----------------------------
    ai_report = generate_ai_report(
        data_dict,
        prediction,
        round(confidence, 2)
    )

    # -----------------------------
    # STORE REPORT FOR CHATBOT
    # -----------------------------
    user_id = current_user.email  # Dynamic from auth

    store_report(user_id, ai_report,"career", raw_data=data_dict)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "ai_report": ai_report,
        "user_id": user_id
    }