from fastapi import APIRouter, Depends
from app.utils.auth_utils import get_current_user
from app.models.user_model import User
import numpy as np
import joblib
import pandas as pd
import os

from app.schemas.mental_health_schema import MentalHealthInput
from app.services.mental_health_report import generate_mental_health_report
from app.routes.chatbot import store_report

router = APIRouter()

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../ml/models/mental_health_model.pkl")
)

ENCODER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../ml/models/mental_health_encoders.pkl")
)

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

label_map = {
    0: "Healthy",
    1: "Mild Stress",
    2: "Moderate Stress",
    3: "High Risk"
}


# -----------------------------
# PREPROCESS INPUT
# -----------------------------
def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # ✅ Clean + normalize input
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # ✅ Encode safely
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                # 🚨 Handle unseen label
                df[col] = df[col].apply(
                    lambda x: encoder.transform([encoder.classes_[0]])[0]
                )

    return df

# -----------------------------
# PREDICT API
# -----------------------------
@router.post("/predict/mental-health")
def predict_mental_health(input_data: MentalHealthInput, current_user: User = Depends(get_current_user)):

    data_dict = input_data.dict()

    processed = preprocess_input(data_dict)

    pred = model.predict(processed)[0]
    probs = model.predict_proba(processed)[0]

    prediction = label_map[int(pred)]
    confidence = float(np.max(probs) * 100)

    # -----------------------------
    # AI REPORT
    # -----------------------------
    ai_report = generate_mental_health_report(
        data_dict,
        prediction,
        round(confidence, 2)
    )

    # -----------------------------
    # STORE FOR CHATBOT
    # -----------------------------
    user_id = current_user.email

    store_report(user_id, ai_report,"mental_health", raw_data=data_dict)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "ai_report": ai_report,
        "user_id": user_id
    }