from fastapi import APIRouter, Depends
from app.utils.auth_utils import get_current_user
from app.models.user_model import User
import joblib
import numpy as np
import pandas as pd
import os

from app.schemas.burnout_schema import BurnoutInput
from app.services.burnout_report import generate_burnout_report
from app.routes.chatbot import store_report

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../ml/models/burnout_model.pkl")
)

ENCODER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../../ml/models/burnout_encoders.pkl")
)

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

label_map = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}


def preprocess_input(data):
    df = pd.DataFrame([data])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                df[col] = encoder.transform([encoder.classes_[0]])

    return df


@router.post("/predict/burnout")
def predict_burnout(input_data: BurnoutInput, current_user: User = Depends(get_current_user)):

    data_dict = input_data.dict()

    processed = preprocess_input(data_dict)

    pred = model.predict(processed)[0]
    probs = model.predict_proba(processed)[0]

    prediction = label_map[int(pred)]
    confidence = float(np.max(probs) * 100)

    ai_report = generate_burnout_report(
        data_dict, prediction, round(confidence, 2)
    )

    user_id = current_user.email

    store_report(user_id, ai_report, "burnout", raw_data=data_dict)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "ai_report": ai_report,
        "user_id": user_id
    }