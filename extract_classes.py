import joblib
import os

BASE_DIR = r"c:\Users\Victus\OneDrive\Desktop\CareerSphereAI2.0\ml\models"

def print_classes(model_name):
    path = os.path.join(BASE_DIR, f"{model_name}_encoders.pkl")
    if os.path.exists(path):
        encoders = joblib.load(path)
        print(f"--- {model_name} ---")
        for col, enc in encoders.items():
            if hasattr(enc, 'classes_'):
                print(f"{col}: {enc.classes_.tolist()}")
            else:
                print(f"{col}: (Numerical/Unknown)")

print_classes("career")
print_classes("mental_health")
print_classes("burnout")
