import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

print("📊 Generating Burnout Plots...")

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
df = pd.read_csv("../datasets/burnout_dataset.csv")

model = joblib.load("../models/burnout_model.pkl")
encoders = joblib.load("../models/burnout_encoders.pkl")

# Encode dataset (same as training)
for col in df.select_dtypes(include="object").columns:
    df[col] = encoders[col].transform(df[col])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

# -----------------------------
# CREATE PLOT DIRECTORY
# -----------------------------
plot_dir = "../models/plots"
os.makedirs(plot_dir, exist_ok=True)

# -----------------------------
# 1. CONFUSION MATRIX
# -----------------------------
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Burnout Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{plot_dir}/burnout_confusion_matrix.png")
plt.close()

# -----------------------------
# 2. FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Burnout Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(f"{plot_dir}/burnout_feature_importance.png")
plt.close()

# -----------------------------
# 3. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Burnout Feature Correlation")
plt.tight_layout()
plt.savefig(f"{plot_dir}/burnout_heatmap.png")
plt.close()

# -----------------------------
# 4. CROSS VALIDATION SCORES
# -----------------------------
scores = cross_val_score(model, X, y, cv=5)

plt.figure(figsize=(8, 5))
plt.plot(scores, marker='o')
plt.title("Burnout CV Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{plot_dir}/burnout_cv_scores.png")
plt.close()

print("✅ All Burnout plots generated successfully!")