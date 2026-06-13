import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    return pd.read_csv(path)


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_data(df):
    df = df.copy()

    # Encode categorical
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=['burnout_risk'])
    y = df['burnout_risk']

    return X, y, encoders


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# VISUALIZATION
# -----------------------------
def save_visuals(model, X_train, y_train, X_test, y_test, model_name):
    os.makedirs("../models/plots", exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # Metrics Heatmap
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T
    plt.figure()
    sns.heatmap(df_report, annot=True, cmap="coolwarm")
    plt.title(f"{model_name} Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_metrics_heatmap.png", bbox_inches='tight')
    plt.close()

    # Train vs Test Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    plt.figure()
    plt.bar(["Train", "Test"], [train_acc, test_acc])
    plt.title(f"{model_name} Train vs Test Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_train_test.png", bbox_inches='tight')
    plt.close()

    # Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    plt.figure()
    plt.plot(cv_scores, marker='o')
    plt.title(f"{model_name} Cross Validation Scores")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_cv_scores.png", bbox_inches='tight')
    plt.close()

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"../models/plots/{model_name}_feature_importance.png", bbox_inches='tight')
        plt.close()


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 BURNOUT Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save all visuals
    save_visuals(model, X_train, y_train, X_test, y_test, "burnout")

    return acc


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("🚀 Training Burnout Model...")

    df = load_data("../datasets/burnout_dataset.csv")

    X, y, encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save artifacts
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/burnout_model.pkl")
    joblib.dump(encoders, "../models/burnout_encoders.pkl")

    print("\n✅ Model and visuals saved!")