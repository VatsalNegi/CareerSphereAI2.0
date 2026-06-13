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
from sklearn.utils.class_weight import compute_class_weight

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False


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

    X = df.drop(columns=['career_readiness'])
    y = df['career_readiness']

    encoders = {}

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Add noise
    noise = np.random.normal(0, 0.3, X.shape)
    X = X + noise

    return X, y, encoders


# -----------------------------
# CLASS WEIGHTS
# -----------------------------
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    return dict(zip(classes, weights))


# -----------------------------
# TRAIN MODELS
# -----------------------------
def train_models(X_train, y_train):
    models = {}

    class_weights = get_class_weights(y_train)

    rf = RandomForestClassifier(
        n_estimators=60,
        max_depth=6,
        class_weight=class_weights,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    if xgb_available:
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb.fit(X_train, y_train)
        models['xgboost'] = xgb

    return models


# -----------------------------
# VISUALIZATION
# -----------------------------
def save_visuals(model, X_train, y_train, X_test, y_test, model_name):
    os.makedirs("../models/plots", exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # -----------------------------
    # METRICS HEATMAP
    # -----------------------------
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T

    plt.figure()
    sns.heatmap(df_report, annot=True, cmap="coolwarm")
    plt.title(f"{model_name} Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_metrics_heatmap.png", bbox_inches='tight')
    plt.close()

    # -----------------------------
    # TRAIN vs TEST ACCURACY
    # -----------------------------
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    plt.figure()
    plt.bar(["Train", "Test"], [train_acc, test_acc])
    plt.title(f"{model_name} Train vs Test Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_train_test.png", bbox_inches='tight')
    plt.close()

    # -----------------------------
    # CROSS VALIDATION (VALIDATION)
    # -----------------------------
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    plt.figure()
    plt.plot(cv_scores, marker='o')
    plt.title(f"{model_name} Cross Validation Scores")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"../models/plots/{model_name}_cv_scores.png", bbox_inches='tight')
    plt.close()

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"../models/plots/{model_name}_feature_importance.png", bbox_inches='tight')
        plt.close()


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}

    label_map = {0: "Low", 1: "Moderate", 2: "High"}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name.upper()} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Save all visuals
        save_visuals(model, X_train, y_train, X_test, y_test, name)

        print("\nSample Predictions:")
        for i in range(5):
            pred = label_map[y_pred[i]]
            confidence = round(np.max(probs[i]) * 100, 2)
            print(f"{pred} ({confidence}%)")

        results[name] = acc

    return results


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_best_model(models, results, encoders):
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    joblib.dump(best_model, "../models/career_model.pkl")
    joblib.dump(encoders, "../models/career_encoders.pkl")

    print("✅ Model and encoders saved successfully!")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = load_data("../datasets/career_dataset.csv")

    X, y, encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_train, y_train, X_test, y_test)

    save_best_model(models, results, encoders)