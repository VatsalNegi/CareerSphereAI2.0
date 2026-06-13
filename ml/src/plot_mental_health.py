import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


# -------------------------------
# Load Data
# -------------------------------
def load_data():
    df = pd.read_csv("../datasets/mental_health_dataset.csv")

    X = df.drop("mental_health_status", axis=1)
    y = df["mental_health_status"]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    return X, y


# -------------------------------
# Train Model
# -------------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


# -------------------------------
# Create Folder
# -------------------------------
def create_plot_folder():
    os.makedirs("../models/plots", exist_ok=True)


# -------------------------------
# 1. Confusion Matrix
# -------------------------------
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 5))
    disp.plot()
    plt.title("Mental Health Confusion Matrix")
    plt.tight_layout()   # ✅ FIX
    plt.savefig("../models/plots/mental_health_confusion_matrix.png")
    plt.close()


# -------------------------------
# 2. CV Scores
# -------------------------------
def plot_cv_scores(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)

    plt.figure(figsize=(6, 4))
    plt.plot(scores, marker='o')
    plt.title("Cross Validation Scores")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.tight_layout()   # ✅ FIX
    plt.savefig("../models/plots/mental_health_cv_scores.png")
    plt.close()


# -------------------------------
# 3. Feature Importance (FIXED)
# -------------------------------
def plot_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))  # ✅ bigger width
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    
    plt.tight_layout()   # ✅ IMPORTANT FIX
    plt.savefig("../models/plots/mental_health_feature_importance.png")
    plt.close()


# -------------------------------
# 4. Heatmap (FIXED)
# -------------------------------
def plot_heatmap(X):
    plt.figure(figsize=(10, 8))  # ✅ bigger size

    corr = X.corr()
    sns.heatmap(corr, annot=False)

    plt.title("Feature Correlation Heatmap")
    plt.xticks(rotation=45, ha='right')  # ✅ rotate labels
    plt.yticks(rotation=0)

    plt.tight_layout()   # ✅ CRITICAL FIX
    plt.savefig("../models/plots/mental_health_heatmap.png")
    plt.close()


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("📊 Generating Mental Health Plots...")

    create_plot_folder()

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    plot_confusion_matrix(model, X_test, y_test)
    plot_cv_scores(model, X, y)
    plot_feature_importance(model, X)
    plot_heatmap(X)

    print("✅ All plots generated successfully!")


if __name__ == "__main__":
    main()