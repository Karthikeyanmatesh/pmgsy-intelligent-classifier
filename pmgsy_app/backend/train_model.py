"""
train_model.py
Train a Random Forest classifier on the PMGSY dataset.
Saves the trained model + label encoder to ../models/
Run once before starting the Flask server:
    python train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "..", "PMGSY_DATASET.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature columns (physical + financial) ──────────────────────────────────
FEATURE_COLS = [
    "NO_OF_ROAD_WORK_SANCTIONED",
    "LENGTH_OF_ROAD_WORK_SANCTIONED",
    "NO_OF_BRIDGES_SANCTIONED",
    "COST_OF_WORKS_SANCTIONED",
    "NO_OF_ROAD_WORKS_COMPLETED",
    "LENGTH_OF_ROAD_WORK_COMPLETED",
    "NO_OF_BRIDGES_COMPLETED",
    "EXPENDITURE_OCCURED",
    "NO_OF_ROAD_WORKS_BALANCE",
    "LENGTH_OF_ROAD_WORK_BALANCE",
    "NO_OF_BRIDGES_BALANCE",
    # Derived features
    "COMPLETION_RATE",
    "EXPENDITURE_RATE",
    "COST_PER_KM",
    "BRIDGE_RATIO",
    "BALANCE_RATIO",
]

TARGET_COL = "PMGSY_SCHEME"


def load_and_engineer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop trailing empty column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.dropna(subset=[TARGET_COL], inplace=True)

    # Convert numeric, coerce errors
    num_cols = [
        "NO_OF_ROAD_WORK_SANCTIONED","LENGTH_OF_ROAD_WORK_SANCTIONED",
        "NO_OF_BRIDGES_SANCTIONED","COST_OF_WORKS_SANCTIONED",
        "NO_OF_ROAD_WORKS_COMPLETED","LENGTH_OF_ROAD_WORK_COMPLETED",
        "NO_OF_BRIDGES_COMPLETED","EXPENDITURE_OCCURED",
        "NO_OF_ROAD_WORKS_BALANCE","LENGTH_OF_ROAD_WORK_BALANCE",
        "NO_OF_BRIDGES_BALANCE",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    eps = 1e-9
    df["COMPLETION_RATE"]   = df["NO_OF_ROAD_WORKS_COMPLETED"] / (df["NO_OF_ROAD_WORK_SANCTIONED"] + eps)
    df["EXPENDITURE_RATE"]  = df["EXPENDITURE_OCCURED"] / (df["COST_OF_WORKS_SANCTIONED"] + eps)
    df["COST_PER_KM"]       = df["COST_OF_WORKS_SANCTIONED"] / (df["LENGTH_OF_ROAD_WORK_SANCTIONED"] + eps)
    df["BRIDGE_RATIO"]      = df["NO_OF_BRIDGES_SANCTIONED"] / (df["NO_OF_ROAD_WORK_SANCTIONED"] + eps)
    df["BALANCE_RATIO"]     = df["NO_OF_ROAD_WORKS_BALANCE"] / (df["NO_OF_ROAD_WORK_SANCTIONED"] + eps)

    return df


def main():
    print("Loading dataset...")
    df = load_and_engineer(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importances
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))

    # Save artifacts
    joblib.dump(clf, os.path.join(MODEL_DIR, "pmgsy_classifier.pkl"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "label_encoder.pkl"))

    meta = {
        "feature_cols": FEATURE_COLS,
        "classes": le.classes_.tolist(),
        "accuracy": round(acc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std":  round(cv_scores.std(), 4),
        "feature_importances": importances,
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nArtifacts saved to pmgsy_app/models/")
    print("  pmgsy_classifier.pkl")
    print("  label_encoder.pkl")
    print("  model_meta.json")


if __name__ == "__main__":
    main()
