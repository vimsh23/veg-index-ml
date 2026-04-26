"""
train.py - Train Random Forest for vegetation / land cover classification

Usage:
    python train.py --data sample_data/training_data.csv --target class --output model/rf_model.joblib

Input CSV columns expected:
    blue, green, red, nir, [optional: class label column]

The script:
  1. Reads band values from CSV
  2. Computes all vegetation indices (NDVI, EVI, SAVI, NDWI, MSAVI)
  3. Builds feature matrix: raw bands + indices
  4. Trains Random Forest with cross-validation
  5. Reports accuracy, feature importance, classification report
  6. Saves model as joblib
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─── Vegetation Index formulas ───────────────────────────────────────────────

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)

def compute_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-10)

def compute_savi(nir, red, L=0.5):
    return ((nir - red) / (nir + red + L + 1e-10)) * (1 + L)

def compute_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-10)

def compute_msavi(nir, red):
    return (2 * nir + 1 - np.sqrt(np.maximum(0, (2 * nir + 1)**2 - 8 * (nir - red)))) / 2


def build_features(df: pd.DataFrame):
    """Build feature matrix from band DataFrame."""
    df.columns = [c.strip().lower() for c in df.columns]

    # Auto-scale DN to reflectance
    for col in ["blue", "green", "red", "nir"]:
        if col in df.columns and df[col].max() > 1.0:
            df[col] = df[col] / 10000.0

    nir   = df["nir"].values.astype(np.float32)
    red   = df["red"].values.astype(np.float32)
    green = df["green"].values.astype(np.float32)
    blue  = df["blue"].values.astype(np.float32)

    ndvi  = compute_ndvi(nir, red)
    evi   = compute_evi(nir, red, blue)
    savi  = compute_savi(nir, red)
    ndwi  = compute_ndwi(green, nir)
    msavi = compute_msavi(nir, red)

    feature_names = ["blue", "green", "red", "nir", "NDVI", "EVI", "SAVI", "NDWI", "MSAVI"]
    X = np.stack([blue, green, red, nir, ndvi, evi, savi, ndwi, msavi], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


def train(data_path: str, target_col: str = "class", output_path: str = "model/rf_model.joblib",
          n_estimators: int = 200, test_size: float = 0.2, cv_folds: int = 5):

    print(f"\n{'='*60}")
    print("  Vegetation Index ML - Random Forest Training")
    print(f"{'='*60}")

    # Load data
    print(f"\n[1/5] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Rows: {len(df)}, Columns: {list(df.columns)}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    y_raw = df[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"      Classes: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    # Build features
    print(f"\n[2/5] Computing vegetation indices...")
    X, feature_names = build_features(df.drop(columns=[target_col]))
    print(f"      Features ({len(feature_names)}): {feature_names}")
    print(f"      X shape: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n[3/5] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Train model
    print(f"\n[4/5] Training Random Forest (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluate
    print(f"\n[5/5] Evaluation")
    print("-" * 40)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {acc*100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    # Cross-validation
    print(f"  {cv_folds}-Fold Cross-Validation:")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Feature importance
    print("\n  Feature Importance:")
    importances = clf.feature_importances_
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {fname:<10} {bar} {imp:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le, "feature_names": feature_names}, output_path)
    print(f"\n  Model saved → {output_path}")
    print(f"{'='*60}\n")

    return clf, le


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF model for vegetation index classification")
    parser.add_argument("--data",        type=str,   default="sample_data/training_data.csv")
    parser.add_argument("--target",      type=str,   default="class")
    parser.add_argument("--output",      type=str,   default="model/rf_model.joblib")
    parser.add_argument("--estimators",  type=int,   default=200)
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--cv-folds",    type=int,   default=5)
    args = parser.parse_args()

    train(args.data, args.target, args.output, args.estimators, args.test_size, args.cv_folds)
