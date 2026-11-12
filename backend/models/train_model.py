"""
PHASE 4 & 5 – Machine Learning Model Training & Selection via Cross-Validation
=============================================================================
Purpose:
    1. Separate features (X) and target (y) using preprocess.py.
    2. Split into 80% Training Set and 20% Holdout Test Set.
       CRITICAL: The holdout test set is kept completely untouched during model selection.
    3. Perform 5-Fold Stratified Cross-Validation on the training set (X_train, y_train)
       for all three candidate algorithms:
           - Logistic Regression    (StandardScaler -> LogisticRegression)
           - Random Forest         (RandomForestClassifier, no scaling needed)
           - Support Vector Machine (StandardScaler -> SVC)
    4. Select the best-performing algorithm based on cross-validation results.
    5. Fit the selected winning pipeline on the full training set (X_train, y_train).
    6. Evaluate the winning model ONCE on the untouched holdout test set (X_test, y_test).
    7. Save the trained pipeline and evaluation metadata to backend/models/best_model.pkl.

Run command (from the project root folder):
    venv/Scripts/python backend/models/train_model.py
"""

import sys
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ── Allow importing from backend/preprocessing/ ───────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
from preprocess import (
    load_dataset,
    validate_dataset,
    separate_features_and_target,
    split_data
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = "dataset/emotions.csv"
MODEL_SAVE_PATH = "backend/models/best_model.pkl"
RANDOM_STATE = 42
CV_FOLDS = 5


# ──────────────────────────────────────────────────────────────
# FUNCTION 1: build_model_pipelines
# ──────────────────────────────────────────────────────────────
def build_model_pipelines():
    """
    Construct the three candidate classification pipelines.

    Why each pipeline is configured this way:
    1. Logistic Regression:
       - StandardScaler -> LogisticRegression
       - Features have vastly different scales. StandardScaler prevents high-magnitude
         features from dominating gradient updates.
       - max_iter=1000 ensures convergence on 2,548 features.

    2. Random Forest:
       - RandomForestClassifier (no scaler)
       - Tree-based models are invariant to monotonic scaling; thresholds depend only
         on value rank order. Omitting StandardScaler saves computation and memory.
       - n_estimators=100 provides a robust ensemble.

    3. Support Vector Machine (SVM):
       - StandardScaler -> SVC(kernel='rbf')
       - Distance/dot-product calculations require scaled features so each dimension
         contributes proportionally. RBF kernel models non-linear EEG boundaries.

    Returns:
        dict: { model_name: sklearn.pipeline.Pipeline }
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return pipelines


# ──────────────────────────────────────────────────────────────
# FUNCTION 2: run_cross_validation
# ──────────────────────────────────────────────────────────────
def run_cross_validation(pipelines, X_train, y_train):
    """
    Perform 5-fold Stratified Cross-Validation on the training set only.

    Why StratifiedKFold on X_train?
        - Evaluates generalization without touching the holdout test set.
        - Stratification ensures each fold has the exact same class distribution
          (approx. 33.3% Negative, 33.3% Neutral, 33.3% Positive).
        - Prevents test-set snooping or data leakage during model comparison.

    Parameters:
        pipelines (dict) : { name: Pipeline }
        X_train (DataFrame): Training features
        y_train (Series)   : Training labels

    Returns:
        dict: Detailed cross-validation results for each algorithm.
    """
    print("\n" + "=" * 65)
    print(f"STEP 1: 5-FOLD STRATIFIED CROSS-VALIDATION ON TRAINING SET")
    print(f"        (Training samples: {len(X_train)}, Folds: {CV_FOLDS})")
    print("=" * 65)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}

    for name, pipeline in pipelines.items():
        print(f"\n  Evaluating: {name} ...")
        start_time = time.time()

        # Compute accuracy for each of the 5 folds
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        duration = time.time() - start_time

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        cv_summary[name] = {
            "pipeline": pipeline,
            "fold_scores": [round(float(s), 4) for s in scores],
            "mean_accuracy": round(mean_score, 4),
            "std_accuracy": round(std_score, 4),
            "duration_sec": round(duration, 2)
        }

        print(f"    - Fold scores : {[round(float(s) * 100, 2) for s in scores]}%")
        print(f"    - Mean Accuracy: {mean_score * 100:.2f}% (± {std_score * 100:.2f}%)")
        print(f"    - Duration     : {duration:.2f}s")

    return cv_summary


# ──────────────────────────────────────────────────────────────
# FUNCTION 3: select_best_model
# ──────────────────────────────────────────────────────────────
def select_best_model(cv_summary):
    """
    Select the winning algorithm based solely on mean cross-validation accuracy.

    Parameters:
        cv_summary (dict): Results from run_cross_validation().

    Returns:
        best_name (str): Name of winning algorithm.
    """
    print("\n" + "=" * 65)
    print("STEP 2: MODEL SELECTION TABLE (BASED ON CROSS-VALIDATION)")
    print("=" * 65)

    header = f"  {'Algorithm':<26}  {'Mean CV Acc':>12}  {'Std Dev':>9}  {'CV Time':>9}"
    print(header)
    print("  " + "-" * 62)

    # Sort descending by mean CV accuracy
    sorted_models = sorted(
        cv_summary.items(),
        key=lambda item: item[1]["mean_accuracy"],
        reverse=True
    )

    for name, stats in sorted_models:
        print(
            f"  {name:<26}  "
            f"{stats['mean_accuracy'] * 100:>11.2f}%  "
            f"±{stats['std_accuracy'] * 100:>6.2f}%  "
            f"{stats['duration_sec']:>8.2f}s"
        )
    print("  " + "-" * 62)

    best_name = sorted_models[0][0]
    best_stats = sorted_models[0][1]

    print(f"\n  ★ SELECTED WINNING MODEL: {best_name}")
    print(f"    Cross-Validation Mean Accuracy: {best_stats['mean_accuracy'] * 100:.2f}% (± {best_stats['std_accuracy'] * 100:.2f}%)\n")

    return best_name


# ──────────────────────────────────────────────────────────────
# FUNCTION 4: train_and_evaluate_final_model
# ──────────────────────────────────────────────────────────────
def train_and_evaluate_final_model(winning_pipeline, winning_name, X_train, y_train, X_test, y_test):
    """
    Fit the selected pipeline on the full training set and evaluate
    it once on the holdout test set.

    Parameters:
        winning_pipeline (Pipeline): Sklearn pipeline for the winning model
        winning_name (str): Algorithm name
        X_train, y_train: Full 80% training data
        X_test, y_test: Untouched 20% holdout test data

    Returns:
        dict: Holdout test metrics, predictions, and confusion matrix.
    """
    print("=" * 65)
    print(f"STEP 3: FINAL TRAINING & EVALUATION OF {winning_name.upper()}")
    print("=" * 65)

    print(f"  Fitting {winning_name} on all {len(X_train)} training samples ...")
    start_time = time.time()
    winning_pipeline.fit(X_train, y_train)
    fit_duration = time.time() - start_time
    print(f"  ✓ Training completed in {fit_duration:.2f} seconds.")

    print(f"\n  Evaluating on untouched holdout test set ({len(X_test)} samples) ...")
    y_pred = winning_pipeline.predict(X_test)

    classes = sorted(y_test.unique())
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_dict = classification_report(y_test, y_pred, labels=classes, zero_division=0, output_dict=True)
    report_str = classification_report(y_test, y_pred, labels=classes, zero_division=0)

    print(f"\n  Holdout Test Set Results:")
    print(f"    - Accuracy  : {acc * 100:.2f}%")
    print(f"    - Precision : {prec * 100:.2f}% (weighted)")
    print(f"    - Recall    : {rec * 100:.2f}% (weighted)")
    print(f"    - F1-Score  : {f1 * 100:.2f}% (weighted)")

    print("\n  Confusion Matrix (Rows=True, Columns=Predicted):")
    print(f"    Classes: {classes}")
    for idx, row in enumerate(cm):
        print(f"    {classes[idx]:<10}: {row.tolist()}")

    print("\n  Classification Report:")
    for line in report_str.splitlines():
        print(f"    {line}")

    test_results = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "classification_report": report_dict,
        "fit_duration_sec": round(fit_duration, 2)
    }

    return test_results


# ──────────────────────────────────────────────────────────────
# FUNCTION 5: save_best_model_bundle
# ──────────────────────────────────────────────────────────────
def save_best_model_bundle(winning_name, winning_pipeline, cv_summary, test_results, feature_names):
    """
    Serialize the winning model pipeline and full experiment metadata using joblib.
    """
    print("\n" + "=" * 65)
    print("STEP 4: SERIALIZING BEST MODEL & EXPERIMENT METADATA")
    print("=" * 65)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    bundle = {
        "algorithm": winning_name,
        "pipeline": winning_pipeline,
        "feature_names": feature_names,
        "classes": test_results["classes"],
        "cv_summary": {
            name: {
                "fold_scores": data["fold_scores"],
                "mean_accuracy": data["mean_accuracy"],
                "std_accuracy": data["std_accuracy"],
                "duration_sec": data["duration_sec"]
            }
            for name, data in cv_summary.items()
        },
        "test_results": test_results
    }

    joblib.dump(bundle, MODEL_SAVE_PATH)
    file_size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024
    print(f"  ✓ Saved complete bundle to: {MODEL_SAVE_PATH}")
    print(f"  ✓ Bundle file size        : {file_size_kb:.1f} KB")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 4/5: MODEL TRAINING & RIGOROUS CROSS-VALIDATION SELECTION")
    print("=" * 65)

    # 1. Load and prepare dataset (reusing preprocess.py)
    df = load_dataset(DATASET_PATH)
    feature_cols, _ = validate_dataset(df)
    X, y = separate_features_and_target(df, feature_cols)

    # 2. Stratified split (80% train / 20% holdout test)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Build model pipelines
    pipelines = build_model_pipelines()

    # 4. Perform 5-fold Stratified CV on X_train only
    cv_summary = run_cross_validation(pipelines, X_train, y_train)

    # 5. Select the winner based on CV performance
    winning_name = select_best_model(cv_summary)
    winning_pipeline = pipelines[winning_name]

    # 6. Fit winning model on full X_train and evaluate on untouched holdout test set
    test_results = train_and_evaluate_final_model(
        winning_pipeline,
        winning_name,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # 7. Save model bundle
    save_best_model_bundle(
        winning_name,
        winning_pipeline,
        cv_summary,
        test_results,
        feature_cols
    )

    print("\n" + "=" * 65)
    print("MODEL TRAINING & SELECTION COMPLETED SUCCESSFULLY")
    print("=" * 65)
