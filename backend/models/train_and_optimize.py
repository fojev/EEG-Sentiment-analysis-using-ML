"""
EEG Sentiment Analysis - Model Training, Hyperparameter Optimization & Benchmarking Suite
========================================================================================
Author: Senior Machine Learning Engineer & Data Scientist
Architecture: Supervised Multi-Class Affective State Classification (Negative, Neutral, Positive)
Dataset: 2,132 records x 2,548 EEG frequency-domain features (Kaggle emotions.csv)

Strict Rigor Standards:
1. Featurization & Scaling: Zero data leakage. Any scaling is strictly wrapped inside Pipelines.
2. Holdout Test Set: 20% stratified holdout set (427 samples) locked until final inference.
3. Cross-Validation: 5-Fold Stratified CV strictly on the 80% training set (1,705 samples).
4. Evaluation Metric: Macro F1 and Accuracy to ensure balanced class performance.
5. Model Comparison: Random Forest vs Extra Trees vs SVM vs HistGradientBoosting vs Logistic Regression.
6. Hyperparameter Optimization: RandomizedSearchCV on Random Forest and Extra Trees.
7. Feature Importance: Gini Impurity (MDI) + Permutation Importance.
"""

import os
import sys
import time
import json
import joblib
import tempfile
import numpy as np
import pandas as pd

# Prevent Matplotlib cache warnings
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance

# Set random seed for complete reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "emotions.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_model.pkl")
METRICS_SAVE_PATH = os.path.join(PROJECT_ROOT, "backend", "evaluation", "metrics.json")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "backend", "evaluation", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_and_validate_data():
    """Load dataset, clean column names, and validate integrity."""
    print("=" * 70)
    print("PHASE 1: DATASET INGESTION & DATA LEAKAGE PREVENTION CHECKS")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if "# mean_0_a" in df.columns:
        df.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)

    print(f"  Dataset Shape : {df.shape[0]} rows, {df.shape[1]} columns")

    if "label" not in df.columns:
        raise ValueError("Missing 'label' target column in dataset.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    # Integrity assertions
    assert "label" not in X.columns, "CRITICAL ERROR: Data leakage detected (label in X)!"
    assert X.isnull().sum().sum() == 0, "Missing values found in features!"
    assert not np.isinf(X.values).any(), "Infinite values found in features!"

    class_counts = y.value_counts()
    print("  Class Distribution:")
    for cls, cnt in class_counts.items():
        print(f"    - {cls:<10}: {cnt} records ({cnt/len(y)*100:.2f}%)")

    # Perform 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\n  Stratified Split:")
    print(f"    - Training Set : {len(X_train)} samples (80%)")
    print(f"    - Holdout Test : {len(X_test)} samples (20%) [LOCKED UNTIL FINAL EVALUATION]")

    return X_train, X_test, y_train, y_test, feature_cols


def benchmark_candidate_models(X_train, y_train):
    """
    Perform 5-fold Stratified Cross-Validation on the 80% training set
    across 5 diverse candidate algorithms without nested thread contention.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: 5-FOLD STRATIFIED CV BENCHMARKING (ON TRAINING DATA ONLY)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted"
    }

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, tol=1e-3, solver="lbfgs", random_state=RANDOM_STATE))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE))
        ]),
        "HistGradientBoosting": Pipeline([
            ("classifier", HistGradientBoostingClassifier(max_iter=50, max_bins=64, random_state=RANDOM_STATE))
        ]),
        "Default Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Extra Trees Classifier": Pipeline([
            ("classifier", ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
        ])
    }

    benchmark_results = {}

    for name, pipeline in candidates.items():
        print(f"  Benchmarking: {name:<26} ...", end=" ", flush=True)
        t0 = time.time()
        # n_jobs=1 in cross_validate allows internal n_jobs=-1 inside tree classifiers without thread deadlock
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True
        )
        elapsed = time.time() - t0

        mean_acc = float(np.mean(scores["test_accuracy"]))
        std_acc = float(np.std(scores["test_accuracy"]))
        mean_f1_macro = float(np.mean(scores["test_f1_macro"]))
        std_f1_macro = float(np.std(scores["test_f1_macro"]))
        mean_f1_weighted = float(np.mean(scores["test_f1_weighted"]))
        train_acc = float(np.mean(scores["train_accuracy"]))

        benchmark_results[name] = {
            "mean_cv_accuracy": round(mean_acc, 4),
            "std_cv_accuracy": round(std_acc, 4),
            "mean_cv_f1_macro": round(mean_f1_macro, 4),
            "std_cv_f1_macro": round(std_f1_macro, 4),
            "mean_cv_f1_weighted": round(mean_f1_weighted, 4),
            "mean_train_accuracy": round(train_acc, 4),
            "train_test_gap": round(train_acc - mean_acc, 4),
            "fold_scores_acc": [round(float(s), 4) for s in scores["test_accuracy"]],
            "cv_duration_sec": round(elapsed, 2)
        }

        print(f"Done in {elapsed:>5.1f}s | Acc: {mean_acc*100:6.2f}% (±{std_acc*100:.2f}%) | Macro F1: {mean_f1_macro*100:6.2f}%")

    print("\n  Summary Table of Candidate Models:")
    print(f"  {'Algorithm':<26}  {'CV Acc':>10}  {'Std Dev':>8}  {'Macro F1':>10}  {'Train-CV Gap':>12}  {'Time':>7}")
    print("  " + "-" * 78)
    for name, res in sorted(benchmark_results.items(), key=lambda x: x[1]["mean_cv_f1_macro"], reverse=True):
        print(
            f"  {name:<26}  "
            f"{res['mean_cv_accuracy']*100:>9.2f}%  "
            f"±{res['std_cv_accuracy']*100:>6.2f}%  "
            f"{res['mean_cv_f1_macro']*100:>9.2f}%  "
            f"{res['train_test_gap']*100:>11.2f}%  "
            f"{res['cv_duration_sec']:>6.1f}s"
        )

    return benchmark_results


def optimize_random_forest(X_train, y_train):
    """
    Perform systematic hyperparameter optimization on Random Forest
    using RandomizedSearchCV over 5-fold Stratified CV.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # Clean parameter distribution
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
        "bootstrap": [True]
    }

    # Set n_jobs=1 in base estimator and n_jobs=-1 in RandomizedSearchCV
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    t0 = time.time()
    print("  Running RandomizedSearchCV (15 parameter combinations x 5 folds = 75 fits)...")
    random_search.fit(X_train, y_train)
    tune_time = time.time() - t0
    print(f"  ✓ Tuning finished in {tune_time:.1f} seconds.")

    best_rf = random_search.best_estimator_
    # Ensure n_jobs=-1 for inference on best_rf
    best_rf.set_params(n_jobs=-1)
    best_params = random_search.best_params_
    best_cv_macro_f1 = float(random_search.best_score_)

    print(f"\n  ★ Best Random Forest Parameters:")
    for k, v in best_params.items():
        print(f"    - {k:<20}: {v}")
    print(f"  Best 5-Fold CV Macro F1: {best_cv_macro_f1*100:.2f}%")

    return best_rf, best_params, best_cv_macro_f1, random_search


def optimize_extra_trees(X_train, y_train):
    """
    Tune Extra Trees Classifier as a strong ensemble competitor.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: EXTRA TREES ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 25],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    base_et = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    et_search = RandomizedSearchCV(
        estimator=base_et,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    t0 = time.time()
    et_search.fit(X_train, y_train)
    tune_time = time.time() - t0
    best_et = et_search.best_estimator_
    best_et.set_params(n_jobs=-1)

    print(f"  ✓ Extra Trees tuning finished in {tune_time:.1f}s.")
    print(f"  Best Extra Trees 5-Fold CV Macro F1: {et_search.best_score_*100:.2f}%")

    return best_et, et_search.best_params_, float(et_search.best_score_)


def select_and_fit_final_model(best_rf, rf_f1, best_et, et_f1, X_train, y_train, X_test, y_test, feature_cols):
    """
    Compare the tuned Random Forest vs Extra Trees and fit the selected model.
    Then evaluate ONCE on the holdout test set.
    """
    print("\n" + "=" * 70)
    print("PHASE 5: FINAL MODEL SELECTION & HOLDOUT TEST SET EVALUATION")
    print("=" * 70)

    # Per prompt instructions: Keep Random Forest as preferred model if it performs best or provides the best practical balance.
    if et_f1 > rf_f1 + 0.005:
        winning_name = "Extra Trees Classifier"
        winning_clf = best_et
        winning_cv_score = et_f1
    else:
        winning_name = "Random Forest (Optimized)"
        winning_clf = best_rf
        winning_cv_score = rf_f1

    print(f"  ★ WINNING SELECTED MODEL : {winning_name}")
    print(f"    5-Fold CV Macro F1      : {winning_cv_score*100:.2f}%")

    # Build Pipeline
    winning_pipeline = Pipeline([
        ("classifier", winning_clf)
    ])

    print(f"\n  Fitting {winning_name} on ALL {len(X_train)} training records...")
    t0 = time.time()
    winning_pipeline.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"  ✓ Training completed in {fit_duration:.2f} seconds.")

    # Evaluate on Holdout Test Set
    print(f"\n  Evaluating on UNTOUCHED Holdout Test Set ({len(X_test)} samples)...")
    t0 = time.time()
    y_test_pred = winning_pipeline.predict(X_test)
    y_test_proba = winning_pipeline.predict_proba(X_test)
    inference_duration = time.time() - t0

    y_train_pred = winning_pipeline.predict(X_train)
    train_acc = float(accuracy_score(y_train, y_train_pred))

    classes = sorted(y_test.unique().tolist())
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec_macro = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_prec_weighted = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_rec_macro = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_rec_weighted = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_f1_macro = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_f1_weighted = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    report_dict = classification_report(y_test, y_test_pred, labels=classes, zero_division=0, output_dict=True)
    report_str = classification_report(y_test, y_test_pred, labels=classes, zero_division=0)

    print("\n  ================================================================")
    print("  FINAL HOLDOUT TEST PERFORMANCE (427 SAMPLES)")
    print("  ================================================================")
    print(f"  Training Accuracy     : {train_acc*100:.2f}%")
    print(f"  Holdout Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Generalization Gap    : {(train_acc - test_acc)*100:.2f}%")
    print(f"  Macro Precision       : {test_prec_macro*100:.2f}%")
    print(f"  Weighted Precision    : {test_prec_weighted*100:.2f}%")
    print(f"  Macro Recall          : {test_rec_macro*100:.2f}%")
    print(f"  Weighted Recall       : {test_rec_weighted*100:.2f}%")
    print(f"  Macro F1-Score        : {test_f1_macro*100:.2f}%")
    print(f"  Weighted F1-Score     : {test_f1_weighted*100:.2f}%")
    print(f"  Inference Latency     : {inference_duration*1000/len(X_test):.2f} ms/sample")

    print("\n  Confusion Matrix (Rows=True, Cols=Predicted):")
    print(f"  {'':<12} " + " ".join([f"{c:>10}" for c in classes]))
    for idx, row in enumerate(cm):
        print(f"  {classes[idx]:<12} " + " ".join([f"{val:>10}" for val in row]))

    print("\n  Per-Class Classification Report:")
    for line in report_str.splitlines():
        print(f"    {line}")

    test_results = {
        "accuracy": round(test_acc, 4),
        "train_accuracy": round(train_acc, 4),
        "generalization_gap": round(train_acc - test_acc, 4),
        "precision_macro": round(test_prec_macro, 4),
        "precision_weighted": round(test_prec_weighted, 4),
        "recall_macro": round(test_rec_macro, 4),
        "recall_weighted": round(test_rec_weighted, 4),
        "f1_macro": round(test_f1_macro, 4),
        "f1_weighted": round(test_f1_weighted, 4),
        "total_test_samples": len(X_test),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "classification_report": report_dict,
        "fit_duration_sec": round(fit_duration, 2),
        "inference_latency_ms": round(inference_duration * 1000 / len(X_test), 3)
    }

    return winning_name, winning_pipeline, winning_clf, test_results


def analyze_feature_importance(winning_clf, feature_cols, X_test, y_test):
    """
    Extract tree MDI importances and calculate Permutation Importance on test set.
    """
    print("\n" + "=" * 70)
    print("PHASE 6: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # 1. MDI Impurity-based importances
    if hasattr(winning_clf, "feature_importances_"):
        importances = winning_clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top_25_idx = sorted_idx[:25]

        print("  Top 15 Most Important EEG Features (MDI Impurity):")
        top_features_list = []
        for rank, idx in enumerate(top_25_idx[:15], 1):
            feat = feature_cols[idx]
            imp = importances[idx]
            print(f"    {rank:>2}. {feat:<24}: {imp*100:.3f}%")
            top_features_list.append({"feature": feat, "importance": round(float(imp), 5)})

        # 2. Permutation Importance on a sample of test data (80 samples for fast reliable evaluation)
        print("\n  Computing Permutation Importance on test set sample (5 repeats)...")
        sub_X_test = X_test.iloc[:80]
        sub_y_test = y_test.iloc[:80]
        perm = permutation_importance(
            winning_clf, sub_X_test, sub_y_test,
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_sorted = np.argsort(perm.importances_mean)[::-1][:5]
        print("  Top 5 Features by Permutation Importance:")
        for rank, idx in enumerate(perm_sorted, 1):
            print(f"    {rank}. {feature_cols[idx]:<24}: drop in accuracy = {perm.importances_mean[idx]*100:.2f}% (±{perm.importances_std[idx]*100:.2f}%)")

        return top_features_list, importances, top_25_idx
    return [], None, None


def generate_evaluation_plots(benchmark_results, cm, classes, importances, feature_cols, top_25_idx):
    """Save high-resolution diagnostic charts."""
    print("\n" + "=" * 70)
    print("PHASE 7: EXPORTING EVALUATION VISUALIZATIONS")
    print("=" * 70)

    # 1. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("Optimized Classifier Confusion Matrix\n(Holdout Test Set N=427)", fontsize=12, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"  ✓ Saved Confusion Matrix: {cm_path}")

    # 2. Feature Importance Bar Chart
    if importances is not None and top_25_idx is not None:
        top_15_idx = top_25_idx[:15][::-1]
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        y_pos = np.arange(len(top_15_idx))
        ax.barh(y_pos, importances[top_15_idx] * 100, color="#00A9D6", edgecolor="#203B5D", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_cols[i] for i in top_15_idx], fontsize=9)
        ax.set_xlabel("Feature Importance (% MDI)", fontsize=10, fontweight="bold")
        ax.set_title("Top 15 Most Influential EEG Signal Features", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(fi_path)
        plt.close()
        print(f"  ✓ Saved Feature Importance Chart: {fi_path}")

    # 3. Model Comparison Cross-Validation Bar Chart
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    model_names = list(benchmark_results.keys())
    acc_scores = [benchmark_results[m]["mean_cv_accuracy"] * 100 for m in model_names]
    f1_scores = [benchmark_results[m]["mean_cv_f1_macro"] * 100 for m in model_names]
    stds = [benchmark_results[m]["std_cv_accuracy"] * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, acc_scores, width, yerr=stds, capsize=4, label="Mean CV Accuracy (%)", color="#00A9D6", edgecolor="#203B5D")
    ax.bar(x + width/2, f1_scores, width, label="Mean CV Macro F1 (%)", color="#203B5D", edgecolor="#00A9D6")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(70, 103)
    ax.set_title("5-Fold Cross-Validation Model Benchmark Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    mc_path = os.path.join(PLOTS_DIR, "model_comparison_cv.png")
    plt.savefig(mc_path)
    plt.close()
    print(f"  ✓ Saved Model Comparison Chart: {mc_path}")


def save_artifacts_and_metrics(winning_name, winning_pipeline, best_params, benchmark_results, test_results, feature_cols, top_features_list):
    """Serialize model bundle and structured metrics JSON."""
    print("\n" + "=" * 70)
    print("PHASE 8: SERIALIZING OPTIMIZED MODEL BUNDLE & METRICS")
    print("=" * 70)

    # 1. Format metrics for backend/evaluation/metrics.json
    cv_summary_for_backend = {}
    for name, res in benchmark_results.items():
        cv_summary_for_backend[name] = {
            "fold_scores": res["fold_scores_acc"],
            "mean_accuracy": res["mean_cv_accuracy"],
            "std_accuracy": res["std_cv_accuracy"],
            "duration_sec": res["cv_duration_sec"]
        }

    metrics_payload = {
        "selected_algorithm": winning_name,
        "best_hyperparameters": best_params,
        "pipeline_steps": ["classifier"],
        "classes": test_results["classes"],
        "model_comparison": benchmark_results,
        "cross_validation": cv_summary_for_backend,
        "holdout_test_metrics": test_results,
        "top_features": top_features_list,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  ✓ Saved metrics to: {METRICS_SAVE_PATH}")

    # 2. Build model bundle compatible with app.py
    model_bundle = {
        "algorithm": winning_name,
        "pipeline": winning_pipeline,
        "feature_names": feature_cols,
        "classes": test_results["classes"],
        "best_params": best_params,
        "cv_summary": cv_summary_for_backend,
        "model_comparison": benchmark_results,
        "test_results": test_results,
        "top_features": top_features_list,
        "training_metadata": {
            "training_samples": 1705,
            "test_samples": 427,
            "total_features": len(feature_cols),
            "random_state": RANDOM_STATE,
            "cv_folds": 5,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    joblib.dump(model_bundle, MODEL_SAVE_PATH)
    file_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"  ✓ Saved complete model bundle to: {MODEL_SAVE_PATH}")
    print(f"  ✓ Bundle size: {file_size_mb:.2f} MB")


def main():
    print("=" * 70)
    print("EEG SENTIMENT ANALYSIS - ML PIPELINE RETRAINING & OPTIMIZATION")
    print("=" * 70)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_cols = load_and_validate_data()

    # 2. Benchmark candidate models on 5-fold Stratified CV
    benchmark_results = benchmark_candidate_models(X_train, y_train)

    # 3. Optimize Random Forest
    best_rf, rf_params, rf_cv_f1, _ = optimize_random_forest(X_train, y_train)

    # 4. Optimize Extra Trees
    best_et, et_params, et_cv_f1 = optimize_extra_trees(X_train, y_train)

    # 5. Select & fit winning model, evaluate on holdout test set
    winning_name, winning_pipeline, winning_clf, test_results = select_and_fit_final_model(
        best_rf, rf_cv_f1, best_et, et_cv_f1,
        X_train, y_train, X_test, y_test, feature_cols
    )

    # 6. Feature importance
    top_features_list, importances, top_25_idx = analyze_feature_importance(
        winning_clf, feature_cols, X_test, y_test
    )

    # 7. Generate diagnostic plots
    generate_evaluation_plots(
        benchmark_results,
        np.array(test_results["confusion_matrix"]),
        test_results["classes"],
        importances,
        feature_cols,
        top_25_idx
    )

    # 8. Save artifacts
    save_artifacts_and_metrics(
        winning_name,
        winning_pipeline,
        rf_params if "Random Forest" in winning_name else et_params,
        benchmark_results,
        test_results,
        feature_cols,
        top_features_list
    )

    print("\n" + "=" * 70)
    print("ALL TRAINING, OPTIMIZATION & EVALUATION TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
