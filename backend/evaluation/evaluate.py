"""
PHASE 5 – Comprehensive Model Evaluation & Visualizations
=========================================================
Purpose:
    1. Load the trained best model bundle from backend/models/best_model.pkl.
    2. Generate and save the Model Comparison Cross-Validation Bar Chart
       (with mean accuracy and standard deviation error bars).
    3. Generate and save the Confusion Matrix Heatmap for the untouched holdout test set.
    4. Generate and display the Detailed Classification Report (Precision, Recall, F1, Support).
    5. Extract and plot Top 10 Feature Importances (if Random Forest is the winning model)
       with appropriate scientific disclaimers.
    6. Save structured metrics to backend/evaluation/metrics.json for the Flask API / Frontend.

Run command (from the project root folder):
    venv/Scripts/python backend/evaluation/evaluate.py
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt

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
from preprocess import load_dataset, separate_features_and_target, split_data

# ─────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH = "backend/models/best_model.pkl"
DATASET_PATH = "dataset/emotions.csv"
PLOTS_DIR = "backend/evaluation/plots"
METRICS_JSON_PATH = "backend/evaluation/metrics.json"

os.makedirs(PLOTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# FUNCTION 1: plot_cv_comparison
# ──────────────────────────────────────────────────────────────
def plot_cv_comparison(cv_summary, output_path):
    """
    Plot a clean bar chart comparing 5-fold cross-validation performance
    across all candidate algorithms with standard deviation error bars.

    Parameters:
        cv_summary (dict): Cross-validation metrics dictionary.
        output_path (str): Destination image path.
    """
    models = list(cv_summary.keys())
    means = [cv_summary[m]["mean_accuracy"] * 100 for m in models]
    stds = [cv_summary[m]["std_accuracy"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf5ff")  # Subtle light-purple/gray background

    # Styling colors: Distinct highlight for winner (highest mean)
    winner_idx = int(np.argmax(means))
    colors = ["#c4b5fd" if i != winner_idx else "#7c3aed" for i in range(len(models))]

    # Plot bars with error bars representing 1 standard deviation
    bars = ax.bar(
        models,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="#6d28d9",
        linewidth=1.2,
        alpha=0.9,
        error_kw={"elinewidth": 1.5, "ecolor": "#4c1d95"}
    )

    ax.set_ylim(85, 102)
    ax.set_ylabel("Cross-Validation Accuracy (%)", fontsize=11, fontweight="bold", color="#1f2937")
    ax.set_title("5-Fold Stratified Cross-Validation Comparison (Training Set)", fontsize=13, fontweight="bold", color="#111827", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#d8b4fe")

    # Add numeric labels above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + stds[i] + 0.8,
            f"{height:.2f}%\n(±{stds[i]:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold" if i == winner_idx else "normal",
            color="#4c1d95" if i == winner_idx else "#374151"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved CV Comparison Plot      -> {output_path}")


# ──────────────────────────────────────────────────────────────
# FUNCTION 2: plot_confusion_matrix
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, output_path):
    """
    Generate and save an annotated Confusion Matrix heatmap using a purple palette.

    Parameters:
        cm (np.ndarray): 3x3 confusion matrix counts.
        classes (list): Class names ['NEGATIVE', 'NEUTRAL', 'POSITIVE'].
        output_path (str): Destination image path.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # Display image matrix with Purple colormap
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold", color="#374151")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold", color="#374151")

    ax.set_xlabel("Predicted Emotion Label", fontsize=11, fontweight="bold", color="#1f2937", labelpad=10)
    ax.set_ylabel("True Emotion Label", fontsize=11, fontweight="bold", color="#1f2937", labelpad=10)
    ax.set_title("Holdout Test Set Confusion Matrix (427 Samples)", fontsize=12, fontweight="bold", color="#111827", pad=15)

    # Annotate counts and percentages inside each cell
    total_samples = np.sum(cm)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            pct = (val / total_samples) * 100
            cell_text = f"{val}\n({pct:.1f}%)"
            text_color = "white" if val > thresh else "#1f2937"
            ax.text(
                j, i, cell_text,
                ha="center", va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved Confusion Matrix Plot   -> {output_path}")


# ──────────────────────────────────────────────────────────────
# FUNCTION 3: plot_feature_importance
# ──────────────────────────────────────────────────────────────
def plot_feature_importance(pipeline, feature_names, output_path):
    """
    Extract and plot top 10 feature importances from a Random Forest classifier.

    Includes a scientific disclaimer stating that Gini impurity importance
    reflects statistical decision-tree utility, not biological causality.

    Parameters:
        pipeline (Pipeline): Fitted pipeline containing RandomForestClassifier.
        feature_names (list): List of 2548 feature column names.
        output_path (str): Destination image path.

    Returns:
        list of tuple: Top 10 (feature_name, importance_score).
    """
    classifier = pipeline.named_steps.get("classifier")
    if not hasattr(classifier, "feature_importances_"):
        print("  ℹ Winning model does not provide feature_importances_. Skipping plot.")
        return []

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Indices of top 10

    top_features = [(feature_names[i], float(importances[i])) for i in indices]

    names = [item[0] for item in top_features][::-1]  # Reverse for bottom-up horizontal bar
    scores = [item[1] for item in top_features][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf5ff")

    bars = ax.barh(names, scores, color="#8b5cf6", edgecolor="#6d28d9", height=0.65)

    ax.set_xlabel("Gini Feature Importance Score", fontsize=10.5, fontweight="bold", color="#1f2937")
    ax.set_title("Top 10 EEG Features by Random Forest Importance", fontsize=12, fontweight="bold", color="#111827", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#d8b4fe")

    # Add numeric score at the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.0003,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=9,
            color="#4c1d95",
            fontweight="bold"
        )

    # Note / Disclaimer at the bottom
    plt.figtext(
        0.5, -0.04,
        "Note: Gini importance quantifies decision tree variance reduction in this dataset;\n"
        "it indicates statistical discriminative utility, not direct neurobiological causality.",
        ha="center", fontsize=8, color="#6b7280", style="italic"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Feature Importance Plot -> {output_path}")

    return top_features


# ──────────────────────────────────────────────────────────────
# FUNCTION 4: generate_and_save_metrics_json
# ──────────────────────────────────────────────────────────────
def generate_and_save_metrics_json(bundle, test_results, top_features, output_path):
    """
    Save all actual computed evaluation metrics into a structured JSON file.
    This enables the Flask backend and React frontend to display live
    experimental values cleanly.
    """
    export_data = {
        "selected_algorithm": bundle["algorithm"],
        "pipeline_steps": [step[0] for step in bundle["pipeline"].steps],
        "classes": bundle["classes"],
        "cross_validation": bundle["cv_summary"],
        "holdout_test_metrics": {
            "accuracy": test_results["accuracy"],
            "precision_weighted": test_results["precision"],
            "recall_weighted": test_results["recall"],
            "f1_weighted": test_results["f1"],
            "total_test_samples": 427,
            "confusion_matrix": test_results["confusion_matrix"],
            "classification_report": test_results["classification_report"]
        },
        "top_10_features": [
            {"feature": f[0], "importance": round(f[1], 5)}
            for f in top_features
        ]
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"  ✓ Saved Structured Metrics JSON -> {output_path}")


# ──────────────────────────────────────────────────────────────
# MAIN EVALUATION ROUTINE
# ──────────────────────────────────────────────────────────────
def run_evaluation_suite():
    print("=" * 65)
    print("PHASE 5: MODEL EVALUATION & VISUALIZATION SUITE")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model bundle '{MODEL_PATH}' not found. Please run train_model.py first."
        )

    # 1. Load the serialized model bundle
    bundle = joblib.load(MODEL_PATH)
    algorithm_name = bundle["algorithm"]
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    classes = bundle["classes"]
    cv_summary = bundle["cv_summary"]
    test_results = bundle["test_results"]

    print(f"\n[1] LOADED MODEL BUNDLE:")
    print(f"    - Winning Algorithm : {algorithm_name}")
    print(f"    - Classes           : {classes}")
    print(f"    - Total Features    : {len(feature_names)}")

    # 2. Print Cross-Validation Summary
    print("\n" + "=" * 65)
    print("[2] 5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY (TRAINING SET)")
    print("=" * 65)
    for model_name, data in cv_summary.items():
        print(f"  {model_name:<26}: Mean Acc = {data['mean_accuracy'] * 100:.2f}%  (± {data['std_accuracy'] * 100:.2f}%)")
        print(f"    Fold Accuracies        : {[round(s * 100, 2) for s in data['fold_scores']]}")

    # 3. Print Holdout Test Performance & Classification Report
    print("\n" + "=" * 65)
    print("[3] HOLDOUT TEST SET PERFORMANCE (UNTOUCHED 427 SAMPLES)")
    print("=" * 65)
    print(f"  - Accuracy            : {test_results['accuracy'] * 100:.2f}%")
    print(f"  - Precision (Weighted): {test_results['precision'] * 100:.2f}%")
    print(f"  - Recall (Weighted)   : {test_results['recall'] * 100:.2f}%")
    print(f"  - F1-Score (Weighted) : {test_results['f1'] * 100:.2f}%")

    print("\n  Class-wise Performance & Support:")
    report_dict = test_results["classification_report"]
    print(f"  {'Class':<12}  {'Precision':>10}  {'Recall':>10}  {'F1-Score':>10}  {'Support':>10}")
    print("  " + "-" * 56)
    for cls in classes:
        cls_data = report_dict[cls]
        print(
            f"  {cls:<12}  "
            f"{cls_data['precision'] * 100:>9.2f}%  "
            f"{cls_data['recall'] * 100:>9.2f}%  "
            f"{cls_data['f1-score'] * 100:>9.2f}%  "
            f"{int(cls_data['support']):>10}"
        )
    print("  " + "-" * 56)
    w_avg = report_dict["weighted avg"]
    print(
        f"  {'Weighted Avg':<12}  "
        f"{w_avg['precision'] * 100:>9.2f}%  "
        f"{w_avg['recall'] * 100:>9.2f}%  "
        f"{w_avg['f1-score'] * 100:>9.2f}%  "
        f"{int(w_avg['support']):>10}\n"
    )

    # 4. Generate Visualizations
    print("=" * 65)
    print("[4] GENERATING & SAVING RESEARCH VISUALIZATIONS")
    print("=" * 65)

    # A. Model Comparison Bar Chart
    cv_plot_path = os.path.join(PLOTS_DIR, "model_comparison_cv.png")
    plot_cv_comparison(cv_summary, cv_plot_path)

    # B. Confusion Matrix
    cm_arr = np.array(test_results["confusion_matrix"])
    cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm_arr, classes, cm_plot_path)

    # C. Feature Importance
    feat_plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    top_features = plot_feature_importance(pipeline, feature_names, feat_plot_path)

    # 5. Display Top 10 Feature Importances
    if top_features:
        print("\n" + "=" * 65)
        print("[5] TOP 10 DISCRIMINATIVE EEG FEATURES (RANDOM FOREST)")
        print("=" * 65)
        print(f"  {'Rank':<6}  {'Feature Name':<28}  {'Importance Score':>16}")
        print("  " + "-" * 54)
        for rank, (name, score) in enumerate(top_features, 1):
            print(f"  #{rank:<5} {name:<28}  {score:>16.5f}")
        print("  " + "-" * 54)
        print("  Disclaimer: Feature importance measures decision tree variance reduction;")
        print("              it does not prove biological or cognitive causality.")

    # 6. Save JSON Metrics
    print("\n" + "=" * 65)
    print("[6] EXPORTING METRICS JSON ARTIFACT")
    print("=" * 65)
    generate_and_save_metrics_json(bundle, test_results, top_features, METRICS_JSON_PATH)

    print("\n" + "=" * 65)
    print("PHASE 5 COMPLETE: ALL EVALUATIONS & PLOTS READY")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation_suite()
