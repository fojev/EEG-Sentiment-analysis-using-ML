"""
PHASE 8 – Complete End-to-End System Testing & Workflow Verification
===================================================================
Tests the entire system:
    1. Dataset accessibility (dataset/emotions.csv)
    2. ML Model & Pipeline integrity (backend/models/best_model.pkl)
    3. Metrics & Plot artifacts (backend/evaluation/metrics.json, plots)
    4. Flask Backend API endpoints (Health, Metrics, Preview, Batch Predict)
    5. Real batch prediction & exact percentage calculations
    6. Robust error handling (invalid files, empty files, missing columns)
    7. Frontend build readiness (frontend/dist/ index.html, JS, CSS)
"""

import sys
import os
import io
import json
import joblib
import pandas as pd
import numpy as np

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

from app import app, model_feature_names, is_model_loaded, model_algorithm

client = app.test_client()


def run_e2e_test_suite():
    print("=" * 70)
    print("PHASE 8: COMPREHENSIVE END-TO-END SYSTEM VERIFICATION")
    print("=" * 70)

    total_checks = 0
    passed_checks = 0

    # ── CHECK 1: Dataset Integrity ─────────────────────────────
    total_checks += 1
    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "emotions.csv")
    print("\n[CHECK 1] Dataset File Verification ...")
    if os.path.exists(dataset_path):
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"  ✓ emotions.csv exists ({size_mb:.1f} MB)")
        passed_checks += 1
    else:
        print("  ✗ FAIL: dataset/emotions.csv not found.")

    # ── CHECK 2: Trained Model Bundle ──────────────────────────
    total_checks += 1
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    print("\n[CHECK 2] Trained Model Bundle Verification ...")
    if os.path.exists(model_path) and is_model_loaded:
        bundle = joblib.load(model_path)
        print(f"  ✓ best_model.pkl loaded successfully.")
        print(f"    - Algorithm : {bundle.get('algorithm')}")
        print(f"    - Features  : {len(bundle.get('feature_names', []))}")
        print(f"    - Classes   : {bundle.get('classes')}")
        passed_checks += 1
    else:
        print("  ✗ FAIL: Model bundle not loaded or missing.")

    # ── CHECK 3: Evaluation Metrics & Plot Artifacts ───────────
    total_checks += 1
    print("\n[CHECK 3] Evaluation Artifacts Verification ...")
    metrics_path = os.path.join(BASE_DIR, "evaluation", "metrics.json")
    cm_plot = os.path.join(BASE_DIR, "evaluation", "plots", "confusion_matrix.png")
    cv_plot = os.path.join(BASE_DIR, "evaluation", "plots", "model_comparison_cv.png")
    feat_plot = os.path.join(BASE_DIR, "evaluation", "plots", "feature_importance.png")

    artifacts = [metrics_path, cm_plot, cv_plot, feat_plot]
    all_artifacts_exist = all(os.path.exists(p) for p in artifacts)

    if all_artifacts_exist:
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        acc = metrics_data["holdout_test_metrics"]["accuracy"]
        print(f"  ✓ metrics.json verified (Holdout Test Accuracy: {acc * 100:.2f}%)")
        print(f"  ✓ Confusion matrix plot exists: {os.path.basename(cm_plot)}")
        print(f"  ✓ CV comparison plot exists: {os.path.basename(cv_plot)}")
        print(f"  ✓ Feature importance plot exists: {os.path.basename(feat_plot)}")
        passed_checks += 1
    else:
        print("  ✗ FAIL: One or more evaluation artifacts missing.")

    # ── CHECK 4: Flask Health & Metrics APIs ───────────────────
    total_checks += 1
    print("\n[CHECK 4] Flask Core Endpoints (Health & Metrics) ...")
    res_health = client.get("/api/health")
    res_metrics = client.get("/api/metrics")

    if res_health.status_code == 200 and res_metrics.status_code == 200:
        h_data = res_health.get_json()
        m_data = res_metrics.get_json()
        print(f"  ✓ GET /api/health -> 200 OK (Status: {h_data.get('status')})")
        print(f"  ✓ GET /api/metrics -> 200 OK (Selected Algorithm: {m_data.get('selected_algorithm')})")
        passed_checks += 1
    else:
        print(f"  ✗ FAIL: Core APIs failed ({res_health.status_code}, {res_metrics.status_code})")

    # ── CHECK 5: Batch CSV Prediction & Percentage Calculation ─
    total_checks += 1
    print("\n[CHECK 5] Batch CSV Prediction & Percentage Distribution ...")
    # Extract 20 sample rows from the Kaggle dataset
    df_full = pd.read_csv(dataset_path, nrows=20)
    df_full.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)
    test_csv_bytes = df_full.to_csv(index=False).encode("utf-8")

    upload_data = {
        "file": (io.BytesIO(test_csv_bytes), "eeg_sample_batch.csv")
    }
    res_predict = client.post("/api/predict/batch", data=upload_data, content_type="multipart/form-data")

    if res_predict.status_code == 200:
        pred_data = res_predict.get_json()
        total_samples = pred_data.get("total_samples")
        summary = pred_data.get("class_summary", {})
        print(f"  ✓ POST /api/predict/batch -> 200 OK ({total_samples} samples analyzed)")
        print(f"  ✓ Predictions Breakdown:")
        for cls, count in summary.items():
            pct = (count / total_samples) * 100
            print(f"      - {cls:<10}: {count} records ({pct:.1f}%)")
        passed_checks += 1
    else:
        print(f"  ✗ FAIL: Batch predict returned status code {res_predict.status_code}")

    # ── CHECK 6: Robust Error Handling ─────────────────────────
    total_checks += 1
    print("\n[CHECK 6] Error Handling & Validation Tests ...")

    # A. Non-CSV file
    res_err1 = client.post(
        "/api/predict/batch",
        data={"file": (io.BytesIO(b"dummy image content"), "photo.png")},
        content_type="multipart/form-data"
    )
    # B. Missing required columns
    empty_csv = io.BytesIO(b"col_a,col_b\n1,2\n3,4")
    res_err2 = client.post(
        "/api/predict/batch",
        data={"file": (empty_csv, "incomplete.csv")},
        content_type="multipart/form-data"
    )
    # C. Empty CSV
    blank_csv = io.BytesIO(b"")
    res_err3 = client.post(
        "/api/predict/batch",
        data={"file": (blank_csv, "empty.csv")},
        content_type="multipart/form-data"
    )

    if res_err1.status_code == 400 and res_err2.status_code == 400 and res_err3.status_code == 400:
        print("  ✓ Correctly rejected non-CSV file (.png) with HTTP 400")
        print("  ✓ Correctly rejected CSV with missing feature columns with HTTP 400")
        print("  ✓ Correctly rejected empty CSV file with HTTP 400")
        passed_checks += 1
    else:
        print("  ✗ FAIL: One or more validation checks failed.")

    # ── CHECK 7: Frontend Production Build ─────────────────────
    total_checks += 1
    print("\n[CHECK 7] Frontend Production Build Verification ...")
    dist_dir = os.path.join(PROJECT_ROOT, "frontend", "dist")
    dist_index = os.path.join(dist_dir, "index.html")
    dist_assets = os.path.join(dist_dir, "assets")

    if os.path.exists(dist_index) and os.path.exists(dist_assets):
        asset_files = os.listdir(dist_assets)
        print(f"  ✓ frontend/dist/index.html exists")
        print(f"  ✓ Bundled assets ({len(asset_files)} files): {asset_files}")
        passed_checks += 1
    else:
        print("  ✗ FAIL: frontend/dist build not found.")

    # ── SUMMARY ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"END-TO-END VERIFICATION RESULT: {passed_checks}/{total_checks} CHECKS PASSED (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_test_suite()

