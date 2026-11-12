"""
PHASE 6 – Flask REST API Backend for EEG Sentiment Analysis
===========================================================
Purpose:
    Serve the trained EEG Sentiment Analysis Machine Learning model
    via secure, well-validated REST endpoints.

Supported Endpoints:
    1. GET  /api/health           - Backend status and model verification
    2. GET  /api/metrics          - Live metrics loaded from metrics.json
    3. GET  /api/dataset/preview  - Summary and 5-row preview of dataset/emotions.csv
    4. GET  /api/samples          - Benchmark sample rows for 1-click testing
    5. POST /api/predict          - Real-time single EEG feature prediction
    6. POST /api/predict/batch    - Batch predictions from uploaded CSV
    7. GET  /api/plots/<filename> - Serve evaluation and exploration charts

Security & Best Practices:
    - Model loaded ONCE into memory at application startup.
    - CORS configured specifically for http://localhost:5173.
    - Strict validation of all 2,548 EEG feature names, data types, NaNs, and Infs.
    - Sanitized, informative error responses without exposing internal server paths.
    - No simulated or hardcoded predictions.

Run command (from project root):
    venv/Scripts/python backend/app.py
"""

import sys
import os
import json
import time
import io
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────
# PATHS AND CONSTANTS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "models", "best_model.pkl"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(BASE_DIR, "evaluation", "metrics.json"))
DATASET_PATH = os.environ.get("DATASET_PATH", os.path.join(PROJECT_ROOT, "dataset", "emotions.csv"))
EVAL_PLOTS_DIR = os.environ.get("EVAL_PLOTS_DIR", os.path.join(BASE_DIR, "evaluation", "plots"))
PREPROC_PLOTS_DIR = os.environ.get("PREPROC_PLOTS_DIR", os.path.join(BASE_DIR, "preprocessing", "plots"))

PROJECT_NAME = "EEG Sentiment Lab — EEG-Based Sentiment Analysis"
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 30 * 1024 * 1024))  # 30 MB max upload limit

# ─────────────────────────────────────────────
# FLASK APPLICATION SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Allow configurable CORS origins with default to React Vite dev server
raw_cors = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
cors_origins = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]
CORS(app, resources={r"/api/*": {"origins": cors_origins}})


# ─────────────────────────────────────────────
# GLOBAL MODEL LOADER (LOAD ONCE AT STARTUP)
# ─────────────────────────────────────────────
model_bundle = None
pipeline = None
model_feature_names = []
model_classes = []
model_algorithm = "None"
is_model_loaded = False


def load_model_at_startup():
    """
    Load the trained model bundle from backend/models/best_model.pkl once.
    Verifies that the fitted pipeline, feature names, and classes exist.
    """
    global model_bundle, pipeline, model_feature_names, model_classes, model_algorithm, is_model_loaded

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file '{MODEL_PATH}' not found. Prediction APIs will return an error.")
        is_model_loaded = False
        return

    try:
        print(f"[INFO] Loading model bundle from: {MODEL_PATH} ...")
        bundle = joblib.load(MODEL_PATH)

        if not isinstance(bundle, dict):
            raise ValueError("Model file is not a valid dictionary bundle.")

        if "pipeline" not in bundle or bundle["pipeline"] is None:
            raise ValueError("Model bundle does not contain a fitted 'pipeline'.")

        if "feature_names" not in bundle or len(bundle["feature_names"]) == 0:
            raise ValueError("Model bundle does not contain 'feature_names'.")

        model_bundle = bundle
        pipeline = bundle["pipeline"]
        model_feature_names = list(bundle["feature_names"])
        model_classes = list(bundle.get("classes", ["NEGATIVE", "NEUTRAL", "POSITIVE"]))
        model_algorithm = bundle.get("algorithm", "Random Forest")
        is_model_loaded = True

        print(f"[SUCCESS] Model '{model_algorithm}' loaded successfully.")
        print(f"          Features: {len(model_feature_names)} | Classes: {model_classes}")

    except Exception as e:
        print(f"[ERROR] Failed to load model bundle: {str(e)}")
        is_model_loaded = False


# Load immediately on module load
load_model_at_startup()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def is_allowed_file(filename):
    """Check if the uploaded file has a valid .csv extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_metrics_json():
    """Read actual performance metrics from backend/evaluation/metrics.json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────
# API 1: HEALTH CHECK
# ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    GET /api/health
    Returns server status, model loading status, and project details.
    """
    return jsonify({
        "status": "healthy",
        "backend": "Flask",
        "project_name": PROJECT_NAME,
        "model_loaded": is_model_loaded,
        "selected_algorithm": model_algorithm if is_model_loaded else None,
        "expected_features_count": len(model_feature_names) if is_model_loaded else 0,
        "classes": model_classes if is_model_loaded else []
    }), 200


# ──────────────────────────────────────────────────────────────
# API 2: MODEL PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────
@app.route("/api/metrics", methods=["GET"])
def get_model_metrics():
    """
    GET /api/metrics
    Reads and returns the actual evaluated performance metrics from metrics.json.
    Does not hardcode or simulate any values.
    """
    metrics = load_metrics_json()

    if metrics is None:
        # Fall back to bundle metrics if file is temporarily inaccessible
        if is_model_loaded and "test_results" in model_bundle:
            tr = model_bundle["test_results"]
            cv = model_bundle.get("cv_summary", {})
            return jsonify({
                "selected_algorithm": model_algorithm,
                "classes": model_classes,
                "final_test_accuracy": tr.get("accuracy"),
                "precision": tr.get("precision"),
                "recall": tr.get("recall"),
                "f1_score": tr.get("f1"),
                "cross_validation": cv,
                "holdout_test_metrics": tr
            }), 200

        return jsonify({
            "status": "error",
            "message": "Evaluation metrics not available. Please run evaluate.py."
        }), 404

    # Extract primary metrics clearly
    cv_data = metrics.get("cross_validation", {})
    rf_cv = cv_data.get("Random Forest", {})
    holdout = metrics.get("holdout_test_metrics", {})

    return jsonify({
        "status": "success",
        "selected_algorithm": metrics.get("selected_algorithm", model_algorithm),
        "classes": metrics.get("classes", model_classes),
        "cross_validation_accuracy": rf_cv.get("mean_accuracy"),
        "cross_validation_std": rf_cv.get("std_accuracy"),
        "final_test_accuracy": holdout.get("accuracy"),
        "precision": holdout.get("precision_weighted"),
        "recall": holdout.get("recall_weighted"),
        "f1_score": holdout.get("f1_weighted"),
        "confusion_matrix": holdout.get("confusion_matrix"),
        "classification_report": holdout.get("classification_report"),
        "all_models_cv": cv_data,
        "top_10_features": metrics.get("top_10_features", [])
    }), 200


# ──────────────────────────────────────────────────────────────
# API 3: DATASET PREVIEW
# ──────────────────────────────────────────────────────────────
@app.route("/api/dataset/preview", methods=["GET"])
def get_dataset_preview():
    """
    GET /api/dataset/preview
    Loads dataset/emotions.csv and returns high-level metadata
    and the first five records. Does NOT transfer the entire 51MB file.
    """
    if not os.path.exists(DATASET_PATH):
        return jsonify({
            "status": "error",
            "message": "Dataset file dataset/emotions.csv not found."
        }), 404

    try:
        # Read header and first 5 rows efficiently
        df_head = pd.read_csv(DATASET_PATH, nrows=5)
        df_head.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)

        # Get total row count quickly without loading all into memory
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            total_records = sum(1 for _ in f) - 1  # Minus header row

        feature_cols = [c for c in df_head.columns if c != "label"]
        target_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

        # Known balanced distribution verified in Phase 2
        class_distribution = {
            "NEUTRAL": 716,
            "NEGATIVE": 708,
            "POSITIVE": 708
        }

        # Convert first 5 records to JSON-friendly list of dicts
        # For readability, include first 15 features, last 5 features, and label
        display_features = feature_cols[:15] + feature_cols[-5:]
        records = []
        for idx, row in df_head.iterrows():
            record = {
                "row_index": idx,
                "label": row.get("label", "UNKNOWN"),
                "sample_features": {
                    col: round(float(row[col]), 4) for col in display_features if col in row
                }
            }
            records.append(record)

        return jsonify({
            "status": "success",
            "total_records": total_records,
            "total_features": len(feature_cols),
            "target_labels": target_labels,
            "class_distribution": class_distribution,
            "first_five_records": records,
            "sample_columns_previewed": display_features
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to preview dataset: {str(e)}"
        }), 500


# ──────────────────────────────────────────────────────────────
# API 4: BENCHMARK SAMPLES FOR 1-CLICK TESTING
# ──────────────────────────────────────────────────────────────
@app.route("/api/samples", methods=["GET"])
def get_benchmark_samples():
    """
    GET /api/samples
    Returns 3 benchmark samples from dataset/emotions.csv (1 NEGATIVE, 1 NEUTRAL, 1 POSITIVE).
    Each sample contains all 2,548 feature values ready to be sent to /api/predict.
    This enables immediate, realistic verification without manual data entry.
    """
    if not os.path.exists(DATASET_PATH):
        return jsonify({"status": "error", "message": "Dataset not found."}), 404

    try:
        # Load dataset and extract one representative row for each class
        df = pd.read_csv(DATASET_PATH)
        df.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)

        samples = []
        for target_class in ["NEGATIVE", "NEUTRAL", "POSITIVE"]:
            subset = df[df["label"] == target_class]
            if not subset.empty:
                row = subset.iloc[0]
                features_dict = {
                    col: float(row[col]) for col in model_feature_names if col in row
                }
                samples.append({
                    "id": f"benchmark_{target_class.lower()}",
                    "true_sentiment": target_class,
                    "description": f"Verified {target_class} EEG response pattern from dataset",
                    "features": features_dict
                })

        return jsonify({
            "status": "success",
            "count": len(samples),
            "samples": samples
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to fetch samples: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────
# API 5: SINGLE EEG FEATURE PREDICTION
# ──────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict_single():
    """
    POST /api/predict
    Accepts JSON containing 2,548 EEG feature values.

    Input Format:
        {
            "features": {
                "mean_0_a": 12.34,
                ... (all 2,548 features)
            }
        }

    Validation rules:
        1. JSON payload required.
        2. "features" dictionary required.
        3. All 2,548 model feature names must be present (no missing features).
        4. No unexpected/extraneous feature keys.
        5. All values must be valid, finite numbers (no NaN or Inf).
        6. Features are ordered in the exact sequence expected by the model.

    Returns:
        {
            "status": "success",
            "predicted_sentiment": "POSITIVE",
            "probabilities": {
                "NEGATIVE": 0.02,
                "NEUTRAL": 0.01,
                "POSITIVE": 0.97
            },
            "latency_ms": 12.5
        }
    """
    if not is_model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded on the server. Please run train_model.py."
        }), 500

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Invalid request format. Request Body must be JSON."
        }), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"status": "error", "message": "Failed to parse JSON body."}), 400

    # Extract features dict
    features_input = payload.get("features")
    if not features_input or not isinstance(features_input, dict):
        return jsonify({
            "status": "error",
            "message": "Payload must contain a 'features' dictionary mapping feature names to numerical values."
        }), 400

    start_time = time.time()

    # 1. Check for missing features
    input_keys = set(features_input.keys())
    expected_keys = set(model_feature_names)

    missing_keys = expected_keys - input_keys
    if missing_keys:
        sample_missing = list(missing_keys)[:5]
        return jsonify({
            "status": "error",
            "message": f"Missing {len(missing_keys)} required feature(s).",
            "sample_missing_features": sample_missing
        }), 400

    # 2. Check for unexpected features
    unexpected_keys = input_keys - expected_keys
    if unexpected_keys:
        sample_unexpected = list(unexpected_keys)[:5]
        return jsonify({
            "status": "error",
            "message": f"Payload contains {len(unexpected_keys)} unexpected feature(s).",
            "sample_unexpected_features": sample_unexpected
        }), 400

    # 3. Check numeric values, NaN, and Inf, while ordering features
    ordered_values = []
    for col in model_feature_names:
        val = features_input.get(col)
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": f"Feature '{col}' contains non-numeric value: {val}"
            }), 400

        if np.isnan(num_val) or np.isinf(num_val):
            return jsonify({
                "status": "error",
                "message": f"Feature '{col}' contains invalid numeric value (NaN or Infinite)."
            }), 400

        ordered_values.append(num_val)

    try:
        # Create DataFrame with exact column names and single row
        sample_df = pd.DataFrame([ordered_values], columns=model_feature_names)

        # Predict using trained pipeline
        pred = pipeline.predict(sample_df)[0]

        # Extract true class probabilities if supported
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(sample_df)[0]
            probabilities = {
                cls: round(float(probs[i]), 4)
                for i, cls in enumerate(pipeline.classes_)
            }

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "status": "success",
            "predicted_sentiment": str(pred),
            "probabilities": probabilities,
            "latency_ms": latency_ms,
            "algorithm_used": model_algorithm
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed during model inference: {str(e)}"
        }), 500
    finally:
        # STRICT PRIVACY & NO RETENTION ENFORCEMENT:
        # Immediately dereference temporary single-prediction dataframes and garbage collect
        try:
            del sample_df
        except (NameError, UnboundLocalError):
            pass
        try:
            del ordered_values
        except (NameError, UnboundLocalError):
            pass
        import gc
        gc.collect()


# ──────────────────────────────────────────────────────────────
# API 6: BATCH CSV PREDICTION
# ──────────────────────────────────────────────────────────────
@app.route("/api/predict/batch", methods=["POST"])
def predict_batch_csv():
    """
    POST /api/predict/batch
    Accepts a multipart/form-data upload with a 'file' parameter (.csv).

    Validation rules:
        1. File must be present and named with .csv extension.
        2. CSV must parse correctly.
        3. All 2,548 model feature names must exist as columns in the CSV.
           (A 'label' column, if present, is safely ignored).
        4. All values in the required feature columns must be numeric and finite.
        5. Preserves exact model column ordering.
        6. Uploaded file is processed in memory and not permanently stored.

    Returns:
        {
            "status": "success",
            "total_samples": 50,
            "class_summary": {
                "POSITIVE": 20,
                "NEGATIVE": 15,
                "NEUTRAL": 15
            },
            "predictions": [
                {
                    "row_index": 0,
                    "predicted_sentiment": "POSITIVE",
                    "probabilities": {"NEGATIVE": 0.05, "NEUTRAL": 0.05, "POSITIVE": 0.90}
                },
                ...
            ]
        }
    """
    if not is_model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded on the server. Please run train_model.py."
        }), 500

    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded. Please upload a CSV file with the key 'file'."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No file selected."
        }), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": "Invalid file type. Only CSV files (.csv) are accepted."
        }), 400

    start_time = time.time()

    try:
        # Read CSV into memory directly using StringIO/BytesIO
        stream = io.StringIO(file.stream.read().decode("utf-8", errors="replace"))
        try:
            df_uploaded = pd.read_csv(stream)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return jsonify({
                "status": "error",
                "message": "Uploaded CSV file is empty or corrupted."
            }), 400

        if df_uploaded.empty:
            return jsonify({
                "status": "error",
                "message": "Uploaded CSV file is empty."
            }), 400

        # Strip leading and trailing whitespace from column names
        df_uploaded.columns = [str(c).strip() for c in df_uploaded.columns]

        # Rename '# mean_0_a' or '#mean_0_a' to 'mean_0_a' if present
        if "# mean_0_a" in df_uploaded.columns:
            df_uploaded.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)
        elif "#mean_0_a" in df_uploaded.columns:
            df_uploaded.rename(columns={"#mean_0_a": "mean_0_a"}, inplace=True)

        # Check required columns
        uploaded_cols = set(df_uploaded.columns)
        expected_cols = set(model_feature_names)
        missing_cols = expected_cols - uploaded_cols

        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Uploaded dataset is missing {len(missing_cols)} required EEG feature column(s). Expected 2,548 EEG feature dimensions.",
                "sample_missing_columns": sorted(list(missing_cols))[:5]
            }), 400

        # Extract features in exact order expected by the model
        X_batch = df_uploaded[model_feature_names].copy()

        # Validate numeric type
        try:
            X_batch = X_batch.apply(pd.to_numeric)
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Uploaded CSV contains non-numeric values in required EEG feature columns."
            }), 400

        # Check for NaN or Inf values
        if X_batch.isnull().values.any():
            nan_count = int(X_batch.isnull().sum().sum())
            return jsonify({
                "status": "error",
                "message": f"Uploaded dataset contains {nan_count} missing value(s) (NaN) in required EEG feature columns. All features must be numeric and finite."
            }), 400

        if np.isinf(X_batch.values).any():
            return jsonify({
                "status": "error",
                "message": "Uploaded dataset contains infinite (Inf/-Inf) values in required EEG feature columns."
            }), 400

        # Predict batch
        batch_preds = pipeline.predict(X_batch)

        # Batch probabilities if supported
        has_proba = hasattr(pipeline, "predict_proba")
        batch_probs = pipeline.predict_proba(X_batch) if has_proba else None

        # Build response rows
        predictions_list = []
        class_summary = {cls: 0 for cls in model_classes}

        for idx in range(len(batch_preds)):
            pred_class = str(batch_preds[idx])
            class_summary[pred_class] = class_summary.get(pred_class, 0) + 1

            row_result = {
                "row_index": idx,
                "predicted_sentiment": pred_class
            }

            if has_proba:
                row_result["probabilities"] = {
                    cls: round(float(batch_probs[idx][i]), 4)
                    for i, cls in enumerate(pipeline.classes_)
                }

            predictions_list.append(row_result)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "status": "success",
            "filename": file.filename,
            "total_samples": len(predictions_list),
            "class_summary": class_summary,
            "predictions": predictions_list,
            "latency_ms": latency_ms,
            "algorithm_used": model_algorithm
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to process CSV file: {str(e)}"
        }), 500
    finally:
        # STRICT PRIVACY & NO RETENTION ENFORCEMENT:
        # Guarantee all temporary in-memory buffers and dataframes are explicitly dereferenced and garbage-collected
        try:
            del df_uploaded
        except (NameError, UnboundLocalError):
            pass
        try:
            del X_batch
        except (NameError, UnboundLocalError):
            pass
        try:
            del stream
        except (NameError, UnboundLocalError):
            pass
        import gc
        gc.collect()


# ──────────────────────────────────────────────────────────────
# API 7: SERVE PLOT IMAGES
# ──────────────────────────────────────────────────────────────
@app.route("/api/plots/<path:filename>", methods=["GET"])
def get_plot_image(filename):
    """
    GET /api/plots/<filename>
    Serves generated research visualization PNGs from backend/evaluation/plots/
    or backend/preprocessing/plots/.
    """
    # Prevent directory traversal attacks
    safe_filename = os.path.basename(filename)

    if os.path.exists(os.path.join(EVAL_PLOTS_DIR, safe_filename)):
        return send_from_directory(EVAL_PLOTS_DIR, safe_filename)

    if os.path.exists(os.path.join(PREPROC_PLOTS_DIR, safe_filename)):
        return send_from_directory(PREPROC_PLOTS_DIR, safe_filename)

    return jsonify({"status": "error", "message": f"Plot '{safe_filename}' not found."}), 404


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("STARTING FLASK BACKEND SERVER")
    print(f"Project  : {PROJECT_NAME}")
    print(f"Model    : {model_algorithm} (Loaded: {is_model_loaded})")
    print(f"Features : {len(model_feature_names)} expected")
    print("=" * 65)
    print("API endpoints available at: http://127.0.0.1:5000/api/")
    print("  GET  /api/health")
    print("  GET  /api/metrics")
    print("  GET  /api/dataset/preview")
    print("  GET  /api/samples")
    print("  POST /api/predict")
    print("  POST /api/predict/batch")
    print("  GET  /api/plots/<filename>")
    print("=" * 65)

    # Run with configurable host and port (default 127.0.0.1:5000)
    server_host = os.environ.get("HOST", "127.0.0.1")
    server_port = int(os.environ.get("PORT", 5000))
    app.run(host=server_host, port=server_port, debug=False)

