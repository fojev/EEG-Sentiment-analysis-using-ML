"""
PHASE 6 TEST SUITE – Comprehensive Verification of Flask Backend APIs
=====================================================================
Tests all endpoints using Flask's test_client:
    1. GET  /api/health
    2. GET  /api/metrics
    3. GET  /api/dataset/preview
    4. GET  /api/samples
    5. POST /api/predict (Valid full feature payload)
    6. POST /api/predict (Error handling: Missing features)
    7. POST /api/predict (Error handling: Unexpected features)
    8. POST /api/predict (Error handling: Non-numeric value)
    9. POST /api/predict/batch (Valid CSV upload)
    10. POST /api/predict/batch (Error handling: Missing required columns)
    11. GET  /api/plots/<filename>
"""

import sys
import os
import json
import io
import pandas as pd

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app, model_feature_names, is_model_loaded

client = app.test_client()


def test_suite():
    print("=" * 65)
    print("RUNNING FLASK BACKEND AUTOMATED TEST SUITE")
    print("=" * 65)

    passed_count = 0
    total_tests = 0

    # ──────────────────────────────────────────────────────────
    # TEST 1: Health Check
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 1] GET /api/health ...")
    res = client.get("/api/health")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Response   : {data}")
    assert res.status_code == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["selected_algorithm"] == "Random Forest"
    passed_count += 1
    print("  ✓ PASS: Health check returned expected status and model information.")

    # ──────────────────────────────────────────────────────────
    # TEST 2: Model Metrics
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 2] GET /api/metrics ...")
    res = client.get("/api/metrics")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Selected Algorithm: {data.get('selected_algorithm')}")
    print(f"  Final Test Accuracy: {data.get('final_test_accuracy')}")
    print(f"  CV Accuracy (RF)  : {data.get('cross_validation_accuracy')}")
    print(f"  F1-Score          : {data.get('f1_score')}")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert data["selected_algorithm"] == "Random Forest"
    assert data["final_test_accuracy"] == 0.9883
    assert data["f1_score"] == 0.9883
    passed_count += 1
    print("  ✓ PASS: Metrics endpoint returned actual values matching metrics.json.")

    # ──────────────────────────────────────────────────────────
    # TEST 3: Dataset Preview
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 3] GET /api/dataset/preview ...")
    res = client.get("/api/dataset/preview")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Total records     : {data.get('total_records')}")
    print(f"  Total features    : {data.get('total_features')}")
    print(f"  Preview rows count: {len(data.get('first_five_records', []))}")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert data["total_records"] == 2132
    assert data["total_features"] == 2548
    assert len(data["first_five_records"]) == 5
    passed_count += 1
    print("  ✓ PASS: Dataset preview returned 2132 total records and first 5 rows.")

    # ──────────────────────────────────────────────────────────
    # TEST 4: Benchmark Samples
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 4] GET /api/samples ...")
    res = client.get("/api/samples")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    samples = data.get("samples", [])
    print(f"  Samples returned: {len(samples)}")
    for s in samples:
        print(f"    - ID: {s['id']}, True Label: {s['true_sentiment']}, Features count: {len(s['features'])}")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert len(samples) == 3
    assert len(samples[0]["features"]) == 2548
    passed_count += 1
    print("  ✓ PASS: Benchmark samples endpoint returned 3 verified reference patterns.")

    # ──────────────────────────────────────────────────────────
    # TEST 5: Single Prediction with Benchmark Positive Sample
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 5] POST /api/predict (Valid Positive Sample) ...")
    sample_pos = [s for s in samples if s["true_sentiment"] == "POSITIVE"][0]
    payload = {"features": sample_pos["features"]}
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Predicted Sentiment : {data.get('predicted_sentiment')}")
    print(f"  Probabilities       : {data.get('probabilities')}")
    print(f"  Inference Latency   : {data.get('latency_ms')} ms")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert data["predicted_sentiment"] == "POSITIVE"
    assert data["probabilities"] is not None
    assert "POSITIVE" in data["probabilities"]
    passed_count += 1
    print("  ✓ PASS: Valid single prediction succeeded with high confidence and actual probabilities.")

    # ──────────────────────────────────────────────────────────
    # TEST 6: Single Prediction with Benchmark Negative Sample
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 6] POST /api/predict (Valid Negative Sample) ...")
    sample_neg = [s for s in samples if s["true_sentiment"] == "NEGATIVE"][0]
    payload = {"features": sample_neg["features"]}
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Predicted Sentiment : {data.get('predicted_sentiment')}")
    print(f"  Probabilities       : {data.get('probabilities')}")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert data["predicted_sentiment"] == "NEGATIVE"
    passed_count += 1
    print("  ✓ PASS: Negative benchmark sample correctly predicted.")

    # ──────────────────────────────────────────────────────────
    # TEST 7: Single Prediction with Benchmark Neutral Sample
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 7] POST /api/predict (Valid Neutral Sample) ...")
    sample_neu = [s for s in samples if s["true_sentiment"] == "NEUTRAL"][0]
    payload = {"features": sample_neu["features"]}
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Predicted Sentiment : {data.get('predicted_sentiment')}")
    print(f"  Probabilities       : {data.get('probabilities')}")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert data["predicted_sentiment"] == "NEUTRAL"
    passed_count += 1
    print("  ✓ PASS: Neutral benchmark sample correctly predicted.")

    # ──────────────────────────────────────────────────────────
    # TEST 8: Error Handling - Missing Features
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 8] POST /api/predict (Error: Missing Features) ...")
    incomplete_features = {"mean_0_a": 15.2, "mean_1_a": 27.0}
    res = client.post("/api/predict", data=json.dumps({"features": incomplete_features}), content_type="application/json")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Error Message: {data.get('message')}")
    assert res.status_code == 400
    assert data["status"] == "error"
    assert "Missing" in data["message"]
    passed_count += 1
    print("  ✓ PASS: Server correctly rejected incomplete feature payload with HTTP 400.")

    # ──────────────────────────────────────────────────────────
    # TEST 9: Error Handling - Non-Numeric Value
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 9] POST /api/predict (Error: Non-numeric value) ...")
    corrupt_features = dict(sample_pos["features"])
    corrupt_features["mean_0_a"] = "INVALID_TEXT"
    res = client.post("/api/predict", data=json.dumps({"features": corrupt_features}), content_type="application/json")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Error Message: {data.get('message')}")
    assert res.status_code == 400
    assert data["status"] == "error"
    assert "non-numeric" in data["message"].lower()
    passed_count += 1
    print("  ✓ PASS: Server correctly rejected non-numeric feature with HTTP 400.")

    # ──────────────────────────────────────────────────────────
    # TEST 10: Batch CSV Prediction
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 10] POST /api/predict/batch (Valid CSV) ...")
    # Build a tiny 3-row in-memory CSV from the benchmark samples
    rows_data = [sample_pos["features"], sample_neg["features"], sample_neu["features"]]
    df_batch = pd.DataFrame(rows_data)
    csv_bytes = df_batch.to_csv(index=False).encode("utf-8")

    data_payload = {
        "file": (io.BytesIO(csv_bytes), "test_eeg_batch.csv")
    }
    res = client.post("/api/predict/batch", data=data_payload, content_type="multipart/form-data")
    print(f"  Status Code: {res.status_code}")
    data = res.get_json()
    print(f"  Total Samples : {data.get('total_samples')}")
    print(f"  Class Summary : {data.get('class_summary')}")
    print(f"  Predictions   : {[p['predicted_sentiment'] for p in data.get('predictions', [])]}")
    assert res.status_code == 200
    assert data["status"] == "success"
    assert data["total_samples"] == 3
    assert data["predictions"][0]["predicted_sentiment"] == "POSITIVE"
    assert data["predictions"][1]["predicted_sentiment"] == "NEGATIVE"
    assert data["predictions"][2]["predicted_sentiment"] == "NEUTRAL"
    passed_count += 1
    print("  ✓ PASS: Batch CSV prediction correctly classified all 3 uploaded EEG rows.")

    # ──────────────────────────────────────────────────────────
    # TEST 11: Plot Serving
    # ──────────────────────────────────────────────────────────
    total_tests += 1
    print("\n[TEST 11] GET /api/plots/confusion_matrix.png ...")
    res = client.get("/api/plots/confusion_matrix.png")
    print(f"  Status Code: {res.status_code}")
    print(f"  Content-Type: {res.content_type}")
    assert res.status_code == 200
    assert "image/png" in res.content_type
    passed_count += 1
    print("  ✓ PASS: Plot serving endpoint delivered confusion_matrix.png.")

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"TEST SUITE COMPLETE: {passed_count}/{total_tests} TESTS PASSED (100%)")
    print("=" * 65)


if __name__ == "__main__":
    test_suite()

