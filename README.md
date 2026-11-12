# EEG-Based Sentiment Analysis Using Machine Learning

**B.Tech Minor Academic Project** | Computer Science & Engineering | Affective Computing

An end-to-end machine learning research application that analyzes multi-channel EEG frequency-domain features and classifies affective sentiment into **Positive**, **Neutral**, and **Negative** states with **98.83% holdout test accuracy**.

---

## 📌 Project Overview

This project implements a reproducible affective computing pipeline to classify human sentiment from electroencephalogram (EEG) brainwave data. Using statistical features extracted from EEG time-series recordings (Fast Fourier Transform frequency bands, alpha/beta/gamma powers, statistical moments), the system applies an optimized Random Forest classifier to predict emotional states with high precision and low latency.

The web application provides a distraction-free academic research dashboard and an intuitive 3-step progressive workflow for uploading EEG datasets, inspecting column integrity, running real-time batch inference, visualizing sentiment distribution, and exporting results—with strict in-memory data privacy.

---

## 🌟 Main Features

- **Trained Machine Learning Model:** Pre-trained Random Forest ensemble evaluated via 5-Fold Stratified Cross-Validation ($98.30\% \pm 1.16\%$) and tested on an untouched holdout test partition ($98.83\%$ accuracy).
- **Minimalist Academic UI:** Designed using the official Drone.io-inspired light color palette (`#E6F2FC`, `#00A9D6`, `#203B5D`, `#FFFFFF`), clean typography, and zero distracting commercial SaaS elements.
- **3-Step Progressive Workflow:**
  1. *Dataset Upload:* Drag-and-drop or file-picker CSV upload.
  2. *Dataset Validation & Preview:* Instant client-side schema inspection (row count, feature count, missing value audit) with a 5-record table preview.
  3. *Results & Visual Analytics:* Real-time inference results, dynamic narrative summary, sentiment percentage donut chart, confidence breakdown, and paginated records table.
- **Strict Data Privacy:** Batch datasets are processed strictly in-memory (`io.StringIO`, `pd.read_csv`, and explicit garbage collection). User-uploaded files are never written to disk or database.
- **Client-Side CSV Export:** One-click instant CSV generation and download directly in the browser.
- **Responsive Layout:** Collapsible desktop sidebar (`w-64` $\leftrightarrow$ `w-20`) with dynamic content margins, and a mobile-friendly slide-over drawer with backdrop blur.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | React 18 + TypeScript 5.7 | Component-driven user interface |
| **Build Tool** | Vite 6 | Development server and production bundling |
| **Styling** | Tailwind CSS 3.4 | Utility-first responsive design system |
| **Charts** | Recharts 2.15 | Responsive SVG Donut chart visualization |
| **Icons** | Lucide React | Minimalist UI iconography |
| **Backend API** | Flask 3.1.1 + Flask-CORS 5.0.1 | Lightweight RESTful microservice |
| **Machine Learning** | Scikit-Learn 1.6.1 | Model training, cross-validation, and pipeline |
| **Data Processing**| Pandas 2.2.3 + NumPy 2.2.6 | Feature manipulation and batch tensor operations |
| **Model Persistence**| Joblib 1.5.1 | Serialization of fitted pipeline bundle |
| **QA Automation**| Puppeteer Core 25.17 | Automated headless browser verification suite |

---

## 🏛️ System Architecture

```
[ EEG CSV Dataset (2,132 × 2,548 features) ]
                  │
                  ▼
[ Exploratory Data Analysis & Integrity Verification ]
  (Zero duplicate rows, zero NaN/inf values, zero data leakage)
                  │
                  ▼
[ Stratified 80/20 Train-Test Split ]
  ├── Training Partition (n = 1,705 samples)
  └── Holdout Test Partition (n = 427 samples - LOCKED)
                  │
                  ▼
[ 5-Fold Stratified Cross-Validation Benchmark ]
  ├── Logistic Regression (StandardScaler -> LogisticRegression) : 78.48% CV
  ├── Support Vector Machine (StandardScaler -> SVC RBF)         : 95.89% CV
  └── Random Forest Classifier (n_estimators=100)               : 98.30% CV ──► [ WINNER ]
                  │
                  ▼
[ Final Fit & Holdout Test Set Evaluation (98.83% Accuracy) ]
                  │
                  ▼
[ Export: best_model.pkl & metrics.json ]
                  │
         ┌────────┴──────────────────────────┐
         ▼                                   ▼
[ Flask REST API Backend ]          [ React + Vite + TypeScript Frontend ]
(Port 5000: /api/predict/batch)      (Port 5173: Drone.io Light Design System)
```

---

## 🧠 Machine Learning Model Information

The classifier is trained on the standard benchmark EEG Emotion dataset (Kaggle), consisting of 2,132 records with 2,548 statistical features derived from 14-channel EEG recordings:

- **Target Affective Classes:**
  - `POSITIVE` (Joy, Enthusiasm, Pleasantness)
  - `NEUTRAL` (Resting, Baseline cognitive state)
  - `NEGATIVE` (Stress, Dislike, Anxiety)
- **Algorithm:** Random Forest Classifier (`RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42)`)
- **Cross-Validation Performance (Training Set $n=1,705$):**
  - Mean 5-Fold CV Accuracy: **98.30%** ($\pm 1.16\%$)
  - Mean Macro F1-Score: **0.9830**
- **Holdout Test Set Performance (Test Set $n=427$):**
  - **Accuracy:** **98.83%** (422/427 correct predictions)
  - **Macro Precision:** **98.85%**
  - **Macro Recall:** **98.82%**
  - **Macro F1-Score:** **98.83%**
  - **Per-Class Metrics:**
    - `NEGATIVE`: Precision: 99.29% | Recall: 99.29% | F1: 99.29%
    - `NEUTRAL`:  Precision: 100.0% | Recall: 97.87% | F1: 98.92%
    - `POSITIVE`: Precision: 97.26% | Recall: 99.30% | F1: 98.27%

---

## 📊 Dataset Format Requirements

When uploading an EEG CSV dataset for prediction:

1. **File Format:** Standard Comma-Separated Values (`.csv`).
2. **Feature Dimensions:** Must contain exactly **2,548 numerical feature columns** matching the model's feature names (`mean_0_a`, `mean_1_a`, ..., `std_49_b`).
3. **Target Column (Optional):** If a `label` column is present in the CSV, it is safely excluded from inference automatically.
4. **Data Integrity:** No missing (`NaN`, `null`, `None`) or infinite values are permitted; all cell values must be valid floating-point numbers.
5. **Sample Datasets Included:**
   - `dataset/test_sample_20_rows.csv`: 20-record valid dataset for immediate testing.
   - `dataset/test_invalid_columns.csv`: 2-column invalid dataset for testing schema validation error handling.

---

## 📁 Clean Project Directory Structure

```
EEG-Sentiment-Analysis/
│
├── dataset/
│   ├── emotions.csv                  # Complete Kaggle EEG dataset (2,132 × 2,549)
│   ├── test_sample_20_rows.csv       # Verified 20-row test dataset for 1-click evaluation
│   └── test_invalid_columns.csv      # Schema mismatch test dataset (error handling test)
│
├── backend/
│   ├── preprocessing/
│   │   ├── explore.py                # Exploratory data analysis & visualization
│   │   ├── preprocess.py             # Feature separation, schema validation & train/test split
│   │   └── plots/                    # Saved EDA visualization plots
│   ├── models/
│   │   ├── train_model.py            # Canonical training & 5-fold CV pipeline
│   │   ├── train_and_optimize.py     # Hyperparameter optimization & benchmark suite
│   │   └── best_model.pkl            # Production trained Random Forest model bundle
│   ├── evaluation/
│   │   ├── evaluate.py               # Model evaluation script
│   │   ├── metrics.json              # Serialized holdout test metrics & CV scores
│   │   └── plots/                    # Confusion matrix, feature importance & CV plots
│   ├── app.py                        # Production Flask REST API server
│   ├── requirements.txt              # Backend-specific Python dependencies
│   ├── .env.example                  # Backend environment configuration template
│   ├── test_backend.py               # 11-endpoint automated unit test suite
│   └── test_e2e_workflow.py          # End-to-end integration verification suite
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Sidebar.tsx           # Collapsible sidebar with mobile navigation drawer
│   │   ├── pages/
│   │   │   ├── DashboardPage.tsx     # Minimalist academic research dashboard
│   │   │   └── AnalysisPage.tsx      # 3-step progressive upload, preview & results view
│   │   ├── services/
│   │   │   └── api.ts                # Axios HTTP client configured for Flask backend
│   │   ├── types/
│   │   │   └── index.ts              # TypeScript interfaces for API contracts
│   │   ├── App.tsx                   # Main application root component
│   │   ├── main.tsx                  # Application entry point
│   │   ├── index.css                 # Global styling, custom scrollbars & fonts
│   │   └── vite-env.d.ts             # Vite environment typings
│   ├── test_qa_suite.mjs             # Chrome Puppeteer automated browser QA suite
│   ├── .env.example                  # Frontend environment configuration template
│   ├── index.html                    # HTML document entry
│   ├── package.json                  # NPM packages & build scripts
│   ├── tailwind.config.js            # Tailwind theme colors & extensions
│   ├── postcss.config.js             # PostCSS Tailwind plugin config
│   ├── tsconfig.json                 # TypeScript compiler configuration
│   └── vite.config.ts                # Vite dev server & proxy settings
│
├── qa_screenshots/                   # Automated Chrome QA verification screenshots
├── qa_test_report.json               # Full JSON test report (8/8 browser tests passed)
├── test_verification.py              # Live server integration & data privacy test
├── run_app.bat                       # Windows 1-click launcher (backend + frontend + browser)
├── run_tests.bat                     # Windows 1-click test suite launcher
├── requirements.txt                  # Python dependencies manifest
├── .env.example                      # Root environment configuration template
└── README.md                         # Complete project documentation
```

---

## 🔌 API Endpoints Reference

The Flask backend exposes the following RESTful API endpoints at `http://127.0.0.1:5000/api`:

| Method | Endpoint | Description | Response Details |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Backend status & model readiness | `status`, `model_loaded`, `algorithm`, `classes` |
| `GET` | `/api/metrics` | Model evaluation statistics | CV accuracy, holdout test metrics, classification report |
| `GET` | `/api/dataset/preview` | Baseline dataset overview | `total_records`, `total_features`, first 5 records |
| `GET` | `/api/samples` | Pre-extracted benchmark EEG samples | 3 sample feature vectors (`NEGATIVE`, `NEUTRAL`, `POSITIVE`) |
| `POST`| `/api/predict` | Single-sample inference | Feature dictionary $\rightarrow$ predicted sentiment & probabilities |
| `POST`| `/api/predict/batch` | Batch CSV upload inference | `multipart/form-data` with `file` $\rightarrow$ per-row predictions |
| `GET` | `/api/plots/<file>` | Serves generated evaluation plots | Returns `confusion_matrix.png`, `feature_importance.png`, etc. |

---

## 🚀 Setup & Execution Guide

### Prerequisites
- **Python:** 3.10 to 3.13
- **Node.js:** 18.x or higher with npm

### 1. Installation

**Backend Virtual Environment:**
```powershell
# In project root
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend Dependencies:**
```powershell
cd frontend
npm install
cd ..
```

### 2. Configuration (Optional)
Copy `.env.example` to `.env` if you need custom ports or hostnames:
```powershell
copy .env.example .env
```

### 3. Running the Application

**Option A: One-Click Launcher (Windows)**
Simply double-click `run_app.bat` or run:
```powershell
.\run_app.bat
```
*This starts the Flask backend, launches the Vite frontend dev server, and opens `http://localhost:5173` in Google Chrome.*

**Option B: Manual Terminal Execution**

*Terminal 1 (Backend):*
```powershell
venv\Scripts\python backend\app.py
```
> Serving on `http://127.0.0.1:5000`

*Terminal 2 (Frontend):*
```powershell
cd frontend
npm run dev
```
> Serving on `http://localhost:5173`

---

## 🧪 Comprehensive Automated Testing

To run the complete automated test suite:
```powershell
.\run_tests.bat
```

Or execute individual test suites:

1. **Backend API Unit Tests (11/11 Passed):**
   ```powershell
   venv\Scripts\python backend\test_backend.py
   ```
2. **End-to-End Workflow Verification (7/7 Checks Passed):**
   ```powershell
   venv\Scripts\python backend\test_e2e_workflow.py
   ```
3. **Live Server Integration & Privacy Verification (4/4 Passed):**
   ```powershell
   venv\Scripts\python test_verification.py
   ```
4. **Chrome Automated Browser QA Suite (8/8 Passed):**
   ```powershell
   cd frontend
   node test_qa_suite.mjs
   ```
5. **Frontend Production TypeScript & Asset Build Check:**
   ```powershell
   cd frontend
   npm run build
   ```

---

## 🔒 Data Privacy & Security

In compliance with academic research privacy requirements:
- Uploaded batch datasets are parsed strictly in-memory into temporary DataFrame structures.
- Prediction tensors are generated in transient memory and dereferenced immediately.
- Zero user data, uploaded CSV files, or prediction records are cached, logged, or saved to the filesystem or any database.
