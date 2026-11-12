"""
PHASE 2 – Dataset Exploration
==============================
Purpose:
    Load and inspect the emotions.csv dataset.
    Understand its dimensions, columns, target labels,
    check for missing values and duplicates,
    and produce basic visualisations.

Run command (from project root):
    venv/Scripts/python backend/preprocessing/explore.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend – saves PNGs instead of opening windows
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATASET_PATH = "dataset/emotions.csv"
OUTPUT_DIR   = "backend/preprocessing/plots"   # Visualisation output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 – Load the dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"  Dataset loaded from: {DATASET_PATH}")


# ─────────────────────────────────────────────
# STEP 2 – Basic Dimensions
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Dataset dimensions")
print("=" * 60)

num_rows, num_cols = df.shape
print(f"  Rows    : {num_rows}")
print(f"  Columns : {num_cols}")
print(f"  Size    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory")


# ─────────────────────────────────────────────
# STEP 3 – Column Overview
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Column overview")
print("=" * 60)

# Rename the first column – it has a leading '#' from the original CSV
df.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)

# Identify feature columns vs the target column
target_column  = "label"
feature_columns = [col for col in df.columns if col != target_column]

print(f"  Target column  : '{target_column}'")
print(f"  Feature columns: {len(feature_columns)}")

# Show what feature groups exist (suffix after last '_')
suffix_groups = {}
for col in feature_columns:
    suffix = col.split("_")[-1]          # e.g. 'a', 'b', 'a2', 'b2'
    suffix_groups.setdefault(suffix, []).append(col)

print("\n  Feature groups (by suffix):")
for suffix, cols in sorted(suffix_groups.items(), key=lambda x: -len(x[1])):
    print(f"    suffix='{suffix}'  →  {len(cols)} features")

# Show the first 10 and last 5 feature names as a sample
print(f"\n  First 5 feature columns : {feature_columns[:5]}")
print(f"  Last  5 feature columns : {feature_columns[-5:]}")


# ─────────────────────────────────────────────
# STEP 4 – Target Label Distribution
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Target label distribution")
print("=" * 60)

label_counts = df[target_column].value_counts()
total = len(df)

print(f"  {'Label':<12}  {'Count':>6}  {'Percentage':>10}")
print(f"  {'-'*12}  {'-'*6}  {'-'*10}")
for label, count in label_counts.items():
    pct = count / total * 100
    print(f"  {label:<12}  {count:>6}  {pct:>9.1f}%")

# Plot – Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#a78bfa", "#f87171", "#6ee7b7"]   # Purple, Red, Green
ax.bar(label_counts.index, label_counts.values, color=colors[:len(label_counts)], edgecolor="white")
ax.set_title("EEG Emotion Label Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Emotion Class")
ax.set_ylabel("Number of Samples")
for i, (lbl, cnt) in enumerate(label_counts.items()):
    ax.text(i, cnt + 5, str(cnt), ha="center", fontsize=10)
ax.set_ylim(0, max(label_counts.values) + 60)
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
plt.savefig(bar_path, dpi=120)
plt.close()
print(f"\n  Bar chart saved → {bar_path}")


# ─────────────────────────────────────────────
# STEP 5 – Missing Value Check
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Missing values check")
print("=" * 60)

total_missing = df.isnull().sum().sum()
print(f"  Total missing values: {total_missing}")

if total_missing == 0:
    print("  ✓ No missing values found. Dataset is complete.")
else:
    # Show which columns have missing values
    missing_per_col = df.isnull().sum()
    missing_cols = missing_per_col[missing_per_col > 0]
    print(f"  Columns with missing values ({len(missing_cols)} total):")
    print(missing_cols.to_string())


# ─────────────────────────────────────────────
# STEP 6 – Duplicate Row Check
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Duplicate rows check")
print("=" * 60)

num_duplicates = df.duplicated().sum()
print(f"  Duplicate rows: {num_duplicates}")

if num_duplicates == 0:
    print("  ✓ No duplicate rows found.")
else:
    print(f"  ⚠ Found {num_duplicates} duplicate rows.")


# ─────────────────────────────────────────────
# STEP 7 – Feature Statistics (sample of 5 features)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Feature statistics (first 5 features)")
print("=" * 60)

sample_features = feature_columns[:5]
print(df[sample_features].describe().round(3).to_string())


# ─────────────────────────────────────────────
# STEP 8 – Feature Value Range Distribution (visualisation)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Generating feature value distribution plot")
print("=" * 60)

# Plot histograms for the first 6 feature columns as a visual sample
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, col in enumerate(feature_columns[:6]):
    axes[i].hist(df[col].dropna(), bins=40, color="#a78bfa", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel("Value", fontsize=7)
    axes[i].set_ylabel("Frequency", fontsize=7)
    axes[i].tick_params(labelsize=6)

plt.suptitle("Sample EEG Feature Distributions (first 6 features)", fontsize=11, fontweight="bold")
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, "feature_distributions.png")
plt.savefig(hist_path, dpi=120)
plt.close()
print(f"  Histogram saved → {hist_path}")


# ─────────────────────────────────────────────
# STEP 9 – Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Exploration Summary")
print("=" * 60)
print(f"  Dataset file     : {DATASET_PATH}")
print(f"  Rows             : {num_rows}")
print(f"  Total columns    : {num_cols}")
print(f"  Feature columns  : {len(feature_columns)}")
print(f"  Target column    : '{target_column}'")
print(f"  Classes          : {list(label_counts.index)}")
print(f"  Missing values   : {total_missing}")
print(f"  Duplicate rows   : {num_duplicates}")
print(f"  Plots saved in   : {OUTPUT_DIR}/")
print()
print("  Phase 2 exploration complete!")
print("=" * 60)
