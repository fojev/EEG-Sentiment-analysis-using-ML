"""
PHASE 3 – Data Preprocessing
==============================
Purpose:
    1. Load emotions.csv
    2. Separate features (X) and target (y)
    3. Validate the data (shape, types, missing, infinite values)
    4. Perform a stratified train-test split (80% / 20%)
    5. Verify the split (class balance + no index overlap)

IMPORTANT – Data Leakage Rules:
    - StandardScaler is NOT applied here.
    - No transformation is fitted on the full dataset.
    - Scaling happens INSIDE the sklearn Pipeline during model training.
    - sklearn classifiers accept string labels directly, so no label encoding needed.

Run command (from the project root folder):
    venv/Scripts/python backend/preprocessing/preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = "dataset/emotions.csv"   # Path to the CSV file
TARGET_COLUMN = "label"                 # Name of the target/class column
TEST_SIZE     = 0.20                    # 20% data for testing
RANDOM_STATE  = 42                      # Fixed seed → reproducible results every run


# ──────────────────────────────────────────────────────────────
# FUNCTION 1: load_dataset
# ──────────────────────────────────────────────────────────────
def load_dataset(dataset_path):
    """
    Load the CSV file into a Pandas DataFrame.

    Also fixes the malformed first column name:
    The original CSV has '# mean_0_a' (with a leading #),
    which we rename to 'mean_0_a' for clean usage.

    Parameters:
        dataset_path (str): Path to the CSV file.

    Returns:
        df (DataFrame): The loaded dataset.
    """
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)

    df = pd.read_csv(dataset_path)

    # Fix the leading '#' in the first column name
    df.rename(columns={"# mean_0_a": "mean_0_a"}, inplace=True)

    print(f"  File           : {dataset_path}")
    print(f"  Rows           : {df.shape[0]}")
    print(f"  Columns        : {df.shape[1]}")

    return df


# ──────────────────────────────────────────────────────────────
# FUNCTION 2: validate_dataset
# ──────────────────────────────────────────────────────────────
def validate_dataset(df):
    """
    Perform data validation checks before splitting.

    Checks:
        1. Target column exists
        2. Dataset dimensions and feature count
        3. Target class distribution
        4. Missing values (NaN)
        5. Infinite values (Inf / -Inf) in feature columns
        6. Data types of all columns

    Parameters:
        df (DataFrame): The loaded dataset.

    Raises:
        ValueError: If the target column is missing.
        ValueError: If missing values are found (since we confirmed none exist,
                    this acts as a safety guard for future use).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Data validation")
    print("=" * 60)

    # ── Check 1: Target column must exist ─────────────────────
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    print(f"  ✓ Target column '{TARGET_COLUMN}' found.")

    # ── Check 2: Dimensions ───────────────────────────────────
    num_rows, num_cols = df.shape
    num_features = num_cols - 1   # All columns except the target
    print(f"\n  Dataset shape     : {df.shape}")
    print(f"  Feature columns   : {num_features}")
    print(f"  Target column     : '{TARGET_COLUMN}'")

    # ── Check 3: Target class distribution ────────────────────
    print(f"\n  Target class distribution:")
    class_counts = df[TARGET_COLUMN].value_counts()
    total = len(df)
    for label, count in class_counts.items():
        pct = count / total * 100
        print(f"    {label:<12} : {count:>4} samples  ({pct:.1f}%)")

    # ── Check 4: Missing values ───────────────────────────────
    total_missing = df.isnull().sum().sum()
    print(f"\n  Missing values    : {total_missing}")
    if total_missing > 0:
        # Show exactly which columns are affected
        cols_with_missing = df.isnull().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0]
        print(f"  Columns affected  : {len(cols_with_missing)}")
        print(cols_with_missing.to_string())
        raise ValueError(
            f"Dataset has {total_missing} missing values. "
            "Please clean the data before proceeding."
        )
    else:
        print(f"  ✓ No missing values.")

    # ── Check 5: Infinite values in feature columns ───────────
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

    # np.isinf only works on numeric data
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    total_inf = np.isinf(numeric_df.values).sum()
    print(f"\n  Infinite values   : {total_inf}")
    if total_inf > 0:
        print(f"  ⚠ Found {total_inf} infinite values in feature columns.")
        print("    These should be handled before model training.")
    else:
        print(f"  ✓ No infinite values.")

    # ── Check 6: Data types ───────────────────────────────────
    print(f"\n  Column data types:")
    dtype_summary = df.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        print(f"    {str(dtype):<12} : {count} columns")

    return feature_cols, class_counts


# ──────────────────────────────────────────────────────────────
# FUNCTION 3: separate_features_and_target
# ──────────────────────────────────────────────────────────────
def separate_features_and_target(df, feature_cols):
    """
    Split the DataFrame into:
        X – feature columns (input to the model)
        y – target column   (what the model predicts)

    Both X and y are kept as Pandas objects (DataFrame and Series)
    so their row indices are preserved. This lets us verify there
    is no overlap between train and test indices later.

    Parameters:
        df          (DataFrame): Full dataset.
        feature_cols (list)    : List of feature column names.

    Returns:
        X (DataFrame): Feature matrix, shape (2132, 2548)
        y (Series)   : Target labels,  shape (2132,)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Separating features (X) and target (y)")
    print("=" * 60)

    X = df[feature_cols]        # All feature columns → DataFrame
    y = df[TARGET_COLUMN]       # Target column       → Series (string labels)

    print(f"  X shape : {X.shape}   ← feature matrix")
    print(f"  y shape : {y.shape}   ← target labels")
    print(f"  y dtype : {y.dtype}   ← string labels (no encoding needed)")
    print(f"  y unique: {sorted(y.unique())}")
    print()
    print("  NOTE: Labels are kept as strings.")
    print("        sklearn models handle string class labels natively.")

    return X, y


# ──────────────────────────────────────────────────────────────
# FUNCTION 4: split_data
# ──────────────────────────────────────────────────────────────
def split_data(X, y):
    """
    Perform a stratified train-test split.

    What is stratified splitting?
        It ensures each class appears in the same proportion in
        both the training and test sets as in the full dataset.
        Example: If 33% of data is POSITIVE, then ~33% of both
        train and test will also be POSITIVE.
        Without stratify=y, a random split might accidentally
        put more of one class in the test set.

    Parameters:
        X (DataFrame): Feature matrix.
        y (Series)   : Target labels.

    Returns:
        X_train, X_test (DataFrames): Feature splits.
        y_train, y_test (Series)    : Label splits.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Stratified train-test split (80% train / 20% test)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size    = TEST_SIZE,     # 20% → test set
        random_state = RANDOM_STATE,  # Fixed seed → same split every run
        stratify     = y              # Preserve class proportions
    )

    total = len(y)
    print(f"  Total samples  : {total}")
    print(f"  Training set   : {len(y_train)} samples  "
          f"({len(y_train)/total*100:.1f}%)")
    print(f"  Test set       : {len(y_test)} samples  "
          f"({len(y_test)/total*100:.1f}%)")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────
# FUNCTION 5: verify_split
# ──────────────────────────────────────────────────────────────
def verify_split(X_train, X_test, y_train, y_test):
    """
    Verify the train-test split is correct by checking:
        1. Class distribution in both sets (should mirror original).
        2. No row index appears in both train AND test sets.
           (If an index is shared, the same sample would be in both
           sets, which would cause optimistic/misleading evaluation.)

    Parameters:
        X_train, X_test (DataFrames): Feature splits.
        y_train, y_test (Series)    : Label splits.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Verifying the split")
    print("=" * 60)

    # ── Class distribution check ──────────────────────────────
    all_classes = sorted(y_train.unique())

    print(f"  {'Class':<12}  {'Train':>7}  {'Test':>7}  {'Train%':>8}  {'Test%':>8}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")

    for cls in all_classes:
        train_count = (y_train == cls).sum()
        test_count  = (y_test  == cls).sum()
        train_pct   = train_count / len(y_train) * 100
        test_pct    = test_count  / len(y_test)  * 100
        print(f"  {cls:<12}  {train_count:>7}  {test_count:>7}  "
              f"{train_pct:>7.1f}%  {test_pct:>7.1f}%")

    print()
    print(f"  Total train : {len(y_train)}")
    print(f"  Total test  : {len(y_test)}")

    # ── Index overlap check ───────────────────────────────────
    # Convert index to Python sets and check the intersection
    train_indices = set(X_train.index)
    test_indices  = set(X_test.index)
    overlap       = train_indices.intersection(test_indices)

    print(f"\n  Index overlap check:")
    if len(overlap) == 0:
        print(f"  ✓ No overlap. {len(train_indices)} train indices and "
              f"{len(test_indices)} test indices are completely separate.")
    else:
        print(f"  ✗ OVERLAP DETECTED: {len(overlap)} indices appear in both sets!")
        print(f"    Overlapping indices: {list(overlap)[:10]} ...")


# ──────────────────────────────────────────────────────────────
# MAIN – Run as a standalone verification script
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1: Load the CSV
    df = load_dataset(DATASET_PATH)

    # Step 2: Validate data quality
    feature_cols, class_counts = validate_dataset(df)

    # Step 3: Separate X and y
    X, y = separate_features_and_target(df, feature_cols)

    # Step 4: Stratified split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Verify the split
    verify_split(X_train, X_test, y_train, y_test)

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    print(f"  Original dataset shape : {df.shape}")
    print(f"  X shape                : {X.shape}")
    print(f"  y shape                : {y.shape}")
    print(f"  Training samples       : {len(y_train)}")
    print(f"  Testing  samples       : {len(y_test)}")
    print(f"  Test size              : {int(TEST_SIZE * 100)}%")
    print(f"  Random state           : {RANDOM_STATE}")
    print(f"  Label type             : {y.dtype} (no encoding applied)")
    print()
    print("  What is NOT done here (intentionally):")
    print("    ✗ StandardScaler NOT applied   → done inside Pipeline in Phase 4")
    print("    ✗ Label encoding NOT applied   → sklearn handles string labels")
    print("    ✗ Feature selection NOT done   → done inside Pipeline in Phase 4")
    print()
    print("  Phase 3 preprocessing complete!")
    print("=" * 60)
