# BASELINE MACHINE LEARNING MODEL

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the model-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nModel-ready dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run feature_selection.py first."
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """Separate features and target."""

    target = "hazard_target"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    """Create stratified training and testing datasets."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(X_train, X_test):
    """Standardize numerical features."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return X_train_scaled, X_test_scaled


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):
    """Train Logistic Regression."""

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(matrix)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO BASELINE MACHINE LEARNING MODEL")
    print("=" * 60)

    # Load data
    df = load_data()

    print(
        f"\nDataset rows: {len(df)}"
    )

    # Prepare X and y
    X, y = prepare_data(df)

    print(
        f"Number of features: {X.shape[1]}"
    )

    # Split
    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # Scale
    X_train_scaled, X_test_scaled = (
        scale_features(
            X_train,
            X_test
        )
    )

    # Train
    print("\nTraining Logistic Regression...")

    model = train_model(
        X_train_scaled,
        y_train
    )

    print(
        "Training completed."
    )

    # Evaluate
    evaluate_model(
        model,
        X_test_scaled,
        y_test
    )

    print("\n" + "=" * 60)
    print("BASELINE MODEL COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()