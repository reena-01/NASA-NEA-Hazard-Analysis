# RANDOM FOREST MACHINE LEARNING MODEL

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)


def load_data():
    """Load the model-ready NASA NEO dataset."""

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nModel-ready dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run feature_selection.py first."
        )

    return pd.read_csv(INPUT_FILE)


def prepare_data(df):
    """Separate features and target."""

    target = "hazard_target"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    return X, y


def split_data(X, y):
    """Create a stratified 80/20 train-test split."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    """Train the Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate Random Forest performance."""

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
    print("RANDOM FOREST MODEL PERFORMANCE")
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


def show_feature_importance(model, X):
    """Display Random Forest feature importance."""

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    importance = (
        importance
        .sort_values(ascending=False)
    )

    print(
        importance.round(4)
    )


def main():

    print("=" * 60)
    print("NASA NEO RANDOM FOREST MODEL")
    print("=" * 60)

    df = load_data()

    print(
        f"\nDataset rows: {len(df)}"
    )

    X, y = prepare_data(df)

    print(
        f"Number of features: {X.shape[1]}"
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        "\nTraining Random Forest..."
    )

    model = train_model(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model,
        X
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()