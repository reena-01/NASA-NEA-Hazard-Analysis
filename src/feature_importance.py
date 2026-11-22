# RANDOM FOREST FEATURE IMPORTANCE

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


def load_data():
    """Load the model-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run feature_selection.py first."
        )

    return pd.read_csv(INPUT_FILE)


def prepare_data(df):
    """Separate features and target."""

    X = df.drop(
        columns=["hazard_target"]
    )

    y = df["hazard_target"]

    return X, y


def split_data(X, y):
    """Create the same stratified 80/20 split."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    """Train Random Forest."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def calculate_importance(model, X):
    """Calculate feature importance."""

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    return (
        importance
        .sort_values(ascending=True)
    )


def display_importance(importance):
    """Display feature importance values."""

    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    for feature, value in importance.sort_values(
        ascending=False
    ).items():

        print(
            f"{feature}: {value:.4f}"
        )


def plot_importance(importance):
    """Create feature importance visualization."""

    plt.figure(figsize=(9, 6))

    plt.barh(
        importance.index,
        importance.values
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "random_forest_feature_importance.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"\nFeature importance plot saved to:\n"
        f"{output_file}"
    )


def main():

    print("=" * 60)
    print("NASA NEO RANDOM FOREST FEATURE IMPORTANCE")
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
        f"Training samples: {len(X_train)}"
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

    importance = calculate_importance(
        model,
        X
    )

    display_importance(
        importance
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_importance(
        importance
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()