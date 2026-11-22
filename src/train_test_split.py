# TRAIN / TEST SPLIT

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


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
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):
    """Separate input features X from target y."""

    target = "hazard_target"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    return X, y


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

def show_class_distribution(y, label):
    """Display target-class distribution."""

    print(
        f"\n{label} class distribution:"
    )

    counts = y.value_counts()

    percentages = (
        y.value_counts(
            normalize=True
        )
        * 100
    )

    result = pd.DataFrame(
        {
            "Count": counts,
            "Percentage": percentages.round(2)
        }
    )

    print(result)


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(X, y):
    """Split data into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO TRAIN / TEST SPLIT")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print(
        f"\nTotal dataset size: {len(df)}"
    )

    # Prepare X and y
    X, y = prepare_data(df)

    print(
        f"\nNumber of features: {X.shape[1]}"
    )

    print(
        f"Number of samples: {X.shape[0]}"
    )

    # Original class distribution
    show_class_distribution(
        y,
        "Original"
    )

    # Split
    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    # Display sizes
    print("\n" + "=" * 60)
    print("SPLIT RESULTS")
    print("=" * 60)

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Training percentage: "
        f"{len(X_train) / len(df) * 100:.1f}%"
    )

    print(
        f"Testing percentage: "
        f"{len(X_test) / len(df) * 100:.1f}%"
    )

    # Training distribution
    show_class_distribution(
        y_train,
        "Training"
    )

    # Testing distribution
    show_class_distribution(
        y_test,
        "Testing"
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()