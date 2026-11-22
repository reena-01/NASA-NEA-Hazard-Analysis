# ============================================================
# NASA NEO HAZARD ANALYSIS
# CONFUSION MATRIX VISUALIZATION
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the model-ready NASA NEO dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run the model preparation pipeline first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded dataset: {len(df):,} rows"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """Separate features and target."""

    target_column = "hazard_target"

    if target_column not in df.columns:
        raise KeyError(
            f"\nTarget column '{target_column}' "
            f"was not found in the dataset."
        )

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # Ensure all model features are numeric
    non_numeric_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            "\nNon-numeric feature columns found:\n"
            f"{non_numeric_columns}\n\n"
            f"Model-ready data should contain numeric features only."
        )

    # Check for missing values
    if X.isnull().any().any():
        missing_columns = (
            X.columns[
                X.isnull().any()
            ]
            .tolist()
        )

        raise ValueError(
            "\nMissing values found in feature columns:\n"
            f"{missing_columns}\n\n"
            f"Clean the model-ready dataset before training."
        )

    if y.isnull().any():
        raise ValueError(
            "\nMissing values found in hazard_target."
        )

    return X, y


# ============================================================
# DATA SPLIT
# ============================================================

def split_data(X, y):
    """Create a reproducible stratified 80/20 train-test split."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():
    """Create the figures directory."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def plot_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train Logistic Regression and save its confusion matrix."""

    print(
        "\nTraining Logistic Regression..."
    )

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=[
            "Not Hazardous",
            "Potentially Hazardous"
        ],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "Logistic Regression - Confusion Matrix"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "logistic_regression_confusion_matrix.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def plot_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train Random Forest and save its confusion matrix."""

    print(
        "\nTraining Random Forest..."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=[
            "Not Hazardous",
            "Potentially Hazardous"
        ],
        cmap="Greens",
        ax=ax
    )

    ax.set_title(
        "Random Forest - Confusion Matrix"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "random_forest_confusion_matrix.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete confusion matrix visualization pipeline."""

    print("=" * 60)
    print("NASA NEO CONFUSION MATRIX VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print(
        f"Feature count: {X.shape[1]}"
    )

    print(
        f"Target distribution:\n{y.value_counts().sort_index()}"
    )

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    prepare_output_directory()

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    plot_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    plot_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX VISUALIZATION COMPLETED")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()