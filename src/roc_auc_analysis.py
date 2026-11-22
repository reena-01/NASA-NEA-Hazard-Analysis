# ============================================================
# NASA NEO ROC-AUC ANALYSIS
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
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

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the model-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run feature_selection.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """Separate features and target."""

    target_column = "hazard_target"

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' "
            f"not found in dataset."
        )

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):
    """Create the same stratified 80/20 split."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X_train,
    X_test,
    y_train
):
    """Train Logistic Regression and return probabilities."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    return probabilities


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    X_test,
    y_train
):
    """Train Random Forest and return probabilities."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return probabilities


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_test,
    logistic_probabilities,
    random_forest_probabilities
):
    """Create and save ROC curve comparison."""

    logistic_fpr, logistic_tpr, _ = roc_curve(
        y_test,
        logistic_probabilities
    )

    random_forest_fpr, random_forest_tpr, _ = roc_curve(
        y_test,
        random_forest_probabilities
    )

    logistic_auc = roc_auc_score(
        y_test,
        logistic_probabilities
    )

    random_forest_auc = roc_auc_score(
        y_test,
        random_forest_probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label=(
            f"Logistic Regression "
            f"(AUC = {logistic_auc:.3f})"
        )
    )

    plt.plot(
        random_forest_fpr,
        random_forest_tpr,
        label=(
            f"Random Forest "
            f"(AUC = {random_forest_auc:.3f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - NASA NEO Hazard Classification"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "roc_curve_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return (
        logistic_auc,
        random_forest_auc,
        output_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO ROC-AUC ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nDataset rows: {len(df)}"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print(
        f"Feature count: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_probabilities = train_logistic_regression(
        X_train,
        X_test,
        y_train
    )

    print(
        "Logistic Regression completed."
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest_probabilities = train_random_forest(
        X_train,
        X_test,
        y_train
    )

    print(
        "Random Forest completed."
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Plot ROC curve
    # --------------------------------------------------------

    (
        logistic_auc,
        random_forest_auc,
        output_file
    ) = plot_roc_curve(
        y_test,
        logistic_probabilities,
        random_forest_probabilities
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ROC-AUC RESULTS")
    print("=" * 60)

    print(
        f"Logistic Regression AUC: "
        f"{logistic_auc:.4f}"
    )

    print(
        f"Random Forest AUC: "
        f"{random_forest_auc:.4f}"
    )

    print(
        f"\nROC curve saved to:\n"
        f"{output_file}"
    )

    print("\n" + "=" * 60)
    print("ROC-AUC ANALYSIS COMPLETED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()