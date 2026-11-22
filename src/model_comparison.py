# MODEL COMPARISON

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)


def load_data():
    """Load the model-ready dataset."""

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
    """Create the same stratified 80/20 split."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def evaluate_predictions(y_test, predictions):
    """Calculate model performance metrics."""

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
    }


def train_logistic_regression(
    X_train,
    X_test,
    y_train
):
    """Train Logistic Regression with feature scaling."""

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

    predictions = model.predict(
        X_test_scaled
    )

    return predictions


def train_random_forest(
    X_train,
    X_test,
    y_train
):
    """Train Random Forest without feature scaling."""

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return predictions


def create_comparison_table(
    y_test,
    logistic_predictions,
    random_forest_predictions
):
    """Create a comparison table."""

    logistic_results = evaluate_predictions(
        y_test,
        logistic_predictions
    )

    random_forest_results = evaluate_predictions(
        y_test,
        random_forest_predictions
    )

    comparison = pd.DataFrame(
        [
            logistic_results,
            random_forest_results
        ],
        index=[
            "Logistic Regression",
            "Random Forest"
        ]
    )

    return comparison


def display_best_model(comparison):
    """Display the best model for each metric."""

    print("\n" + "=" * 60)
    print("BEST MODEL BY METRIC")
    print("=" * 60)

    for metric in comparison.columns:

        best_model = comparison[
            metric
        ].idxmax()

        best_score = comparison.loc[
            best_model,
            metric
        ]

        print(
            f"{metric}: "
            f"{best_model} "
            f"({best_score:.4f})"
        )


def main():

    print("=" * 60)
    print("NASA NEO MODEL COMPARISON")
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
        "\nTraining Logistic Regression..."
    )

    logistic_predictions = train_logistic_regression(
        X_train,
        X_test,
        y_train
    )

    print(
        "Logistic Regression completed."
    )

    print(
        "\nTraining Random Forest..."
    )

    random_forest_predictions = train_random_forest(
        X_train,
        X_test,
        y_train
    )

    print(
        "Random Forest completed."
    )

    comparison = create_comparison_table(
        y_test,
        logistic_predictions,
        random_forest_predictions
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    print(
        comparison.round(4)
    )

    display_best_model(
        comparison
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()