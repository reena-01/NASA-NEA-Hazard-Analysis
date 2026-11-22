# CROSS-VALIDATION

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from sklearn.pipeline import Pipeline
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


def create_models():
    """Create Logistic Regression and Random Forest models."""

    logistic_regression = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest
    }


def run_cross_validation(model, X, y, cv):
    """Run cross-validation for one model."""

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    return results


def display_results(model_name, results):
    """Display cross-validation results."""

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    for metric in metrics:

        scores = results[
            f"test_{metric}"
        ]

        print(
            f"{metric.capitalize():10}: "
            f"{scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    print("\nIndividual fold scores:")

    for metric in metrics:

        scores = results[
            f"test_{metric}"
        ]

        print(
            f"{metric.capitalize():10}: "
            f"{scores.round(4)}"
        )


def create_summary(results_dict):
    """Create a cross-validation summary table."""

    rows = []

    for model_name, results in results_dict.items():

        rows.append(
            {
                "Model": model_name,
                "Accuracy Mean": results[
                    "test_accuracy"
                ].mean(),

                "Precision Mean": results[
                    "test_precision"
                ].mean(),

                "Recall Mean": results[
                    "test_recall"
                ].mean(),

                "F1 Mean": results[
                    "test_f1"
                ].mean()
            }
        )

    return pd.DataFrame(rows)


def main():

    print("=" * 60)
    print("NASA NEO CROSS-VALIDATION")
    print("=" * 60)

    df = load_data()

    print(
        f"\nDataset rows: {len(df)}"
    )

    X, y = prepare_data(df)

    print(
        f"Number of features: {X.shape[1]}"
    )

    print(
        f"Number of samples: {X.shape[0]}"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print(
        "\nCross-validation strategy:"
    )

    print(
        "5-Fold Stratified Cross-Validation"
    )

    models = create_models()

    results_dict = {}

    for model_name, model in models.items():

        print(
            f"\nRunning {model_name}..."
        )

        results = run_cross_validation(
            model,
            X,
            y,
            cv
        )

        results_dict[
            model_name
        ] = results

        display_results(
            model_name,
            results
        )

    summary = create_summary(
        results_dict
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
    