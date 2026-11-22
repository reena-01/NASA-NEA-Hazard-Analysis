# ============================================================
# NASA NEO FINAL MODEL EVALUATION
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

REPORT_FILE = (
    REPORTS_DIR
    / "final_model_evaluation.txt"
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

    return pd.read_csv(INPUT_FILE)


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
# CREATE MODELS
# ============================================================

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
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }


# ============================================================
# TEST SET EVALUATION
# ============================================================

def evaluate_test_set(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Evaluate models on the fixed test set."""

    results = {}

    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        results[model_name] = {
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
            "Confusion Matrix": confusion_matrix(
                y_test,
                predictions
            ),
        }

    return results


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    models,
    X,
    y
):
    """Run 5-fold stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",

        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),

        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
    }

    results = {}

    for model_name, model in models.items():

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise"
        )

        results[model_name] = {
            "CV Accuracy": cv_results[
                "test_accuracy"
            ].mean(),

            "CV Precision": cv_results[
                "test_precision"
            ].mean(),

            "CV Recall": cv_results[
                "test_recall"
            ].mean(),

            "CV F1 Score": cv_results[
                "test_f1"
            ].mean(),
        }

    return results


# ============================================================
# CREATE FINAL TABLE
# ============================================================

def create_final_table(
    test_results,
    cv_results
):
    """Combine test and cross-validation results."""

    rows = []

    for model_name in test_results:

        rows.append(
            {
                "Model": model_name,

                "Test Accuracy":
                    test_results[
                        model_name
                    ]["Accuracy"],

                "Test Precision":
                    test_results[
                        model_name
                    ]["Precision"],

                "Test Recall":
                    test_results[
                        model_name
                    ]["Recall"],

                "Test F1":
                    test_results[
                        model_name
                    ]["F1 Score"],

                "CV Accuracy":
                    cv_results[
                        model_name
                    ]["CV Accuracy"],

                "CV Precision":
                    cv_results[
                        model_name
                    ]["CV Precision"],

                "CV Recall":
                    cv_results[
                        model_name
                    ]["CV Recall"],

                "CV F1":
                    cv_results[
                        model_name
                    ]["CV F1 Score"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MODEL SELECTION
# ============================================================

def select_model(summary):
    """Select the model with the highest cross-validation F1."""

    best_index = summary[
        "CV F1"
    ].idxmax()

    return summary.loc[
        best_index,
        "Model"
    ]


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    summary,
    selected_model,
    test_results,
    total_samples,
    feature_count
):
    """Create the final text evaluation report."""

    lines = []

    lines.append(
        "NASA NEAR-EARTH ASTEROID HAZARD ANALYSIS"
    )

    lines.append(
        "FINAL MODEL EVALUATION REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    lines.append(
        "DATASET"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Total samples: {total_samples}"
    )

    lines.append(
        f"Features: {feature_count}"
    )

    lines.append(
        "Train/Test split: 80/20"
    )

    lines.append(
        "Cross-validation: 5-fold stratified"
    )

    lines.append("")

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    lines.append(
        "MODEL PERFORMANCE"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        summary.round(4).to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        f"Selected model based on highest "
        f"cross-validation F1: {selected_model}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    lines.append(
        "CONFUSION MATRICES"
    )

    lines.append(
        "=" * 70
    )

    for model_name, result in test_results.items():

        lines.append(
            f"\n{model_name}:"
        )

        lines.append(
            str(
                result["Confusion Matrix"]
            )
        )

    lines.append("")

    # --------------------------------------------------------
    # Scientific limitations
    # --------------------------------------------------------

    lines.append(
        "SCIENTIFIC LIMITATIONS"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "The dataset contains only a small sample "
        "of NASA Near-Earth Objects."
    )

    lines.append(
        "The dataset is also imbalanced between "
        "hazardous and non-hazardous classes."
    )

    lines.append(
        "Model results should therefore be interpreted "
        "as exploratory rather than production-ready."
    )

    lines.append(
        "The target represents the potentially hazardous "
        "classification used in the dataset."
    )

    lines.append(
        "The model identifies statistical relationships "
        "with this classification and does not establish "
        "physical causation or provide an official "
        "asteroid hazard assessment."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NASA NEO FINAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nDataset rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y = prepare_data(df)

    print(
        f"Number of features: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models()

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating test set..."
    )

    test_results = evaluate_test_set(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        "Test evaluation completed."
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print(
        "\nRunning 5-fold cross-validation..."
    )

    cv_results = run_cross_validation(
        models,
        X,
        y
    )

    print(
        "Cross-validation completed."
    )

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    summary = create_final_table(
        test_results,
        cv_results
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select final model
    # --------------------------------------------------------

    selected_model = select_model(
        summary
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL SELECTION")
    print("=" * 70)

    print(
        f"\nSelected model: {selected_model}"
    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = create_report(
        summary,
        selected_model,
        test_results,
        total_samples=len(df),
        feature_count=X.shape[1]
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    print(
        f"\nFinal report saved to:\n"
        f"{REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()