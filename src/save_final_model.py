# SAVE FINAL RANDOM FOREST MODEL

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "random_forest_model.joblib"
)


def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run feature_selection.py first."
        )

    return pd.read_csv(INPUT_FILE)


def prepare_data(df):

    X = df.drop(
        columns=["hazard_target"]
    )

    y = df["hazard_target"]

    return X, y


def train_final_model(X, y):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    return model


def save_model(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_FILE}"
    )


def main():

    print("=" * 60)
    print("NASA NEO FINAL MODEL TRAINING")
    print("=" * 60)

    df = load_data()

    print(
        f"\nDataset rows: {len(df)}"
    )

    X, y = prepare_data(df)

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "\nTraining final Random Forest..."
    )

    model = train_final_model(
        X,
        y
    )

    print(
        "Training completed."
    )

    save_model(
        model
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL SAVED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()