from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_model.joblib"
)


def test_model_dataset_exists():
    assert MODEL_DATA_FILE.exists()


def test_model_dataset_has_required_columns():
    df = pd.read_csv(MODEL_DATA_FILE)

    required_columns = [
        "absolute_magnitude_h",
        "average_diameter_km",
        "diameter_range_km",
        "relative_velocity_km_s",
        "miss_distance_au",
        "hazard_target",
    ]

    for column in required_columns:
        assert column in df.columns


def test_model_dataset_is_not_empty():
    df = pd.read_csv(MODEL_DATA_FILE)

    assert len(df) > 0


def test_target_contains_two_classes():
    df = pd.read_csv(MODEL_DATA_FILE)

    unique_classes = df["hazard_target"].unique()

    assert len(unique_classes) == 2


def test_saved_model_can_predict():
    """Check that the saved model can make a prediction."""

    model = joblib.load(MODEL_FILE)

    sample_data = pd.DataFrame(
        [
            {
                "absolute_magnitude_h": 22,
                "average_diameter_km": 0.5,
                "diameter_range_km": 0.2,
                "relative_velocity_km_s": 18,
                "miss_distance_au": 0.05,
            }
        ]
    )

    prediction = model.predict(sample_data)

    assert prediction[0] in [0, 1]