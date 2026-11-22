# REUSABLE HAZARD PREDICTION

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_model.joblib"
)


FEATURES = [
    "absolute_magnitude_h",
    "average_diameter_km",
    "diameter_range_km",
    "relative_velocity_km_s",
    "miss_distance_au",
]


def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"\nSaved model not found:\n"
            f"{MODEL_FILE}\n\n"
            f"Run save_final_model.py first."
        )

    return joblib.load(MODEL_FILE)


def create_input_data():

    print("\nUsing asteroid input values:")
    print("-" * 40)

    data = pd.DataFrame(
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

    print(data.to_string(index=False))

    return data


def predict_hazard(model, data):

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    probability_not_hazardous = probabilities[0]
    probability_hazardous = probabilities[1]

    return (
        prediction,
        probability_not_hazardous,
        probability_hazardous,
    )


def display_result(
    prediction,
    probability_not_hazardous,
    probability_hazardous,
):

    print("\n" + "=" * 60)
    print("HAZARD PREDICTION RESULT")
    print("=" * 60)

    if prediction == 1:

        print(
            "\nModel prediction:"
            " POTENTIALLY HAZARDOUS"
        )

    else:

        print(
            "\nModel prediction:"
            " NOT POTENTIALLY HAZARDOUS"
        )

    print(
        f"\nProbability - Not Hazardous: "
        f"{probability_not_hazardous:.2%}"
    )

    print(
        f"Probability - Potentially Hazardous: "
        f"{probability_hazardous:.2%}"
    )

    print("\nNote:")
    print(
        "This is a machine-learning prediction "
        "based on the project dataset."
    )

    print(
        "It is not an official NASA hazard assessment."
    )

    print("\n" + "=" * 60)


def main():

    print("=" * 60)
    print("NASA NEO HAZARD PREDICTION SYSTEM")
    print("=" * 60)

    model = load_model()

    print("\nSaved Random Forest model loaded successfully.")

    data = create_input_data()

    (
        prediction,
        probability_not_hazardous,
        probability_hazardous,
    ) = predict_hazard(
        model,
        data
    )

    display_result(
        prediction,
        probability_not_hazardous,
        probability_hazardous,
    )


if __name__ == "__main__":
    main()