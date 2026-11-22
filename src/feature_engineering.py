# FEATURE ENGINEERING

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_neo_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_ready_neo_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the cleaned NASA NEO dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nClean dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run data_cleaning.py first."
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# CREATE DIAMETER FEATURES
# ============================================================

def create_diameter_features(df):
    """Create useful asteroid diameter features."""

    df["average_diameter_km"] = (
        df["estimated_diameter_min_km"]
        + df["estimated_diameter_max_km"]
    ) / 2

    df["diameter_range_km"] = (
        df["estimated_diameter_max_km"]
        - df["estimated_diameter_min_km"]
    )

    return df


# ============================================================
# CREATE VELOCITY FEATURES
# ============================================================

def create_velocity_features(df):
    """Create velocity-related features."""

    df["velocity_category"] = pd.cut(
        df["relative_velocity_km_s"],
        bins=[
            -float("inf"),
            15,
            30,
            float("inf")
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    return df


# ============================================================
# CREATE DISTANCE FEATURES
# ============================================================

def create_distance_features(df):
    """Create miss-distance categories."""

    df["distance_category"] = pd.cut(
        df["miss_distance_au"],
        bins=[
            -float("inf"),
            0.05,
            0.10,
            float("inf")
        ],
        labels=[
            "Very Close",
            "Close",
            "Far"
        ]
    )

    return df


# ============================================================
# CREATE HAZARD TARGET
# ============================================================

def create_hazard_target(df):
    """Convert NASA hazard classification into a binary target."""

    df["hazard_target"] = (
        df["is_potentially_hazardous"]
        .astype(int)
    )

    return df


# ============================================================
# DISPLAY FEATURE INFORMATION
# ============================================================

def show_features(df):
    """Display the engineered features."""

    print("\n" + "=" * 60)
    print("ENGINEERED FEATURES")
    print("=" * 60)

    features = [
        "average_diameter_km",
        "diameter_range_km",
        "velocity_category",
        "distance_category",
        "hazard_target"
    ]

    for feature in features:

        if feature in df.columns:
            print(f"- {feature}")

    print(
        f"\nTotal rows: {len(df)}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )


# ============================================================
# HAZARD DISTRIBUTION
# ============================================================

def show_hazard_distribution(df):
    """Display target-class distribution."""

    print("\n" + "=" * 60)
    print("HAZARD TARGET DISTRIBUTION")
    print("=" * 60)

    print(
        df["hazard_target"].value_counts()
    )

    print("\nPercentages:")

    percentages = (
        df["hazard_target"]
        .value_counts(normalize=True)
        * 100
    )

    print(
        percentages.round(2)
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_data(df):
    """Save the ML-ready dataset."""

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nML-ready dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO FEATURE ENGINEERING")
    print("=" * 60)

    # Load clean dataset
    df = load_data()

    # Create features
    df = create_diameter_features(df)

    df = create_velocity_features(df)

    df = create_distance_features(df)

    df = create_hazard_target(df)

    # Display results
    show_features(df)

    show_hazard_distribution(df)

    # Save
    save_data(df)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()