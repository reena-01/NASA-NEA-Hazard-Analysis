# EXPLORATORY DATA ANALYSIS

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_ready_neo_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the ML-ready NASA NEO dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nML-ready dataset not found:\n"
            f"{DATA_FILE}\n\n"
            f"Run feature_engineering.py first."
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# DATASET OVERVIEW
# ============================================================

def dataset_overview(df):
    """Display basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")

    for column in df.columns:
        print(f" - {column}")


# ============================================================
# DATA TYPES
# ============================================================

def show_data_types(df):
    """Display column data types."""

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

def missing_value_analysis(df):
    """Analyze missing values."""

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if missing.empty:

        print("No missing values found.")

    else:

        print(missing)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def duplicate_analysis(df):
    """Analyze duplicate rows."""

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates}"
    )


# ============================================================
# HAZARD ANALYSIS
# ============================================================

def hazard_analysis(df):
    """Analyze NASA hazard classification."""

    print("\n" + "=" * 60)
    print("HAZARD CLASSIFICATION")
    print("=" * 60)

    counts = (
        df["is_potentially_hazardous"]
        .value_counts()
    )

    print("\nCounts:")

    print(counts)

    print("\nPercentages:")

    percentages = (
        df["is_potentially_hazardous"]
        .value_counts(
            normalize=True
        )
        * 100
    )

    print(
        percentages.round(2)
    )


# ============================================================
# DIAMETER ANALYSIS
# ============================================================

def diameter_analysis(df):
    """Analyze asteroid diameter."""

    print("\n" + "=" * 60)
    print("DIAMETER ANALYSIS")
    print("=" * 60)

    column = "average_diameter_km"

    print(
        df[column].describe()
    )


# ============================================================
# MAGNITUDE ANALYSIS
# ============================================================

def magnitude_analysis(df):
    """Analyze absolute magnitude."""

    print("\n" + "=" * 60)
    print("ABSOLUTE MAGNITUDE ANALYSIS")
    print("=" * 60)

    column = "absolute_magnitude_h"

    print(
        df[column].describe()
    )


# ============================================================
# VELOCITY ANALYSIS
# ============================================================

def velocity_analysis(df):
    """Analyze relative velocity."""

    print("\n" + "=" * 60)
    print("RELATIVE VELOCITY ANALYSIS")
    print("=" * 60)

    column = "relative_velocity_km_s"

    print(
        df[column].describe()
    )


# ============================================================
# DISTANCE ANALYSIS
# ============================================================

def distance_analysis(df):
    """Analyze miss distance."""

    print("\n" + "=" * 60)
    print("MISS DISTANCE ANALYSIS")
    print("=" * 60)

    print("\nDistance in kilometers:")

    print(
        df["miss_distance_km"].describe()
    )

    print("\nDistance in astronomical units:")

    print(
        df["miss_distance_au"].describe()
    )


# ============================================================
# HAZARD VS FEATURES
# ============================================================

def hazard_feature_comparison(df):
    """Compare numerical features by hazard classification."""

    print("\n" + "=" * 60)
    print("HAZARD VS NUMERICAL FEATURES")
    print("=" * 60)

    columns = [
        "average_diameter_km",
        "absolute_magnitude_h",
        "relative_velocity_km_s",
        "miss_distance_au"
    ]

    comparison = (
        df.groupby(
            "is_potentially_hazardous"
        )[columns]
        .mean()
    )

    print(
        comparison
    )


# ============================================================
# VELOCITY CATEGORY
# ============================================================

def velocity_category_analysis(df):
    """Analyze velocity categories."""

    print("\n" + "=" * 60)
    print("VELOCITY CATEGORY")
    print("=" * 60)

    print(
        df["velocity_category"]
        .value_counts()
    )


# ============================================================
# DISTANCE CATEGORY
# ============================================================

def distance_category_analysis(df):
    """Analyze distance categories."""

    print("\n" + "=" * 60)
    print("DISTANCE CATEGORY")
    print("=" * 60)

    print(
        df["distance_category"]
        .value_counts()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEAR-EARTH ASTEROID EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_data()

    dataset_overview(df)

    show_data_types(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    hazard_analysis(df)

    diameter_analysis(df)

    magnitude_analysis(df)

    velocity_analysis(df)

    distance_analysis(df)

    hazard_feature_comparison(df)

    velocity_category_analysis(df)

    distance_category_analysis(df)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()