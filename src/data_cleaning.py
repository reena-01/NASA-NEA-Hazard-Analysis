# DATA CLEANING AND VALIDATION

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
    / "neo_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_neo_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the processed NASA NEO dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nProcessed dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run data_loader.py first."
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# DATA OVERVIEW
# ============================================================

def show_overview(df):
    """Display basic dataset information."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nData types:")
    print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

def check_missing_values(df):
    """Check missing values."""

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    total_missing = missing.sum()

    print(
        f"\nTotal missing values: {total_missing}"
    )


# ============================================================
# DUPLICATES
# ============================================================

def check_duplicates(df):
    """Check duplicate rows."""

    print("\n" + "=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates}"
    )


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

def convert_data_types(df):
    """Convert columns to appropriate data types."""

    numeric_columns = [
        "absolute_magnitude_h",
        "estimated_diameter_min_km",
        "estimated_diameter_max_km",
        "relative_velocity_km_s",
        "miss_distance_km",
        "miss_distance_au",
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    if "close_approach_date" in df.columns:

        df["close_approach_date"] = pd.to_datetime(
            df["close_approach_date"],
            errors="coerce"
        )

    return df


# ============================================================
# INVALID VALUE CHECK
# ============================================================

def check_invalid_values(df):
    """Check for physically invalid negative values."""

    print("\n" + "=" * 60)
    print("INVALID VALUE CHECK")
    print("=" * 60)

    numeric_columns = [
        "estimated_diameter_min_km",
        "estimated_diameter_max_km",
        "relative_velocity_km_s",
        "miss_distance_km",
        "miss_distance_au",
    ]

    for column in numeric_columns:

        if column in df.columns:

            invalid_count = (
                df[column] < 0
            ).sum()

            print(
                f"{column}: "
                f"{invalid_count} negative values"
            )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """Remove exact duplicate rows."""

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(
        f"\nRemoved duplicates: "
        f"{before - after}"
    )

    return df


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

def remove_invalid_records(df):
    """Remove records with invalid numerical values."""

    numeric_columns = [
        "estimated_diameter_min_km",
        "estimated_diameter_max_km",
        "relative_velocity_km_s",
        "miss_distance_km",
        "miss_distance_au",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df = df[
                df[column].notna()
                & (df[column] >= 0)
            ]

    return df


# ============================================================
# HAZARD CLASSIFICATION CHECK
# ============================================================

def check_target_column(df):
    """Check the machine-learning target column."""

    print("\n" + "=" * 60)
    print("TARGET CLASS DISTRIBUTION")
    print("=" * 60)

    if "is_potentially_hazardous" not in df.columns:

        print(
            "Target column not found."
        )

        return

    print(
        df["is_potentially_hazardous"]
        .value_counts(dropna=False)
    )


# ============================================================
# SAVE CLEAN DATA
# ============================================================

def save_clean_data(df):
    """Save the cleaned dataset."""

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nClean dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO DATA CLEANING PIPELINE")
    print("=" * 60)

    # Load
    df = load_data()

    # Overview
    show_overview(df)

    # Missing values
    check_missing_values(df)

    # Duplicates
    check_duplicates(df)

    # Convert data types
    df = convert_data_types(df)

    # Invalid values
    check_invalid_values(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Remove invalid records
    df = remove_invalid_records(df)

    # Target distribution
    check_target_column(df)

    # Save
    save_clean_data(df)

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()