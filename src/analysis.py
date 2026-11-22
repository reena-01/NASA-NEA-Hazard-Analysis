# Exploratory Data Analysis (EDA)

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
    / "neo_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load processed NEO data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nProcessed data not found.\n\n"
            f"Expected file:\n{DATA_FILE}\n\n"
            f"Run data_loader.py first."
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# HAZARD ANALYSIS
# ============================================================

def analyze_hazardous_objects(df):
    """Analyze potentially hazardous NEOs."""

    column = "is_potentially_hazardous"

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' "
            f"is missing from the dataset."
        )

    counts = df[column].value_counts()

    total_objects = len(df)

    hazardous_objects = int(
        df[column].fillna(False).sum()
    )

    non_hazardous_objects = (
        total_objects - hazardous_objects
    )

    if total_objects > 0:
        hazardous_percentage = (
            hazardous_objects
            / total_objects
            * 100
        )
    else:
        hazardous_percentage = 0

    print("\n" + "=" * 60)
    print("NEO HAZARD ANALYSIS")
    print("=" * 60)

    print(
        f"\nTotal NEOs: "
        f"{total_objects}"
    )

    print(
        f"Potentially Hazardous NEOs: "
        f"{hazardous_objects}"
    )

    print(
        f"Non-Hazardous NEOs: "
        f"{non_hazardous_objects}"
    )

    print(
        f"Hazardous Percentage: "
        f"{hazardous_percentage:.2f}%"
    )

    print("\nClassification counts:")

    print(counts)

    print("=" * 60)

    return {
        "total": total_objects,
        "hazardous": hazardous_objects,
        "non_hazardous": non_hazardous_objects,
        "hazardous_percentage": hazardous_percentage,
    }


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(df):
    """Display basic dataset information."""

    print("\nDATASET SUMMARY")
    print("-" * 60)

    print(
        f"Rows: {df.shape[0]}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f"- {column}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    dataset_summary(df)

    analyze_hazardous_objects(df)


if __name__ == "__main__":
    main()