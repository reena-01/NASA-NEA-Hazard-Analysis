##VISUALIZATION


from pathlib import Path

import matplotlib.pyplot as plt
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

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load processed NASA NEO data."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nProcessed data file not found.\n\n"
            f"Expected file:\n{DATA_FILE}\n\n"
            f"Run data_loader.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print(
        f"Loaded {len(df)} NEO records."
    )

    return df


# ============================================================
# HAZARD CLASSIFICATION CHART
# ============================================================

def create_hazard_chart(df):
    """Create hazardous vs non-hazardous NEO chart."""

    column = "is_potentially_hazardous"

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' was not found.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # Make sure missing values are treated as False
    hazard_data = df[column].fillna(False)

    counts = (
        hazard_data
        .value_counts()
        .reindex(
            [False, True],
            fill_value=0
        )
    )

    counts.index = [
        "Not Hazardous",
        "Potentially Hazardous"
    ]

    print("\nHazard classification:")
    print(counts)

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        counts.index,
        counts.values
    )

    plt.title(
        "Potentially Hazardous vs Non-Hazardous NEOs",
        fontsize=14
    )

    plt.xlabel(
        "Classification"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    output_file = (
        FIGURES_DIR
        / "hazard_classification.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nChart saved successfully:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO HAZARD ANALYSIS - VISUALIZATION")
    print("=" * 60)

    df = load_data()

    create_hazard_chart(df)

    print(
        "\nVisualization completed successfully."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()