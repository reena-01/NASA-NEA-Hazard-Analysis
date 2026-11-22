# ============================================================
# NASA NEO HAZARD ANALYSIS
# DATA VISUALIZATION
# ============================================================

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
    / "ml_ready_neo_data.csv"
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
    """Load the ML-ready NASA NEO dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATA_FILE}\n\n"
            f"Run feature_engineering.py first."
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():
    """Create the figures directory if it does not exist."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HAZARD CLASSIFICATION
# ============================================================

def plot_hazard_classification(df):
    """Plot hazardous vs non-hazardous objects."""

    counts = (
        df["is_potentially_hazardous"]
        .value_counts()
        .sort_index()
    )

    labels = [
        "Not Hazardous",
        "Potentially Hazardous"
    ]

    values = [
        counts.get(False, 0),
        counts.get(True, 0)
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values
    )

    plt.title(
        "NASA NEO Hazard Classification"
    )

    plt.xlabel(
        "Classification"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.xticks(rotation=0)

    plt.tight_layout()

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
        f"Saved: {output_file}"
    )


# ============================================================
# DIAMETER DISTRIBUTION
# ============================================================

def plot_diameter_distribution(df):
    """Plot asteroid diameter distribution."""

    diameter = (
        df["average_diameter_km"]
        .dropna()
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        diameter,
        bins=30
    )

    plt.title(
        "Average Asteroid Diameter Distribution"
    )

    plt.xlabel(
        "Average Diameter (km)"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "diameter_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# VELOCITY DISTRIBUTION
# ============================================================

def plot_velocity_distribution(df):
    """Plot relative velocity distribution."""

    velocity = (
        df["relative_velocity_km_s"]
        .dropna()
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        velocity,
        bins=30
    )

    plt.title(
        "Relative Velocity Distribution"
    )

    plt.xlabel(
        "Relative Velocity (km/s)"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "velocity_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# MISS DISTANCE DISTRIBUTION
# ============================================================

def plot_distance_distribution(df):
    """Plot miss-distance distribution."""

    distance = (
        df["miss_distance_au"]
        .dropna()
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        distance,
        bins=30
    )

    plt.title(
        "Miss Distance Distribution"
    )

    plt.xlabel(
        "Miss Distance (AU)"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "miss_distance_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# DIAMETER VS VELOCITY
# ============================================================

def plot_diameter_vs_velocity(df):
    """Plot asteroid diameter against relative velocity."""

    plot_data = df[
        [
            "average_diameter_km",
            "relative_velocity_km_s"
        ]
    ].dropna()

    plt.figure(figsize=(8, 5))

    plt.scatter(
        plot_data["average_diameter_km"],
        plot_data["relative_velocity_km_s"],
        alpha=0.6
    )

    plt.title(
        "Asteroid Diameter vs Relative Velocity"
    )

    plt.xlabel(
        "Average Diameter (km)"
    )

    plt.ylabel(
        "Relative Velocity (km/s)"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "diameter_vs_velocity.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# HAZARD VS DIAMETER
# ============================================================

def plot_hazard_vs_diameter(df):
    """Compare asteroid diameter by hazard classification."""

    not_hazardous = (
        df.loc[
            df["is_potentially_hazardous"] == False,
            "average_diameter_km"
        ]
        .dropna()
    )

    potentially_hazardous = (
        df.loc[
            df["is_potentially_hazardous"] == True,
            "average_diameter_km"
        ]
        .dropna()
    )

    hazard_groups = [
        not_hazardous,
        potentially_hazardous
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        hazard_groups,
        tick_labels=[
            "Not Hazardous",
            "Potentially Hazardous"
        ]
    )

    plt.title(
        "Asteroid Diameter by Hazard Classification"
    )

    plt.xlabel(
        "Classification"
    )

    plt.ylabel(
        "Average Diameter (km)"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "diameter_by_hazard.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# HAZARD VS VELOCITY
# ============================================================

def plot_hazard_vs_velocity(df):
    """Compare relative velocity by hazard classification."""

    not_hazardous = (
        df.loc[
            df["is_potentially_hazardous"] == False,
            "relative_velocity_km_s"
        ]
        .dropna()
    )

    potentially_hazardous = (
        df.loc[
            df["is_potentially_hazardous"] == True,
            "relative_velocity_km_s"
        ]
        .dropna()
    )

    hazard_groups = [
        not_hazardous,
        potentially_hazardous
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        hazard_groups,
        tick_labels=[
            "Not Hazardous",
            "Potentially Hazardous"
        ]
    )

    plt.title(
        "Relative Velocity by Hazard Classification"
    )

    plt.xlabel(
        "Classification"
    )

    plt.ylabel(
        "Relative Velocity (km/s)"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "velocity_by_hazard.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def plot_correlation_matrix(df):
    """Create and save a numerical feature correlation matrix."""

    columns = [
        "absolute_magnitude_h",
        "average_diameter_km",
        "diameter_range_km",
        "relative_velocity_km_s",
        "miss_distance_km",
        "miss_distance_au",
        "hazard_target"
    ]

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing columns required for correlation matrix: "
            f"{missing_columns}"
        )

    correlation = (
        df[columns]
        .corr()
    )

    plt.figure(figsize=(10, 8))

    image = plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Correlation"
    )

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(columns)),
        columns
    )

    plt.title(
        "NASA NEO Feature Correlation Matrix"
    )

    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "correlation_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete NASA NEO visualization pipeline."""

    print("=" * 60)
    print("NASA NEO DATA VISUALIZATION")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print(
        f"Loaded dataset: {len(df):,} rows"
    )

    # Prepare output directory
    prepare_output_directory()

    # Generate visualizations
    plot_hazard_classification(df)

    plot_diameter_distribution(df)

    plot_velocity_distribution(df)

    plot_distance_distribution(df)

    plot_diameter_vs_velocity(df)

    plot_hazard_vs_diameter(df)

    plot_hazard_vs_velocity(df)

    plot_correlation_matrix(df)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()