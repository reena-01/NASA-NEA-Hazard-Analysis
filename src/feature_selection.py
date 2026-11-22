# FEATURE SELECTION

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
    / "ml_ready_neo_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_neo_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the ML-ready dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nML-ready dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run feature_engineering.py first."
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(df):
    """Select numerical features for machine learning."""

    selected_features = [
        "absolute_magnitude_h",
        "average_diameter_km",
        "diameter_range_km",
        "relative_velocity_km_s",
        "miss_distance_au"
    ]

    target = "hazard_target"

    missing_features = [
        column
        for column in selected_features + [target]
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_features)
        )

    model_df = df[
        selected_features + [target]
    ].copy()

    return model_df, selected_features, target


# ============================================================
# CHECK MISSING VALUES
# ============================================================

def check_missing_values(df):
    """Check missing values in model data."""

    print("\n" + "=" * 60)
    print("MISSING VALUES IN MODEL DATA")
    print("=" * 60)

    print(
        df.isnull().sum()
    )


# ============================================================
# CHECK CORRELATIONS
# ============================================================

def correlation_analysis(df, features):
    """Display feature correlations."""

    print("\n" + "=" * 60)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 60)

    correlation = (
        df[features]
        .corr()
        .round(3)
    )

    print(
        correlation
    )


# ============================================================
# TARGET CORRELATION
# ============================================================

def target_correlation(df, features, target):
    """Display correlation between features and target."""

    print("\n" + "=" * 60)
    print("FEATURE VS TARGET CORRELATION")
    print("=" * 60)

    correlations = (
        df[features + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations.round(3)
    )


# ============================================================
# REMOVE MISSING ROWS
# ============================================================

def remove_missing_rows(df):
    """Remove rows containing missing model values."""

    before = len(df)

    df = df.dropna()

    after = len(df)

    print(
        f"\nRows removed because of missing values: "
        f"{before - after}"
    )

    return df


# ============================================================
# SAVE MODEL DATA
# ============================================================

def save_model_data(df):
    """Save final model-ready dataset."""

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nModel-ready dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# DISPLAY FINAL FEATURES
# ============================================================

def display_features(features, target):
    """Display selected features."""

    print("\n" + "=" * 60)
    print("SELECTED FEATURES")
    print("=" * 60)

    for feature in features:
        print(f"X → {feature}")

    print(
        f"\nTarget (y) → {target}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO FEATURE SELECTION")
    print("=" * 60)

    # Load data
    df = load_data()

    # Select features
    model_df, features, target = select_features(df)

    # Display features
    display_features(
        features,
        target
    )

    # Missing values
    check_missing_values(model_df)

    # Remove missing rows
    model_df = remove_missing_rows(
        model_df
    )

    # Correlation analysis
    correlation_analysis(
        model_df,
        features
    )

    # Target correlation
    target_correlation(
        model_df,
        features,
        target
    )

    # Final information
    print("\n" + "=" * 60)
    print("FINAL MODEL DATA")
    print("=" * 60)

    print(
        f"Rows: {model_df.shape[0]}"
    )

    print(
        f"Columns: {model_df.shape[1]}"
    )

    # Save
    save_model_data(
        model_df
    )

    print("\n" + "=" * 60)
    print("FEATURE SELECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()