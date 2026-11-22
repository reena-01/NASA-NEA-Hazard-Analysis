# DATA LOADER JSON

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "neo_data.json"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "neo_data.csv"
)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():
    """Load raw NASA NEO JSON data."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nRaw NASA data was not found.\n\n"
            f"Expected file:\n{RAW_DATA_FILE}\n\n"
            f"Run download_data.py first."
        )

    return pd.read_json(RAW_DATA_FILE)


# ============================================================
# PROCESS NEO DATA
# ============================================================

def process_data():
    """Convert NASA NEO JSON into a flat CSV dataset."""

    import json

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nRaw data not found:\n{RAW_DATA_FILE}"
        )

    with open(
        RAW_DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        raw_data = json.load(file)

    objects = []

    near_earth_objects = raw_data.get(
        "near_earth_objects",
        {}
    )

    for date, daily_objects in near_earth_objects.items():

        for neo in daily_objects:

            close_approach_data = neo.get(
                "close_approach_data",
                []
            )

            approach = (
                close_approach_data[0]
                if close_approach_data
                else {}
            )

            relative_velocity = approach.get(
                "relative_velocity",
                {}
            )

            miss_distance = approach.get(
                "miss_distance",
                {}
            )

            estimated_diameter = neo.get(
                "estimated_diameter",
                {}
            )

            kilometers = estimated_diameter.get(
                "kilometers",
                {}
            )

            objects.append(
                {
                    "id": neo.get("id"),
                    "name": neo.get("name"),
                    "absolute_magnitude_h": neo.get(
                        "absolute_magnitude_h"
                    ),
                    "estimated_diameter_min_km": kilometers.get(
                        "estimated_diameter_min"
                    ),
                    "estimated_diameter_max_km": kilometers.get(
                        "estimated_diameter_max"
                    ),
                    "is_potentially_hazardous": neo.get(
                        "is_potentially_hazardous_asteroid"
                    ),
                    "close_approach_date": approach.get(
                        "close_approach_date"
                    ),
                    "relative_velocity_km_s": relative_velocity.get(
                        "kilometers_per_second"
                    ),
                    "miss_distance_km": miss_distance.get(
                        "kilometers"
                    ),
                }
            )

    df = pd.DataFrame(objects)

    return df


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(df):
    """Save processed NEO data as CSV."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False
    )

    print(
        f"\nProcessed data saved to:\n"
        f"{PROCESSED_DATA_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NASA NEO DATA PROCESSING")
    print("=" * 60)

    df = process_data()

    print(
        f"\nProcessed objects: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    save_processed_data(df)

    print("\nData processing completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()