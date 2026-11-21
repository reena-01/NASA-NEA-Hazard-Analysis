import json
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "neo_data.json"

PROCESSED_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "neo_data.csv"


def load_raw_data():
    """Load the raw NASA NEO JSON data."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"NASA data file not found: {RAW_DATA_FILE}"
        )

    with open(RAW_DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def create_dataframe(data):
    """Convert NASA NEO JSON data into a Pandas DataFrame."""

    asteroid_records = []

    for date, asteroids in data["near_earth_objects"].items():

        for asteroid in asteroids:

            record = {
                "id": asteroid.get("id"),
                "name": asteroid.get("name"),
                "absolute_magnitude": asteroid.get(
                    "absolute_magnitude_h"
                ),
                "is_potentially_hazardous": asteroid.get(
                    "is_potentially_hazardous_asteroid"
                ),
                "date": date,
            }

            asteroid_records.append(record)

    return pd.DataFrame(asteroid_records)


def main():
    print("Loading NASA Near-Earth Object data...")

    # Load JSON data
    data = load_raw_data()

    # Convert JSON to DataFrame
    df = create_dataframe(data)

    # Create processed-data folder if it doesn't exist
    PROCESSED_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed data
    df.to_csv(PROCESSED_DATA_FILE, index=False)

    print("\nDataset loaded successfully!")
    print("Processed dataset saved to:")
    print(PROCESSED_DATA_FILE)

    print("--------------------------------")

    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()