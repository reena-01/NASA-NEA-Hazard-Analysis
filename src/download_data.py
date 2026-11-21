import json
from pathlib import Path
from urllib.request import urlopen


API_URL = (
    "https://api.nasa.gov/neo/rest/v1/feed"
    "?start_date=2025-01-01"
    "&end_date=2025-01-07"
    "&api_key=DEMO_KEY"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "neo_data.json"


def download_data():
    print("Downloading NASA Near-Earth Object data...")

    with urlopen(API_URL) as response:
        data = json.load(response)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("Download complete!")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    download_data()