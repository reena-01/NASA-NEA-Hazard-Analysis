## DATA DOWNLOADER

from pathlib import Path
import json
import urllib.request


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "neo_data.json"


# ============================================================
# NASA API
# ============================================================

API_URL = (
    "https://api.nasa.gov/neo/rest/v1/feed"
    "?start_date=2026-01-01"
    "&end_date=2026-01-07"
    "&api_key=DEMO_KEY"
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():
    """Download Near-Earth Object data from NASA."""

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("NASA NEO DATA DOWNLOAD")
    print("=" * 60)

    print("\nConnecting to NASA API...")

    try:
        with urllib.request.urlopen(API_URL) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as error:
        raise RuntimeError(
            f"Failed to download NASA NEO data.\n"
            f"Error: {error}"
        ) from error

    with open(
        RAW_DATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print("\nDownload completed successfully.")

    print(
        f"Raw data saved to:\n"
        f"{RAW_DATA_FILE}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    download_data()