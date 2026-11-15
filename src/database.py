
import sqlite3
import re
from pathlib import Path

# Project database location
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "ANPR_Complete_Dataset"
    / "metadata"
    / "anpr_demo_database.sqlite"
)


def normalize_plate(plate):
    """Remove spaces and special characters."""
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(plate).upper()
    )


def lookup_vehicle(registration_number):
    """Find a vehicle in the synthetic SQLite registry."""

    plate = normalize_plate(registration_number)

    if not plate:
        return None

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM vehicle_registry
            WHERE UPPER(
                REPLACE(
                    REPLACE(registration_number_candidate, ' ', ''),
                    '-',
                    ''
                )
            ) = ?
            LIMIT 1
            """,
            (plate,)
        )

        record = cursor.fetchone()

        if record is None:
            return None

        return dict(record)

    finally:
        connection.close()


if __name__ == "__main__":
    test_plate = "25BH2229"

    result = lookup_vehicle(test_plate)

    if result:
        print("VEHICLE FOUND")
        print("-------------------------")

        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("No matching vehicle found.")
