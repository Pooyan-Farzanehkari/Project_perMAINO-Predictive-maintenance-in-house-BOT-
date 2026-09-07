"""Load the MetroPT-3 (Air Compressor) dataset into a pandas DataFrame."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = PROJECT_ROOT / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv"
PROCESSED_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "metropt3.parquet"

ANALOG_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]
DIGITAL_COLUMNS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


def load_raw_data(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the raw MetroPT-3 CSV and return a cleaned, timestamp-indexed DataFrame."""
    dtypes = {col: "float32" for col in ANALOG_COLUMNS}
    dtypes.update({col: "int8" for col in DIGITAL_COLUMNS})

    df = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=["timestamp"],
        dtype=dtypes,
    )
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(data.shape)
    print(data.dtypes)
    print(data.head())

    PROCESSED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(PROCESSED_PARQUET_PATH)
    print(f"Saved parquet copy to {PROCESSED_PARQUET_PATH}")
