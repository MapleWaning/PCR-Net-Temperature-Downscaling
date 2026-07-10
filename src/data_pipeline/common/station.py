from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_METADATA_COLUMNS = ["Station_ID", "Station_Name", "Latitude", "Longitude"]


def normalize_station_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    try:
        if "e" in text.lower():
            return str(int(float(text)))
    except ValueError:
        pass
    return text


def build_noaa_id(usaf: object, wban: object) -> str:
    return f"{str(usaf).strip()}{str(wban).strip().zfill(5)}"


def standardize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Station_ID" not in result.columns:
        if "NOAA_ID" in result.columns:
            result["Station_ID"] = result["NOAA_ID"]
        elif {"USAF", "WBAN"}.issubset(result.columns):
            result["Station_ID"] = [build_noaa_id(u, w) for u, w in zip(result["USAF"], result["WBAN"])]
        elif "ID" in result.columns:
            result["Station_ID"] = result["ID"]

    if "Station_Name" not in result.columns:
        for candidate in ("STATION NAME", "NAME", "Name", "station_name"):
            if candidate in result.columns:
                result["Station_Name"] = result[candidate]
                break
        else:
            result["Station_Name"] = "UNKNOWN"

    if "Latitude" not in result.columns:
        for candidate in ("LAT", "Lat", "lat", "latitude"):
            if candidate in result.columns:
                result["Latitude"] = result[candidate]
                break

    if "Longitude" not in result.columns:
        for candidate in ("LON", "Lon", "lon", "longitude"):
            if candidate in result.columns:
                result["Longitude"] = result[candidate]
                break

    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in result.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    result = result.reset_index(drop=True)
    result["Station_ID"] = result["Station_ID"].map(normalize_station_id)
    result["Station_Name"] = result["Station_Name"].fillna("UNKNOWN").astype(str)
    result["Latitude"] = result["Latitude"].astype(float)
    result["Longitude"] = result["Longitude"].astype(float)

    ordered = REQUIRED_METADATA_COLUMNS + [col for col in result.columns if col not in REQUIRED_METADATA_COLUMNS]
    return result[ordered]


def read_metadata(path: str | Path) -> pd.DataFrame:
    return standardize_metadata(pd.read_csv(path, dtype={"Station_ID": str, "NOAA_ID": str, "USAF": str, "WBAN": str}))


def write_metadata(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    standardize_metadata(df).to_csv(output, index=False)
    return output
