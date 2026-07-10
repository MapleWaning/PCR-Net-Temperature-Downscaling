import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TEMP_MEAN = 10.9198
TEMP_STD = 14.4475
DEM_MEAN = 1016.2878
DEM_STD = 1045.2930
STATIC_START_YEAR = 2008
GLOBAL_START_DATE = pd.Timestamp("2008-01-01")


def resolve_year_file(data_dir, year):
    candidates = [
        f"era5_t2m_{year}.h5",
        f"t_base_advanced_{year}.h5",
        f"t_base_{year}.h5",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No temperature base file for {year} in {data_dir}")


def resolve_guidance_file(data_dir, year):
    candidates = [
        f"rf_t2m_{year}.h5",
        f"cb_t2m_{year}.h5",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No RF/CB guidance file for {year} in {data_dir}")


def aggregate_temperature_channels(raw):
    return np.stack(
        [
            np.mean(raw, axis=0),
            np.max(raw, axis=0),
            np.min(raw, axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_static_and_base(dem, albedo, base_3ch):
    elevation = (dem[0] - DEM_MEAN) / (DEM_STD + 1e-6)
    slope = dem[1] / 90.0
    dem_norm = np.stack([elevation, slope, dem[2], dem[3]], axis=0).astype(np.float32)
    albedo_norm = np.clip(albedo, 0, 1.0).astype(np.float32)
    base_norm = ((base_3ch - TEMP_MEAN) / TEMP_STD).astype(np.float32)
    return dem_norm, albedo_norm, base_norm


def normalize_guidance(guidance_3ch):
    return ((guidance_3ch - TEMP_MEAN) / TEMP_STD).astype(np.float32)


def make_time_features(day_of_year, height=128, width=128):
    radians = 2 * np.pi * day_of_year / 365.0
    time_map = np.zeros((2, height, width), dtype=np.float32)
    time_map[0, :, :] = np.sin(radians)
    time_map[1, :, :] = np.cos(radians)
    return time_map


def station_lookup(static_data, station_id):
    if station_id in static_data:
        return static_data[station_id]
    if str(station_id) in static_data:
        return static_data[str(station_id)]
    try:
        int_id = int(station_id)
    except (TypeError, ValueError):
        int_id = None
    if int_id is not None and int_id in static_data:
        return static_data[int_id]
    raise KeyError(f"Station {station_id} was not found in static data")


def open_h5(path):
    return h5py.File(path, "r", rdcc_nbytes=32 * 1024 * 1024, rdcc_nslots=50023)


class _BaseDownscaleDataset(Dataset):
    def __init__(
        self,
        tbase_dir,
        static_npy_path,
        truth_npy_path,
        years,
        target_station_ids=None,
        rf_base_dir=None,
        num_land_classes=10,
        station_coords_dict=None,
        split="train",
        augment=False,
        sample_index_csv=None,
    ):
        self.tbase_dir = tbase_dir
        self.rf_base_dir = rf_base_dir
        self.years = list(years)
        self.num_land_classes = num_land_classes
        self.station_coords_dict = station_coords_dict or {}
        self.split = split
        self.augment = augment
        self.sample_index_csv = sample_index_csv
        self.files = {}

        self.static_data = np.load(static_npy_path, allow_pickle=True).item()
        self.truth_data = np.load(truth_npy_path)
        self.station_ids = list(self.static_data.keys())
        self.index_map = self._build_index(target_station_ids)

    def _build_index(self, target_station_ids):
        if self.sample_index_csv:
            return self._build_index_from_csv(target_station_ids)

        target_ids = None
        if target_station_ids is not None:
            target_ids = {str(station_id) for station_id in target_station_ids}

        index_map = []
        for year in self.years:
            date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
            for day_idx, current_date in enumerate(date_range):
                for station_idx, station_id in enumerate(self.station_ids):
                    if target_ids is not None and str(station_id) not in target_ids:
                        continue
                    index_map.append(
                        {
                            "station_idx": station_idx,
                            "station_id": station_id,
                            "year": year,
                            "day_idx": day_idx,
                            "day_of_year": current_date.dayofyear,
                            "month_idx": current_date.month - 1,
                            "date": current_date,
                        }
                    )
        return index_map

    def _build_index_from_csv(self, target_station_ids):
        target_ids = None
        if target_station_ids is not None:
            target_ids = {str(station_id) for station_id in target_station_ids}

        df = pd.read_csv(self.sample_index_csv)
        station_to_idx = {str(station_id): idx for idx, station_id in enumerate(self.station_ids)}
        index_map = []
        for _, row in df.iterrows():
            station_id = str(row["Station_ID"])
            if target_ids is not None and station_id not in target_ids:
                continue
            if station_id not in station_to_idx:
                continue

            date_value = row.get("target_date", row.get("date"))
            if pd.isna(date_value):
                year = int(row["year"])
                day_idx = int(row.get("target_day_idx", row.get("day_idx", 0)))
                current_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_idx)
            else:
                current_date = pd.Timestamp(date_value)
                year = int(current_date.year)
                day_idx = int(row.get("target_day_idx", current_date.dayofyear - 1))

            index_map.append(
                {
                    "station_idx": station_to_idx[station_id],
                    "station_id": station_id,
                    "year": year,
                    "day_idx": day_idx,
                    "day_of_year": current_date.dayofyear,
                    "month_idx": current_date.month - 1,
                    "date": current_date,
                    "sample_id": row.get("sample_id"),
                }
            )
        if not index_map:
            raise ValueError(f"No samples were selected from {self.sample_index_csv}.")
        return index_map

    def __len__(self):
        return len(self.index_map)

    def _file(self, path):
        if path not in self.files or not self.files[path].id.valid:
            self.files[path] = open_h5(path)
        return self.files[path]

    def _truth(self, station_idx, date):
        try:
            return self.truth_data[station_idx, (date - GLOBAL_START_DATE).days]
        except IndexError:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    def _base_sample(self, year, day_idx, station_idx):
        path = resolve_year_file(self.tbase_dir, year)
        raw = self._file(path)["data"][day_idx, station_idx]
        return aggregate_temperature_channels(raw)

    def _guidance_sample(self, year, day_idx, station_idx):
        path = resolve_guidance_file(self.rf_base_dir, year)
        return self._file(path)["data"][day_idx, station_idx].astype(np.float32)

    def _land_use(self, land_use, year):
        year_idx = min(max(0, year - STATIC_START_YEAR), len(land_use) - 1)
        lu_map = land_use[year_idx]
        if lu_map.ndim == 3:
            lu_map = np.squeeze(lu_map)
        lu = torch.from_numpy(lu_map).long()
        if lu.max() >= self.num_land_classes:
            lu = lu - 1
        return torch.clamp(lu, min=0, max=self.num_land_classes - 1)

    def _target_and_mask(self, station_id, truth_value):
        target = torch.zeros((3, 128, 128), dtype=torch.float32)
        mask = torch.zeros((1, 128, 128), dtype=torch.float32)
        row, col = self.station_coords_dict.get(station_id, (64, 64))
        if not np.isnan(truth_value).any() and not (truth_value < -100).any() and not (truth_value > 100).any():
            target[:, row, col] = torch.from_numpy(((truth_value - TEMP_MEAN) / TEMP_STD).astype(np.float32))
            mask[:, row, col] = 1.0
        return target, mask

    def _maybe_augment(self, sample, target, mask):
        if self.split != "train" or not self.augment or random.random() <= 0.5:
            return sample, target, mask

        keys = list(sample.keys())
        for key in keys:
            sample[key] = torch.flip(sample[key], [-1])

        target = torch.flip(target, [-1])
        mask = torch.flip(mask, [-1])

        if "dem" in sample:
            # East-west flipping changes the sign of the aspect sine channel.
            sample["dem"][2, :, :] *= -1.0

        return sample, target, mask

    def __getitem__(self, idx):
        info = self.index_map[idx]
        station_id = info["station_id"]
        station_idx = info["station_idx"]
        year = info["year"]

        base = self._base_sample(year, info["day_idx"], station_idx)
        static = station_lookup(self.static_data, station_id)
        month_idx = info["month_idx"]
        year_idx = min(max(0, year - STATIC_START_YEAR), len(static["albedo"]) - 1)
        albedo = static["albedo"][year_idx, month_idx]
        if albedo.ndim == 2:
            albedo = albedo[np.newaxis, ...]

        dem_norm, alb_norm, base_norm = normalize_static_and_base(
            static["dem"].astype(np.float32),
            albedo.astype(np.float32),
            base,
        )

        sample = {
            "base": torch.from_numpy(base_norm),
            "dem": torch.from_numpy(dem_norm),
            "alb": torch.from_numpy(alb_norm),
            "time": torch.from_numpy(make_time_features(info["day_of_year"])),
            "lu": self._land_use(static["land_use"], year),
        }

        if self.rf_base_dir is not None:
            # PCR-Net keeps the tree-model output as a normalized guidance field for the shape loss.
            guidance = self._guidance_sample(year, info["day_idx"], station_idx)
            sample["rf_base"] = torch.from_numpy(normalize_guidance(guidance))

        truth = self._truth(station_idx, info["date"])
        target, mask = self._target_and_mask(station_id, truth)
        return self._maybe_augment(sample, target, mask)


class ClimateDownscaleDataset_V3(_BaseDownscaleDataset):
    def __init__(self, tbase_dir, rf_base_dir, static_npy_path, truth_npy_path, years, **kwargs):
        super().__init__(
            tbase_dir=tbase_dir,
            rf_base_dir=rf_base_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            years=years,
            **kwargs,
        )


class ClimateDownscaleDataset_V4(_BaseDownscaleDataset):
    def __init__(self, tbase_dir, rf_base_dir, static_npy_path, truth_npy_path, target_station_ids, **kwargs):
        super().__init__(
            tbase_dir=tbase_dir,
            rf_base_dir=rf_base_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            years=range(2008, 2025),
            target_station_ids=target_station_ids,
            augment=True,
            **kwargs,
        )


class ClimateDownscaleDataset_Baseline(_BaseDownscaleDataset):
    def __init__(self, tbase_dir, static_npy_path, truth_npy_path, years, **kwargs):
        super().__init__(
            tbase_dir=tbase_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            years=years,
            **kwargs,
        )


class ClimateDownscaleDataset_Baseline_V2(_BaseDownscaleDataset):
    def __init__(self, tbase_dir, static_npy_path, truth_npy_path, target_station_ids, **kwargs):
        super().__init__(
            tbase_dir=tbase_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            years=range(2008, 2025),
            target_station_ids=target_station_ids,
            **kwargs,
        )
