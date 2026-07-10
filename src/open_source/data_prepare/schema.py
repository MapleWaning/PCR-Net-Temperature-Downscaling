METADATA_COLUMNS = ["Station_ID", "Year", "DOY"]

ERA5_FEATURES = ["era5_mean", "era5_max", "era5_min"]
LST_FEATURES = ["lst_mean", "lst_max", "lst_min"]
DEM_FEATURES = ["dem_ch1", "dem_ch2", "dem_ch3", "dem_ch4"]
STATIC_TIME_FEATURES = ["land_use", "albedo", "doy_sin", "doy_cos"]

FEATURES_WITH_LST = ERA5_FEATURES + LST_FEATURES + DEM_FEATURES + STATIC_TIME_FEATURES
FEATURES_NO_LST = ERA5_FEATURES + DEM_FEATURES + STATIC_TIME_FEATURES

TARGET_COLUMNS = ["Residual_Mean", "Residual_Max", "Residual_Min"]
CATEGORICAL_COLUMNS = ["land_use"]
CHANNELS = [
    ("Mean", "era5_mean", "Residual_Mean"),
    ("Max", "era5_max", "Residual_Max"),
    ("Min", "era5_min", "Residual_Min"),
]

FEATURE_SETS = {
    "catboost_lst": FEATURES_WITH_LST,
    "baseline_cb": FEATURES_NO_LST,
    "baseline_rf": FEATURES_NO_LST,
}


def feature_columns(target):
    return FEATURE_SETS[target]
