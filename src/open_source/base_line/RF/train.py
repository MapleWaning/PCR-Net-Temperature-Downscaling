import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from open_source.data_prepare.schema import CHANNELS, FEATURES_NO_LST, TARGET_COLUMNS


NUMERIC_FEATURES = [col for col in FEATURES_NO_LST if col != "land_use"]
CATEGORICAL_FEATURES = ["land_use"]


def default_data_dir():
    return Path("open_source") / "data_prepare" / "baseline_rf" / "station"


def evaluate_metrics(y_true, y_pred, channel_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    print(f"[{channel_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | MBE: {mbe:.4f} | R2: {r2:.4f}")


def build_pipeline(n_estimators=150, max_depth=10, random_seed=42, n_jobs=-1, verbose=1):
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_random_forest_residual(
    train_parquet_path,
    val_parquet_path,
    test_parquet_path,
    model_save_path,
    n_estimators=150,
    max_depth=10,
    n_jobs=-1,
    verbose=1,
):
    print("Loading parquet datasets...")
    train_df = pd.read_parquet(train_parquet_path)
    val_df = pd.read_parquet(val_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)
    print(f"Rows -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    pipeline = build_pipeline(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, verbose=verbose)

    # Random Forest has no early stopping, so the validation split is kept for external checks.
    pipeline.fit(train_df[FEATURES_NO_LST], train_df[TARGET_COLUMNS])

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"Saved model to {model_save_path}")

    print("Evaluating reconstructed temperatures on the test split...")
    pred_residuals = pipeline.predict(test_df[FEATURES_NO_LST])
    for idx, (name, base_col, residual_col) in enumerate(CHANNELS):
        pred_temperature = test_df[base_col].to_numpy() + pred_residuals[:, idx]
        true_temperature = test_df[base_col].to_numpy() + test_df[residual_col].to_numpy()
        evaluate_metrics(true_temperature, pred_temperature, name)

    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    print("Feature importances:")
    for name, score in sorted(zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True):
        print(f"{name.replace('num__', '').replace('cat__', '')}: {score * 100:.2f}%")

    return pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train the no-LST Random Forest baseline residual model.")
    parser.add_argument("--train-parquet", default=str(default_data_dir() / "train.parquet"))
    parser.add_argument("--val-parquet", default=str(default_data_dir() / "val.parquet"))
    parser.add_argument("--test-parquet", default=str(default_data_dir() / "test.parquet"))
    parser.add_argument("--model-save-path", default=str(Path("open_source") / "base_line" / "RF" / "rf_no_lst.joblib"))
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    train_random_forest_residual(
        train_parquet_path=args.train_parquet,
        val_parquet_path=args.val_parquet,
        test_parquet_path=args.test_parquet,
        model_save_path=args.model_save_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
