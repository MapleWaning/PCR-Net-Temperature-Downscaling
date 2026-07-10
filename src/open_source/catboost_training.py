from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from open_source.data_prepare.schema import CATEGORICAL_COLUMNS, CHANNELS, TARGET_COLUMNS


def evaluate_metrics(y_true, y_pred, channel_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    print(f"[{channel_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | MBE: {mbe:.4f} | R2: {r2:.4f}")


def prepare_feature_frame(df, feature_columns):
    features = df[feature_columns].copy()
    features["land_use"] = features["land_use"].astype(int)
    return features


def train_catboost_residual(
    train_parquet_path,
    val_parquet_path,
    test_parquet_path,
    feature_columns,
    model_save_path,
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=42,
    task_type="GPU",
    devices="0",
    early_stopping_rounds=150,
):
    print("Loading parquet datasets...")
    train_df = pd.read_parquet(train_parquet_path)
    val_df = pd.read_parquet(val_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)
    print(f"Rows -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_pool = Pool(
        data=prepare_feature_frame(train_df, feature_columns),
        label=train_df[TARGET_COLUMNS],
        cat_features=CATEGORICAL_COLUMNS,
    )
    val_pool = Pool(
        data=prepare_feature_frame(val_df, feature_columns),
        label=val_df[TARGET_COLUMNS],
        cat_features=CATEGORICAL_COLUMNS,
    )
    test_pool = Pool(
        data=prepare_feature_frame(test_df, feature_columns),
        cat_features=CATEGORICAL_COLUMNS,
    )

    # The target is the residual between station observations and aggregated ERA5 baseline.
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        random_seed=random_seed,
        task_type=task_type,
        devices=devices,
        allow_writing_files=False,
    )

    print("Training CatBoost residual model...")
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stopping_rounds,
        verbose=100,
    )

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_save_path)
    print(f"Saved model to {model_save_path}")

    print("Evaluating reconstructed temperatures on the test split...")
    pred_residuals = model.predict(test_pool)
    for idx, (name, base_col, residual_col) in enumerate(CHANNELS):
        pred_temperature = test_df[base_col].to_numpy() + pred_residuals[:, idx]
        true_temperature = test_df[base_col].to_numpy() + test_df[residual_col].to_numpy()
        evaluate_metrics(true_temperature, pred_temperature, name)

    print("Feature importances:")
    for name, score in zip(feature_columns, model.get_feature_importance(train_pool)):
        print(f"{name}: {score:.2f}%")

    return model
