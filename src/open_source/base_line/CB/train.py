import argparse
from pathlib import Path

from open_source.catboost_training import train_catboost_residual
from open_source.data_prepare.schema import FEATURES_NO_LST


def default_data_dir():
    return Path("open_source") / "data_prepare" / "baseline_cb" / "station"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the no-LST CatBoost baseline residual model.")
    parser.add_argument("--train-parquet", default=str(default_data_dir() / "train.parquet"))
    parser.add_argument("--val-parquet", default=str(default_data_dir() / "val.parquet"))
    parser.add_argument("--test-parquet", default=str(default_data_dir() / "test.parquet"))
    parser.add_argument("--model-save-path", default=str(Path("open_source") / "base_line" / "CB" / "catboost_no_lst.cbm"))
    parser.add_argument("--task-type", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--devices", default="0")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    train_catboost_residual(
        train_parquet_path=args.train_parquet,
        val_parquet_path=args.val_parquet,
        test_parquet_path=args.test_parquet,
        feature_columns=FEATURES_NO_LST,
        model_save_path=args.model_save_path,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        task_type=args.task_type,
        devices=args.devices,
        early_stopping_rounds=args.early_stopping_rounds,
    )


if __name__ == "__main__":
    main()
