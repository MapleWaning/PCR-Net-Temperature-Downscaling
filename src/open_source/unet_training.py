import argparse
import csv
import importlib
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from open_source.unet_datasets import (
    ClimateDownscaleDataset_Baseline,
    ClimateDownscaleDataset_Baseline_V2,
    ClimateDownscaleDataset_V3,
    ClimateDownscaleDataset_V4,
)
from open_source.unet_losses import HybridRefinementLoss_V2, PureMSELoss
from open_source.unet_models import BasicUNet, ResAttUNet


def import_project_config():
    return importlib.import_module("config")


def comma_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_station_ids(path):
    return pd.read_csv(path)["Station_ID"].tolist()


def run_dir(root, name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{timestamp}-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_guided_batch(raw_batch, device):
    batch, targets, masks = raw_batch
    targets = targets.to(device)
    masks = masks.to(device)
    base = batch["base"].to(device)
    rf_base = batch["rf_base"].to(device)
    dem = batch["dem"].to(device)
    alb = batch["alb"].to(device)
    time_feat = batch["time"].to(device)
    lu = batch["lu"].to(device)
    lu_onehot = F.one_hot(lu, num_classes=10).permute(0, 3, 1, 2).float()

    # Input order is shared by ResAttUNet and BasicUNet: base, terrain, albedo, time, land use.
    inputs = torch.cat([base, dem, alb, time_feat, lu_onehot], dim=1)
    extras = {"mask": masks, "guidance_map": rf_base, "input_dem": dem}
    return inputs, targets, extras


def prepare_baseline_batch(raw_batch, device):
    batch, targets, masks = raw_batch
    targets = targets.to(device)
    masks = masks.to(device)
    base = batch["base"].to(device)
    dem = batch["dem"].to(device)
    alb = batch["alb"].to(device)
    time_feat = batch["time"].to(device)
    lu = batch["lu"].to(device)
    lu_onehot = F.one_hot(lu, num_classes=10).permute(0, 3, 1, 2).float()
    inputs = torch.cat([base, dem, alb, time_feat, lu_onehot], dim=1)
    return inputs, targets, {"mask": masks}


def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_dir,
    prepare_batch,
    max_lr,
    max_norm,
):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000,
    )
    save_dir = Path(save_dir)
    best_val = float("inf")
    history = []
    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for raw_batch in train_loader:
            inputs, targets, extras = prepare_batch(raw_batch, device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets, **extras)
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for raw_batch in val_loader:
                inputs, targets, extras = prepare_batch(raw_batch, device)
                val_loss += criterion(model(inputs), targets, **extras).item()
        val_loss /= max(len(val_loader), 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, save_dir / "best_model.pth")

    torch.save({"model_state_dict": model.state_dict(), "epoch": epochs - 1, "val_loss": val_loss}, save_dir / "final_model.pth")
    with open(save_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    return {"best_val_loss": best_val, "save_dir": str(save_dir)}


def build_guided_datasets(args, config):
    if args.generalization == "year":
        train_dataset = ClimateDownscaleDataset_V3(
            tbase_dir=args.tbase_dir,
            rf_base_dir=args.guidance_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            years=args.train_years,
            split="train",
            sample_index_csv=args.train_sample_csv,
        )
        val_dataset = ClimateDownscaleDataset_V3(
            tbase_dir=args.tbase_dir,
            rf_base_dir=args.guidance_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            years=args.val_years,
            split="val",
            sample_index_csv=args.val_sample_csv,
        )
    else:
        train_dataset = ClimateDownscaleDataset_V4(
            tbase_dir=args.tbase_dir,
            rf_base_dir=args.guidance_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            target_station_ids=read_station_ids(args.train_meta_csv),
            split="train",
            sample_index_csv=args.train_sample_csv,
        )
        val_dataset = ClimateDownscaleDataset_V4(
            tbase_dir=args.tbase_dir,
            rf_base_dir=args.guidance_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            target_station_ids=read_station_ids(args.val_meta_csv),
            split="val",
            sample_index_csv=args.val_sample_csv,
        )
    return train_dataset, val_dataset


def build_baseline_datasets(args, config):
    if args.generalization == "year":
        train_dataset = ClimateDownscaleDataset_Baseline(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            years=args.train_years,
            split="train",
            sample_index_csv=args.train_sample_csv,
        )
        val_dataset = ClimateDownscaleDataset_Baseline(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            years=args.val_years,
            split="val",
            sample_index_csv=args.val_sample_csv,
        )
    else:
        train_dataset = ClimateDownscaleDataset_Baseline_V2(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            target_station_ids=read_station_ids(args.train_meta_csv),
            split="train",
            sample_index_csv=args.train_sample_csv,
        )
        val_dataset = ClimateDownscaleDataset_Baseline_V2(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            target_station_ids=read_station_ids(args.val_meta_csv),
            split="val",
            sample_index_csv=args.val_sample_csv,
        )
    return train_dataset, val_dataset


def make_loaders(train_dataset, val_dataset, args):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def add_common_args(parser, config, default_output_dir):
    parser.add_argument("--generalization", choices=["year", "station"], default="station")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max-lr", type=float, default=config.MAX_LR)
    parser.add_argument("--max-norm", type=float, default=config.MAX_NORM)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--input-channels", type=int, default=config.INPUT_CHANNELS)
    parser.add_argument("--output-channels", type=int, default=config.OUTPUT_CHANNELS)
    parser.add_argument("--lambda-grad", type=float, default=config.LAMBDA_GRAD)
    parser.add_argument("--alpha-terrain", type=float, default=config.ALPHA_TERRAIN)
    parser.add_argument("--train-years", type=comma_ints, default=config.TRAIN_YEAR)
    parser.add_argument("--val-years", type=comma_ints, default=config.VERIFY_YEAR)
    parser.add_argument("--static-path", default=config.STATIC_PATH)
    parser.add_argument("--truth-path", default=config.TRUTH_PATH)
    parser.add_argument("--train-meta-csv", default=config.TRAIN_META_CSV)
    parser.add_argument("--val-meta-csv", default=config.VAL_META_CSV)
    parser.add_argument("--train-sample-csv", default=None)
    parser.add_argument("--val-sample-csv", default=None)
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-pretrained-backbone", action="store_true")


def fit_pcr_net(args, config, run_label="pcr-net"):
    train_dataset, val_dataset = build_guided_datasets(args, config)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, args)
    model = ResAttUNet(
        in_channels=args.input_channels,
        out_channels=args.output_channels,
        use_attention=args.use_attention,
        use_sft=args.use_sft,
        pretrained_backbone=not args.no_pretrained_backbone,
    )
    criterion = HybridRefinementLoss_V2(lambda_grad=args.lambda_grad, alpha_terrain=args.alpha_terrain)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = run_dir(args.output_dir, args.run_name or f"{run_label}-{args.generalization}")
    return train_loop(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, save_dir, prepare_guided_batch, args.max_lr, args.max_norm)


def fit_baseline_unet(args, config, run_label="baseline-unet"):
    train_dataset, val_dataset = build_baseline_datasets(args, config)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, args)
    model = BasicUNet(in_channels=args.input_channels, out_channels=args.output_channels)
    criterion = PureMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = run_dir(args.output_dir, args.run_name or f"{run_label}-{args.generalization}")
    return train_loop(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, save_dir, prepare_baseline_batch, args.max_lr, args.max_norm)


def fit_resunet_pure_mse(args, config, run_label="ablation-pure-mse"):
    train_dataset, val_dataset = build_baseline_datasets(args, config)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, args)
    model = ResAttUNet(
        in_channels=args.input_channels,
        out_channels=args.output_channels,
        use_attention=args.use_attention,
        use_sft=args.use_sft,
        pretrained_backbone=not args.no_pretrained_backbone,
    )
    criterion = PureMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = run_dir(args.output_dir, args.run_name or f"{run_label}-{args.generalization}")
    return train_loop(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, save_dir, prepare_baseline_batch, args.max_lr, args.max_norm)


def add_model_switch_args(parser):
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sft", action=argparse.BooleanOptionalAction, default=True)
