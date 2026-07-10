import csv
import math
import time
from pathlib import Path

import torch


DEFAULT_TEMP_MEAN = 10.9198
DEFAULT_TEMP_STD = 14.4475
DEFAULT_CHANNEL_NAMES = ("Mean", "Max", "Min")


class AccuracyAccumulator:
    def __init__(self, output_channels=3, temp_mean=DEFAULT_TEMP_MEAN, temp_std=DEFAULT_TEMP_STD):
        self.output_channels = output_channels
        self.temp_mean = temp_mean
        self.temp_std = temp_std
        self.sse = [0.0] * output_channels
        self.sae = [0.0] * output_channels
        self.bias = [0.0] * output_channels
        self.target_sum = [0.0] * output_channels
        self.target_sq_sum = [0.0] * output_channels
        self.pixels = [0] * output_channels

    def update(self, preds, targets, masks):
        masks = _expand_masks(masks, preds)

        for channel in range(self.output_channels):
            mask = masks[:, channel, :, :].bool()
            if not mask.any():
                continue

            pred_values = preds[:, channel, :, :][mask].detach().double()
            target_values = targets[:, channel, :, :][mask].detach().double()
            pred_values = pred_values * self.temp_std + self.temp_mean
            target_values = target_values * self.temp_std + self.temp_mean
            diff = pred_values - target_values

            self.sse[channel] += torch.sum(diff * diff).item()
            self.sae[channel] += torch.sum(torch.abs(diff)).item()
            self.bias[channel] += torch.sum(diff).item()
            self.target_sum[channel] += torch.sum(target_values).item()
            self.target_sq_sum[channel] += torch.sum(target_values * target_values).item()
            self.pixels[channel] += int(target_values.numel())

    def compute(self):
        metrics = {"RMSE": [], "MAE": [], "MBE": [], "R2": []}

        for channel in range(self.output_channels):
            count = self.pixels[channel]
            if count == 0:
                metrics["RMSE"].append(0.0)
                metrics["MAE"].append(0.0)
                metrics["MBE"].append(0.0)
                metrics["R2"].append(0.0)
                continue

            mse = self.sse[channel] / count
            ss_tot = self.target_sq_sum[channel] - (self.target_sum[channel] ** 2) / count
            r2 = 1.0 - self.sse[channel] / ss_tot if ss_tot > 0 else 0.0
            metrics["RMSE"].append(math.sqrt(mse))
            metrics["MAE"].append(self.sae[channel] / count)
            metrics["MBE"].append(self.bias[channel] / count)
            metrics["R2"].append(r2)

        return {
            "metrics": metrics,
            "pixels": list(self.pixels),
            "average_rmse": _average(metrics["RMSE"]),
            "average_mae": _average(metrics["MAE"]),
            "average_mbe": _average(metrics["MBE"]),
            "average_r2": _average(metrics["R2"]),
        }


def evaluate_model(
    model,
    test_loader,
    device,
    prepare_batch_fn,
    denormalize_params=None,
    output_channels=3,
    print_interval=50,
):
    params = denormalize_params or {"mean": DEFAULT_TEMP_MEAN, "std": DEFAULT_TEMP_STD}
    accumulator = AccuracyAccumulator(
        output_channels=output_channels,
        temp_mean=params["mean"],
        temp_std=params["std"],
    )
    model = model.to(device)
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(test_loader, start=1):
            inputs, targets, extras = prepare_batch_fn(raw_batch, device)
            preds = model(inputs)
            masks = extras.get("mask", extras.get("masks"))
            if masks is None:
                masks = torch.ones_like(targets[:, :1, :, :])
            accumulator.update(preds, targets, masks)

            if print_interval and batch_idx % print_interval == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} batches")

    results = accumulator.compute()
    results["total_time"] = time.time() - start_time
    results["num_samples"] = len(test_loader.dataset)
    return results


def print_metric_table(results, channel_names=DEFAULT_CHANNEL_NAMES):
    metrics = results["metrics"]
    names = list(channel_names)[: len(metrics["RMSE"])]
    header = f"{'Metric':<8} | " + " | ".join(f"{name:>10}" for name in names) + " | Average"
    print(header)
    print("-" * len(header))
    for metric_name in ("RMSE", "MAE", "MBE", "R2"):
        values = metrics[metric_name]
        average = _average(values)
        row = f"{metric_name:<8} | " + " | ".join(f"{value:>10.4f}" for value in values)
        print(f"{row} | {average:>7.4f}")
    print(f"Pixels   | " + " | ".join(f"{value:>10}" for value in results["pixels"]))
    print(f"Samples: {results['num_samples']} | Time: {results['total_time']:.2f}s")


def write_metric_csv(path, results, channel_names=DEFAULT_CHANNEL_NAMES):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = results["metrics"]
    names = list(channel_names)[: len(metrics["RMSE"])]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Channel", "RMSE", "MAE", "MBE", "R2", "Pixels"])
        writer.writeheader()
        for idx, name in enumerate(names):
            writer.writerow(
                {
                    "Channel": name,
                    "RMSE": f"{metrics['RMSE'][idx]:.6f}",
                    "MAE": f"{metrics['MAE'][idx]:.6f}",
                    "MBE": f"{metrics['MBE'][idx]:.6f}",
                    "R2": f"{metrics['R2'][idx]:.6f}",
                    "Pixels": results["pixels"][idx],
                }
            )


def _expand_masks(masks, preds):
    masks = masks.to(preds.device)
    if masks.dim() == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)
    elif masks.dim() == 3:
        masks = masks.unsqueeze(1)
    if masks.shape[1] == 1:
        masks = masks.expand_as(preds)
    return masks


def _average(values):
    return sum(values) / len(values) if values else 0.0
