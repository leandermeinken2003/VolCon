"""Define functions to process the metrics to evaluate the volume fraction model."""

import os
import json

import torch
from torch import nn

def process_metrics(
        true_volume_fractions: torch.Tensor,
        predicted_volume_fractions: torch.Tensor,
        epoch: int,
        save_path: str,
        train: bool,
) -> None:
    """Output the metrics to console and save them."""
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    mse = mse_func(true_volume_fractions, predicted_volume_fractions)
    mae = mae_func(true_volume_fractions, predicted_volume_fractions)
    mpe = mean_percentage_error(true_volume_fractions, predicted_volume_fractions)
    if train:
        print(f"Epoch {epoch+1}: mae={mae:.5f}, mse={mse:.5f}, mpe: {mpe:.3f} (Training Metrics)")
    else:
        print(f"Epoch {epoch+1}: mae={mae:.5f}, mse={mse:.5f}, mpe: {mpe:.3f} (Testing Metrics)")
    _save_metrics(mae, mse, mpe, save_path)


def _save_metrics(mae: torch.Tensor, mse: torch.Tensor, mpe: torch.Tensor, save_path: str) -> None:
    metrics = _get_metrics_dict(save_path)
    metrics["mse"].append(mse.item())
    metrics["mae"].append(mae.item())
    metrics["mpe"].append(mpe.item())
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)


def _get_metrics_dict(save_path: str) -> dict[str, list]:
    if os.path.exists(save_path):
        with open(save_path, "rb") as file:
            return json.load(file)
    return {"mse": [], "mae": [], "mpe": []}


def mean_percentage_error(
        true_values: torch.Tensor, predicted_values: torch.Tensor,
) -> torch.Tensor:
    """Calculate the mean percentage error."""
    return 100 * torch.mean(torch.abs(true_values - predicted_values) / true_values)
