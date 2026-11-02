"""Train and evaluate the UCI traffic-volume LSTM."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from traffic_ai import (
    BATCH_SIZE,
    DATASET_CITATION,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    WINDOW_SIZE,
    TrafficVolumeLSTM,
    chronological_splits,
    download_dataset,
    load_traffic_series,
    regression_metrics,
    restore_volume,
    seed_everything,
)


def train(data_dir: Path, artifact_dir: Path, report_dir: Path) -> dict[str, object]:
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    seed_everything()
    csv_path = download_dataset(data_dir)
    timestamps, volumes = load_traffic_series(csv_path)
    tensors, metadata = chronological_splits(volumes)

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(tensors["train_x"], tensors["train_y"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    model = TrafficVolumeLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        training_loss_sum = 0.0
        training_count = 0
        for sequences, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(sequences), targets)
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item() * len(targets)
            training_count += len(targets)

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(tensors["validation_x"]), tensors["validation_y"]
            ).item()
        training_loss = training_loss_sum / training_count
        history.append(
            {"epoch": epoch, "training_mse": training_loss, "validation_mse": validation_loss}
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(tensors["test_x"]).tolist()
    mean = float(metadata["mean"])
    scale = float(metadata["scale"])
    actual = volumes[int(metadata["validation_end"]) :]
    predicted = restore_volume(predicted_normalized, mean, scale)
    test_metrics = regression_metrics(actual, predicted)
    persistence = volumes[int(metadata["validation_end"]) - 1 : -1]
    persistence_metrics = regression_metrics(actual, persistence)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "traffic_lstm.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "mean": mean,
        "scale": scale,
        "window_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "train_volume_90th_percentile": sorted(volumes[: int(metadata["train_end"])])[
            int(0.90 * (int(metadata["train_end"]) - 1))
        ],
        "dataset_url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
    }
    torch.save(checkpoint, checkpoint_path)

    metrics: dict[str, object] = {
        "dataset": {
            "name": "UCI Metro Interstate Traffic Volume",
            "citation": DATASET_CITATION,
            "station": "I-94 westbound ATR station 301, Minneapolis-St Paul",
            "observations": len(volumes),
            "target": "traffic_volume (vehicles per hour)",
            "chronological_split": "70% train / 15% validation / 15% test",
        },
        "model": {
            "architecture": "1-layer LSTM (1 input feature, 32 hidden units) + linear output",
            "framework": f"PyTorch {torch.__version__}",
            "training_device": "CPU",
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "input_feature": "traffic_volume only",
            "input_window_hours": WINDOW_SIZE,
            "forecast_horizon_hours": 1,
            "normalization": "z-score using training-partition mean and standard deviation",
            "optimizer": "Adam",
            "loss": "mean squared error",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_validation_mse_normalized": best_validation_loss,
        },
        "test_metrics": test_metrics,
        "persistence_baseline_metrics": persistence_metrics,
        "history": history,
        "artifact": str(checkpoint_path),
        "limitations": [
            "The source is a single westbound highway station, not a four-way intersection.",
            "The model predicts one hour ahead from the previous 24 hourly volume readings.",
            "Reported prediction metrics are from a chronological held-out test partition.",
        ],
    }
    (report_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    predictions_path = report_dir / "test_predictions.csv"
    test_start = int(metadata["validation_end"])
    with predictions_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["date_time", "actual_volume", "lstm_prediction", "persistence_prediction"]
        )
        for index, (actual_value, prediction, baseline) in enumerate(
            zip(actual, predicted, persistence)
        ):
            writer.writerow(
                [
                    timestamps[test_start + index],
                    round(actual_value, 3),
                    round(prediction, 3),
                    round(baseline, 3),
                ]
            )

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [item["training_mse"] for item in history], label="Training MSE")
    plt.plot(epochs, [item["validation_mse"] for item in history], label="Validation MSE")
    plt.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("UCI traffic-volume LSTM training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "training_curves.png", dpi=160)
    plt.close()

    sample_size = min(336, len(actual))
    plt.figure(figsize=(11, 4.5))
    plt.plot(actual[:sample_size], label="Observed", linewidth=1.2)
    plt.plot(predicted[:sample_size], label="LSTM forecast", linewidth=1.1)
    plt.plot(persistence[:sample_size], label="Persistence baseline", linewidth=0.9)
    plt.xlabel("Held-out hours")
    plt.ylabel("Vehicles per hour")
    plt.title("One-step traffic-volume forecasts (first 14 test days)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "forecast_vs_actual.png", dpi=160)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    results = train(args.data_dir, args.artifact_dir, args.report_dir)
    print(json.dumps({key: results[key] for key in ("model", "test_metrics", "persistence_baseline_metrics")}, indent=2))


if __name__ == "__main__":
    main()
