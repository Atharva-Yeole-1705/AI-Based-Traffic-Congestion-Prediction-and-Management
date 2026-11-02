"""Dataset preparation and reproducible LSTM traffic-volume forecasting."""

from __future__ import annotations

import csv
import gzip
import io
import math
import random
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch import nn

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
DATASET_CITATION = (
    "Hogue, J. (2019). Metro Interstate Traffic Volume "
    "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B."
)
WINDOW_SIZE = 24
HIDDEN_SIZE = 32
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.001
SEED = 42


class TrafficVolumeLSTM(nn.Module):
    """One-layer LSTM that forecasts the next hourly volume from 24 hours."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequences)
        return self.output(outputs[:, -1, :]).squeeze(-1)


def download_dataset(data_dir: Path) -> Path:
    """Download and extract the official UCI CSV unless it is already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "Metro_Interstate_Traffic_Volume.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        return csv_path

    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "traffic-management-course-project/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith(
                ("metro_interstate_traffic_volume.csv", ".csv.gz")
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one UCI traffic CSV in archive; found {candidates!r}"
            )
        payload = archive.read(candidates[0])

    if candidates[0].lower().endswith(".gz"):
        payload = gzip.decompress(payload)
    csv_path.write_bytes(payload)
    return csv_path


def load_traffic_series(csv_path: Path) -> tuple[list[str], list[float]]:
    """Read ordered timestamps and traffic volumes from the UCI CSV."""
    timestamps: list[str] = []
    volumes: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"date_time", "traffic_volume"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("Dataset must contain date_time and traffic_volume columns")
        for row in reader:
            timestamp = (row.get("date_time") or "").strip()
            raw_volume = (row.get("traffic_volume") or "").strip()
            if not timestamp or not raw_volume:
                continue
            volume = float(raw_volume)
            if not math.isfinite(volume) or volume < 0:
                raise ValueError(f"Invalid traffic volume {raw_volume!r}")
            if timestamps and timestamp < timestamps[-1]:
                raise ValueError("Dataset rows must be in chronological order")
            timestamps.append(timestamp)
            volumes.append(volume)

    if len(volumes) < 1000:
        raise ValueError(f"Dataset unexpectedly small: {len(volumes)} rows")
    return timestamps, volumes


def chronological_splits(
    values: list[float], window_size: int = WINDOW_SIZE
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    """Scale from training data only and split one-step windows by target time."""
    if len(values) <= window_size + 2:
        raise ValueError("Not enough observations for the requested input window")
    train_end = int(len(values) * 0.70)
    validation_end = int(len(values) * 0.85)
    if train_end <= window_size:
        raise ValueError("Training partition is smaller than the input window")

    mean = sum(values[:train_end]) / train_end
    variance = sum((value - mean) ** 2 for value in values[:train_end]) / train_end
    scale = math.sqrt(variance) or 1.0
    normalized = [(value - mean) / scale for value in values]

    splits: dict[str, tuple[int, int]] = {
        "train": (window_size, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(values)),
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, (start, stop) in splits.items():
        sequences = [
            normalized[index - window_size : index] for index in range(start, stop)
        ]
        targets = normalized[start:stop]
        tensors[f"{name}_x"] = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        tensors[f"{name}_y"] = torch.tensor(targets, dtype=torch.float32)

    metadata: dict[str, int | float] = {
        "window_size": window_size,
        "train_end": train_end,
        "validation_end": validation_end,
        "mean": mean,
        "scale": scale,
    }
    return tensors, metadata


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def regression_metrics(
    actual: list[float], predicted: list[float]
) -> dict[str, float]:
    """Return MAE, RMSE, and WAPE in traffic-volume units."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must be non-empty and aligned")
    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    absolute_error = sum(abs(error) for error in errors)
    total_actual = sum(abs(value) for value in actual)
    return {
        "mae": absolute_error / len(errors),
        "rmse": math.sqrt(sum(error**2 for error in errors) / len(errors)),
        "wape_percent": 100 * absolute_error / total_actual if total_actual else 0.0,
    }


def restore_volume(normalized: list[float], mean: float, scale: float) -> list[float]:
    return [max(0.0, value * scale + mean) for value in normalized]


def load_forecaster(checkpoint_path: Path) -> tuple[TrafficVolumeLSTM, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TrafficVolumeLSTM(int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_next_volume(
    model: TrafficVolumeLSTM, checkpoint: dict[str, object], recent_volumes: list[float]
) -> float:
    window_size = int(checkpoint["window_size"])
    if len(recent_volumes) != window_size:
        raise ValueError(f"Expected exactly {window_size} recent hourly observations")
    mean = float(checkpoint["mean"])
    scale = float(checkpoint["scale"])
    normalized = [(value - mean) / scale for value in recent_volumes]
    sequence = torch.tensor(normalized, dtype=torch.float32).view(1, window_size, 1)
    with torch.no_grad():
        prediction = float(model(sequence).item())
    return restore_volume([prediction], mean, scale)[0]
