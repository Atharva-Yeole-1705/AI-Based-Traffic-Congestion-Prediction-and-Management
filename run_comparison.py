"""Compare forecast-informed signal timing with a fixed-time baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from traffic_ai import TrafficVolumeLSTM, chronological_splits, load_traffic_series, restore_volume
from traffic_control import simulate_policy


def load_test_predictions(
    csv_path: Path, checkpoint_path: Path, limit: int
) -> tuple[list[str], list[float], list[float], float]:
    timestamps, volumes = load_traffic_series(csv_path)
    tensors, metadata = chronological_splits(volumes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TrafficVolumeLSTM(int(checkpoint["hidden_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(tensors["test_x"]).tolist()
    test_start = int(metadata["validation_end"])
    actual = volumes[test_start:]
    predicted = restore_volume(
        predicted_normalized, float(checkpoint["mean"]), float(checkpoint["scale"])
    )
    timestamps = timestamps[test_start:]
    if limit > 0:
        actual = actual[:limit]
        predicted = predicted[:limit]
        timestamps = timestamps[:limit]
    if not actual:
        raise ValueError("No held-out test observations to evaluate")
    training_values = volumes[: int(metadata["train_end"])]
    training_values.sort()
    volume_scale = training_values[int(0.90 * (len(training_values) - 1))]
    return timestamps, actual, predicted, volume_scale


def run(data_path: Path, checkpoint_path: Path, output_dir: Path, hours: int, seeds: int) -> dict[str, object]:
    timestamps, actual, predicted, _ = load_test_predictions(
        data_path, checkpoint_path, hours
    )
    results: list[dict[str, float | int | str]] = []
    for seed in range(1, seeds + 1):
        for policy in ("fixed", "forecast"):
            results.append(
                simulate_policy(actual, predicted, timestamps, seed, policy)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "policy_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary: dict[str, object] = {
        "evaluation": {
            "test_hours_per_seed": len(actual),
            "synthetic_directional_scenarios": seeds,
            "arrival_data": "chronological UCI held-out test traffic volumes",
            "policies": {
                "fixed": "27 seconds green per approach plus 3 seconds clearance",
                "forecast": "LSTM one-hour volume forecast; Webster-style cycle and synthetic direction split",
            },
            "simulation_assumptions": [
                "UCI records one westbound highway station, not an intersection.",
                "The four-way approach split is synthetic and perturbed by each recorded seed.",
                "Highway volume is scaled to 25% to create a synthetic intersection demand.",
                "Service is one vehicle per two green seconds per approach.",
                "These are queue-model results, not measured real-world intersection outcomes.",
            ],
        },
        "per_seed_results": results,
    }
    for policy in ("fixed", "forecast"):
        policy_results = [row for row in results if row["policy"] == policy]
        summary[policy] = {
            metric: sum(float(row[metric]) for row in policy_results)
            / len(policy_results)
            for metric in (
                "vehicles_passed",
                "mean_wait_seconds_served",
                "mean_wait_seconds_per_arrival_to_horizon",
                "mean_queue_vehicles",
                "max_queue_vehicles",
                "vehicles_remaining",
            )
        }
    fixed_passed = float(summary["fixed"]["vehicles_passed"])  # type: ignore[index]
    forecast_passed = float(summary["forecast"]["vehicles_passed"])  # type: ignore[index]
    fixed_wait = float(summary["fixed"]["mean_wait_seconds_served"])  # type: ignore[index]
    forecast_wait = float(summary["forecast"]["mean_wait_seconds_served"])  # type: ignore[index]
    fixed_horizon_wait = float(
        summary["fixed"]["mean_wait_seconds_per_arrival_to_horizon"]  # type: ignore[index]
    )
    forecast_horizon_wait = float(
        summary["forecast"]["mean_wait_seconds_per_arrival_to_horizon"]  # type: ignore[index]
    )
    summary["relative_change_percent"] = {
        "vehicles_passed": (
            (forecast_passed - fixed_passed) / fixed_passed * 100
            if fixed_passed
            else 0.0
        ),
        "mean_wait_seconds_served": (
            (forecast_wait - fixed_wait) / fixed_wait * 100 if fixed_wait else 0.0
        ),
        "mean_wait_seconds_per_arrival_to_horizon": (
            (forecast_horizon_wait - fixed_horizon_wait) / fixed_horizon_wait * 100
            if fixed_horizon_wait
            else 0.0
        ),
    }
    (output_dir / "policy_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    metrics = [
        "vehicles_passed",
        "mean_wait_seconds_served",
        "mean_wait_seconds_per_arrival_to_horizon",
        "mean_queue_vehicles",
    ]
    labels = [
        "Vehicles passed",
        "Wait of served vehicles (s)",
        "Wait per arrival to horizon (s)",
        "Mean queue (vehicles)",
    ]
    fixed_means = [
        sum(float(row[metric]) for row in results if row["policy"] == "fixed") / seeds
        for metric in metrics
    ]
    forecast_means = [
        sum(float(row[metric]) for row in results if row["policy"] == "forecast") / seeds
        for metric in metrics
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5))
    for axis, label, fixed, forecast in zip(
        axes, labels, fixed_means, forecast_means
    ):
        bars = axis.bar(["Fixed", "Forecast"], [fixed, forecast], color=["#718096", "#3182ce"])
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f", padding=3)
    figure.suptitle("Matched-demand synthetic signal-policy comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "policy_comparison.png", dpi=160)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Metro_Interstate_Traffic_Volume.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/traffic_lstm.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--hours", type=int, default=168, help="Held-out hours to simulate (default: 1 week)")
    parser.add_argument("--seeds", type=int, default=5, help="Synthetic direction profiles (default: 5)")
    args = parser.parse_args()
    if args.hours <= 0 or args.seeds <= 0:
        parser.error("--hours and --seeds must be positive")
    summary = run(args.data, args.checkpoint, args.output_dir, args.hours, args.seeds)
    print(json.dumps({key: summary[key] for key in ("fixed", "forecast", "relative_change_percent")}, indent=2))


if __name__ == "__main__":
    main()
