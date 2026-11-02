from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from traffic_ai import TrafficVolumeLSTM, chronological_splits, load_traffic_series, regression_metrics
from traffic_control import allocate_green_times, forecast_cycle, simulate_policy


class TrafficForecastTests(unittest.TestCase):
    def test_chronological_splits_keep_test_targets_after_training(self) -> None:
        values = [float(index) for index in range(100)]
        tensors, metadata = chronological_splits(values, window_size=5)
        self.assertEqual(tensors["train_x"].shape, (65, 5, 1))
        self.assertEqual(tensors["validation_x"].shape, (15, 5, 1))
        self.assertEqual(tensors["test_x"].shape, (15, 5, 1))
        self.assertEqual(metadata["train_end"], 70)
        self.assertEqual(metadata["validation_end"], 85)
        expected = (85 - float(metadata["mean"])) / float(metadata["scale"])
        self.assertAlmostEqual(float(tensors["test_y"][0]), expected, places=6)

    def test_model_returns_one_prediction_per_input_window(self) -> None:
        model = TrafficVolumeLSTM()
        result = model(torch.zeros((3, 24, 1)))
        self.assertEqual(tuple(result.shape), (3,))

    def test_metrics_are_in_original_units(self) -> None:
        result = regression_metrics([10, 20], [12, 17])
        self.assertAlmostEqual(result["mae"], 2.5)
        self.assertAlmostEqual(result["rmse"], (6.5) ** 0.5)
        self.assertAlmostEqual(result["wape_percent"], 5 / 30 * 100)

    def test_loader_reads_official_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "traffic.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=["date_time", "traffic_volume"]
                )
                writer.writeheader()
                for index in range(1001):
                    writer.writerow(
                        {"date_time": f"2020-01-{index + 1:04d}", "traffic_volume": index}
                    )
            timestamps, values = load_traffic_series(path)
            self.assertEqual(len(timestamps), 1001)
            self.assertEqual(values[0], 0)
            self.assertEqual(values[-1], 1000)


class TrafficControlTests(unittest.TestCase):
    def test_green_allocation_obeys_budget_and_bounds(self) -> None:
        result = allocate_green_times([90, 5, 3, 2], 108)
        self.assertEqual(sum(result), 108)
        self.assertTrue(all(8 <= duration <= 90 for duration in result))
        self.assertGreater(result[0], result[1])

    def test_forecast_cycle_is_bounded(self) -> None:
        greens, cycle_length = forecast_cycle(4000, [0.25] * 4)
        self.assertEqual(sum(greens) + 4 * 3, cycle_length)
        self.assertTrue(60 <= cycle_length <= 180)

    def test_forecast_volume_changes_cycle_length(self) -> None:
        _, low_demand_cycle = forecast_cycle(0, [0.25] * 4)
        _, high_demand_cycle = forecast_cycle(7000, [0.25] * 4)
        self.assertLess(low_demand_cycle, high_demand_cycle)

    def test_matched_policy_runs_are_reproducible(self) -> None:
        actual = [800.0, 1200.0]
        predicted = [750.0, 1300.0]
        timestamps = ["2020-01-01 07:00:00", "2020-01-01 17:00:00"]
        first = simulate_policy(actual, predicted, timestamps, 17, "forecast")
        second = simulate_policy(actual, predicted, timestamps, 17, "forecast")
        self.assertEqual(first, second)

    def test_fixed_policy_ignores_model_predictions(self) -> None:
        actual = [800.0]
        timestamps = ["2020-01-01 12:00:00"]
        low_forecast = simulate_policy(actual, [200.0], timestamps, 3, "fixed")
        high_forecast = simulate_policy(actual, [7000.0], timestamps, 3, "fixed")
        self.assertEqual(low_forecast, high_forecast)

    def test_horizon_wait_includes_queued_vehicles(self) -> None:
        result = simulate_policy(
            [800.0], [750.0], ["2020-01-01 12:00:00"], 3, "fixed"
        )
        self.assertAlmostEqual(
            result["mean_wait_seconds_per_arrival_to_horizon"],
            result["mean_queue_vehicles"] * 3600 / result["arrivals"],
        )


if __name__ == "__main__":
    unittest.main()
