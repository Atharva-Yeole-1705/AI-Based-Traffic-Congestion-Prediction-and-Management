"""Synthetic four-approach signal benchmark driven by trained traffic forecasts."""

from __future__ import annotations

import math
import random
from collections import deque
from datetime import datetime

CLEARANCE_SECONDS = 3
FIXED_GREEN_SECONDS = 27
MIN_GREEN_SECONDS = 8
MAX_GREEN_SECONDS = 90
SATURATION_FLOW_PER_HOUR = 1800
DEMAND_SCALE = 0.25


def allocate_green_times(
    demands: list[float],
    green_budget: int,
    minimum: int = MIN_GREEN_SECONDS,
    maximum: int = MAX_GREEN_SECONDS,
) -> list[int]:
    """Allocate a fixed green budget proportionally, respecting phase bounds."""
    if not demands or any(value < 0 for value in demands):
        raise ValueError("Demands must be a non-empty list of non-negative values")
    if minimum * len(demands) > green_budget or maximum * len(demands) < green_budget:
        raise ValueError("Green budget cannot satisfy the per-approach bounds")

    allocations = [minimum] * len(demands)
    remaining = green_budget - sum(allocations)
    active = set(range(len(demands)))
    weights = [max(value, 0.0) for value in demands]
    while remaining and active:
        total_weight = sum(weights[index] for index in active)
        if total_weight == 0:
            weights = [1.0] * len(demands)
            total_weight = len(active)
        shares = {
            index: remaining * weights[index] / total_weight for index in active
        }
        saturated = [
            index
            for index in active
            if shares[index] >= maximum - allocations[index]
        ]
        if saturated:
            for index in saturated:
                increment = maximum - allocations[index]
                allocations[index] += increment
                remaining -= increment
                active.remove(index)
            continue
        increments = {index: math.floor(shares[index]) for index in active}
        for index, increment in increments.items():
            allocations[index] += increment
            remaining -= increment
        remainders = sorted(
            active, key=lambda index: shares[index] - increments[index], reverse=True
        )
        for index in remainders[:remaining]:
            allocations[index] += 1
            remaining -= 1
        break
    return allocations


def synthetic_direction_shares(timestamp: str, seed: int, sample_index: int) -> list[float]:
    """Create an explicit synthetic directional split for a single-station series."""
    try:
        hour = datetime.fromisoformat(timestamp).hour
    except ValueError:
        hour = sample_index % 24
    if 6 <= hour <= 9:
        profile = [0.52, 0.18, 0.18, 0.12]
    elif 16 <= hour <= 19:
        profile = [0.14, 0.18, 0.54, 0.14]
    else:
        profile = [0.28, 0.24, 0.26, 0.22]
    rng = random.Random(seed * 1_000_003 + sample_index)
    perturbed = [weight * rng.uniform(0.85, 1.15) for weight in profile]
    total = sum(perturbed)
    return [weight / total for weight in perturbed]


def forecast_cycle(
    predicted_volume: float, directional_shares: list[float]
) -> tuple[list[int], int]:
    """Use forecast demand for Webster-style cycle length and phase allocation."""
    predicted_demand = max(0.0, predicted_volume) * DEMAND_SCALE
    approach_demands = [predicted_demand * share for share in directional_shares]
    critical_flow_ratio = sum(approach_demands) / SATURATION_FLOW_PER_HOUR
    critical_flow_ratio = min(critical_flow_ratio, 0.90)
    lost_time = CLEARANCE_SECONDS * len(approach_demands)
    cycle_length = round((1.5 * lost_time + 5) / (1 - critical_flow_ratio))
    cycle_length = max(60, min(180, cycle_length))
    green_budget = cycle_length - lost_time
    greens = allocate_green_times(approach_demands, green_budget)
    return greens, cycle_length


def simulate_policy(
    actual_volumes: list[float],
    predicted_volumes: list[float],
    timestamps: list[str],
    seed: int,
    policy: str,
) -> dict[str, float | int | str]:
    """Run a deterministic, second-resolution queue model for matched hourly demand."""
    if not (len(actual_volumes) == len(predicted_volumes) == len(timestamps)):
        raise ValueError("Actuals, predictions, and timestamps must have equal length")
    if not actual_volumes:
        raise ValueError("At least one traffic observation is required")
    if policy not in {"fixed", "forecast"}:
        raise ValueError(f"Unknown signal policy: {policy}")

    queues: list[deque[int]] = [deque() for _ in range(4)]
    arrival_remainders = [0.0] * 4
    service_remainders = [0.0] * 4
    total_arrivals = 0
    total_served = 0
    total_wait_seconds = 0
    queue_area = 0
    maximum_queue = 0
    absolute_second = 0
    cycle_position = 0
    greens = [FIXED_GREEN_SECONDS] * 4
    cycle_length = (FIXED_GREEN_SECONDS + CLEARANCE_SECONDS) * 4
    shares_per_hour = [
        synthetic_direction_shares(timestamp, seed, index)
        for index, timestamp in enumerate(timestamps)
    ]

    for hour_index, actual_volume in enumerate(actual_volumes):
        if actual_volume < 0 or predicted_volumes[hour_index] < 0:
            raise ValueError("Traffic volumes must be non-negative")
        shares = shares_per_hour[hour_index]
        if policy == "forecast":
            greens, cycle_length = forecast_cycle(
                predicted_volumes[hour_index], shares
            )
        else:
            greens = [FIXED_GREEN_SECONDS] * 4
            cycle_length = (FIXED_GREEN_SECONDS + CLEARANCE_SECONDS) * 4

        actual_demand = actual_volume * DEMAND_SCALE
        arrivals_per_second = [actual_demand * share / 3600 for share in shares]
        phase_starts: list[int] = []
        phase_end = 0
        for green in greens:
            phase_starts.append(phase_end)
            phase_end += green + CLEARANCE_SECONDS

        for _ in range(3600):
            cycle_position %= cycle_length
            active_direction = -1
            for direction, start in enumerate(phase_starts):
                if start <= cycle_position < start + greens[direction]:
                    active_direction = direction
                    break

            for direction in range(4):
                arrival_remainders[direction] += arrivals_per_second[direction]
                arriving = int(arrival_remainders[direction])
                arrival_remainders[direction] -= arriving
                for _ in range(arriving):
                    queues[direction].append(absolute_second)
                    total_arrivals += 1

            if active_direction >= 0 and queues[active_direction]:
                service_remainders[active_direction] += (
                    SATURATION_FLOW_PER_HOUR / 3600
                )
                departing = int(service_remainders[active_direction])
                service_remainders[active_direction] -= departing
                for _ in range(departing):
                    if queues[active_direction]:
                        arrival_time = queues[active_direction].popleft()
                        total_wait_seconds += absolute_second - arrival_time
                        total_served += 1

            current_queue = sum(len(queue) for queue in queues)
            queue_area += current_queue
            maximum_queue = max(maximum_queue, current_queue)
            absolute_second += 1
            cycle_position += 1

    mean_wait = total_wait_seconds / total_served if total_served else 0.0
    mean_queue = queue_area / absolute_second if absolute_second else 0.0
    return {
        "policy": policy,
        "seed": seed,
        "hours": len(actual_volumes),
        "arrivals": total_arrivals,
        "vehicles_passed": total_served,
        "mean_wait_seconds_served": mean_wait,
        "mean_wait_seconds_per_arrival_to_horizon": (
            queue_area / total_arrivals if total_arrivals else 0.0
        ),
        "mean_queue_vehicles": mean_queue,
        "max_queue_vehicles": maximum_queue,
        "vehicles_remaining": sum(len(queue) for queue in queues),
    }
