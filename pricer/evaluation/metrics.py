import math
from itertools import accumulate
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(truths: list[float], guesses: list[float]) -> dict:
    """Calculates regression performance metrics."""
    truths_arr = np.array(truths)
    guesses_arr = np.array(guesses)
    errors = np.abs(guesses_arr - truths_arr)

    rmse = float(np.sqrt(mean_squared_error(truths_arr, guesses_arr)))
    mae = float(mean_absolute_error(truths_arr, guesses_arr))
    r2 = float(r2_score(truths_arr, guesses_arr) * 100.0)
    avg_error = float(np.mean(errors))

    # Error tolerances (percentage of predictions within 10% and 20% of true price)
    rel_errors = errors / np.maximum(truths_arr, 1.0)
    within_10_pct = float(np.mean(rel_errors <= 0.10) * 100.0)
    within_20_pct = float(np.mean(rel_errors <= 0.20) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "avg_error": avg_error,
        "within_10_pct": within_10_pct,
        "within_20_pct": within_20_pct,
    }


def calculate_confidence_interval(errors: list[float]) -> tuple[list[float], list[float], float, float]:
    """Calculates running mean and 95% confidence interval series."""
    n = len(errors)
    running_sums = list(accumulate(errors))
    x = list(range(1, n + 1))
    running_means = [s / i for s, i in zip(running_sums, x)]

    running_squares = list(accumulate(e * e for e in errors))
    running_stds = [
        math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0.0
        for i, sq_sum, mean in zip(x, running_squares, running_means)
    ]

    ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0.0 for i, sd in zip(x, running_stds)]
    final_mean = running_means[-1] if running_means else 0.0
    final_ci = ci[-1] if ci else 0.0
    return running_means, ci, final_mean, final_ci
