"""Evaluation harness, performance metrics, and interactive Plotly visualizations."""

from pricer.evaluation.metrics import calculate_metrics, calculate_confidence_interval
from pricer.evaluation.evaluator import (
    BenchmarkEvaluator,
    EvaluationResult,
    evaluate,
)
from pricer.evaluation.visualizer import (
    plot_scatter,
    plot_error_trend,
    plot_leaderboard,
)

__all__ = [
    "calculate_metrics",
    "calculate_confidence_interval",
    "BenchmarkEvaluator",
    "EvaluationResult",
    "evaluate",
    "plot_scatter",
    "plot_error_trend",
    "plot_leaderboard",
]
