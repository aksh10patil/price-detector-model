"""Pricer: Amazon Product Price Prediction Project

A unified toolkit implementing data ingestion, batch summarization, classical ML baselines,
PyTorch deep residual neural networks, frontier LLMs, and open-source QLoRA fine-tuning.
"""

from pricer.config import *
from pricer.data.schema import Item
from pricer.data.parser import parse, scrub, simplify
from pricer.data.loader import ItemLoader
from pricer.data.summarizer import Preprocessor
from pricer.data.batch_processor import BatchProcessor
from pricer.data.prompt_formatter import PromptFormatter
from pricer.models.baselines import (
    ConstantPricer,
    LinearRegressionPricer,
    NLPLearningPricer,
    RandomForestPricer,
    XGBoostPricer,
    HumanPricer,
)
from pricer.models.dnn import (
    DeepNeuralNetwork,
    ResidualBlock,
    DeepNeuralNetworkRunner,
)
from pricer.models.frontier import FrontierPricer
from pricer.models.qlora_model import QLoRAPricer
from pricer.evaluation.evaluator import BenchmarkEvaluator, EvaluationResult, evaluate
from pricer.evaluation.metrics import calculate_metrics, calculate_confidence_interval
from pricer.evaluation.visualizer import plot_scatter, plot_error_trend, plot_leaderboard

__all__ = [
    # Schema & Data
    "Item",
    "parse",
    "scrub",
    "simplify",
    "ItemLoader",
    "Preprocessor",
    "BatchProcessor",
    "PromptFormatter",
    # Models
    "ConstantPricer",
    "LinearRegressionPricer",
    "NLPLearningPricer",
    "RandomForestPricer",
    "XGBoostPricer",
    "HumanPricer",
    "DeepNeuralNetwork",
    "ResidualBlock",
    "DeepNeuralNetworkRunner",
    "FrontierPricer",
    "QLoRAPricer",
    # Evaluation
    "BenchmarkEvaluator",
    "EvaluationResult",
    "evaluate",
    "calculate_metrics",
    "calculate_confidence_interval",
    "plot_scatter",
    "plot_error_trend",
    "plot_leaderboard",
]
