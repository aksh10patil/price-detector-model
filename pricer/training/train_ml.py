"""Pipeline to train and evaluate classical Machine Learning baselines."""

import argparse
from pricer.data.schema import Item
from pricer.models.baselines import (
    ConstantPricer,
    LinearRegressionPricer,
    NLPLearningPricer,
    RandomForestPricer,
    XGBoostPricer,
    HumanPricer,
)
from pricer.evaluation.evaluator import evaluate
from pricer.config import DEFAULT_LITE_DATASET


def train_and_eval_all(dataset_name: str = DEFAULT_LITE_DATASET, test_size: int = 200):
    print(f"Loading dataset: {dataset_name}...")
    train, val, test = Item.from_hub(dataset_name)
    test_sample = test[:test_size]

    models = [
        ("Constant Pricer", ConstantPricer(train)),
        ("Linear Regression (Metadata)", LinearRegressionPricer()),
        ("NLP + Ridge (Bag-of-Words)", NLPLearningPricer()),
        ("Random Forest", RandomForestPricer()),
        ("XGBoost", XGBoostPricer()),
        ("Human Benchmark", HumanPricer(test)),
    ]

    results = []
    for name, model in models:
        if hasattr(model, "fit"):
            print(f"\nTraining {name}...")
            model.fit(train)
        print(f"Evaluating {name} on {len(test_sample)} holdout test items...")
        res = evaluate(model, test_sample, title=name)
        results.append((name, res.metrics["rmse"], res.metrics["r2"]))

    print("\n" + "=" * 50)
    print("BASELINE ML BENCHMARK SUMMARY")
    print("=" * 50)
    for name, rmse, r2 in results:
        print(f"{name:<30} | RMSE: ${rmse:>6.2f} | R²: {r2:>5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--dataset", type=str, default=DEFAULT_LITE_DATASET)
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    train_and_eval_all(args.dataset, args.size)
