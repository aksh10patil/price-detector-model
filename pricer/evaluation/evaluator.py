import re
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from pricer.evaluation.metrics import calculate_metrics
from pricer.config import DEFAULT_EVAL_SIZE, DEFAULT_EVAL_WORKERS

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}


class EvaluationResult:
    """Stores predictions, ground truths, errors, and calculated metrics."""

    def __init__(self, model_name: str, truths: list[float], guesses: list[float], titles: list[str]):
        self.model_name = model_name
        self.truths = truths
        self.guesses = guesses
        self.titles = titles
        self.errors = [abs(g - t) for g, t in zip(guesses, truths)]
        self.metrics = calculate_metrics(truths, guesses)

    def summary(self) -> str:
        return (
            f"Model: {self.model_name}\n"
            f"  RMSE:        ${self.metrics['rmse']:,.2f}\n"
            f"  MAE:         ${self.metrics['mae']:,.2f}\n"
            f"  R² Score:    {self.metrics['r2']:.1f}%\n"
            f"  Within 10%:  {self.metrics['within_10_pct']:.1f}%\n"
            f"  Within 20%:  {self.metrics['within_20_pct']:.1f}%"
        )


class BenchmarkEvaluator:
    """Universal evaluation harness for both Item objects and prompt dictionaries."""

    def __init__(self, predictor, data: list, title: str | None = None, size: int = DEFAULT_EVAL_SIZE, workers: int = 1):
        self.predictor = predictor
        self.data = data[:size]
        self.size = len(self.data)
        self.title = title or getattr(predictor, "__name__", predictor.__class__.__name__)
        self.workers = workers

    @staticmethod
    def parse_value(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
        return float(match.group()) if match else 0.0

    @staticmethod
    def extract_truth(item) -> float:
        if isinstance(item, dict):
            if "completion" in item:
                return float(item["completion"])
            if "price" in item:
                return float(item["price"])
        return float(getattr(item, "price", 0.0))

    @staticmethod
    def extract_title(item) -> str:
        if isinstance(item, dict):
            prompt = item.get("prompt", "")
            if "Title: " in prompt:
                return prompt.split("Title: ")[1].split("\n")[0][:40]
            return str(item.get("title", "Item"))[:40]
        return getattr(item, "title", "Item")[:40]

    def run_item(self, i: int) -> tuple[str, float, float]:
        item = self.data[i]
        title = self.extract_title(item)
        truth = self.extract_truth(item)
        pred_raw = self.predictor(item)
        guess = self.parse_value(pred_raw)
        return title, truth, guess

    def evaluate(self) -> EvaluationResult:
        titles, truths, guesses = [], [], []

        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                results = list(tqdm(executor.map(self.run_item, range(self.size)), total=self.size, desc=f"Evaluating {self.title}"))
            for t, tr, g in results:
                titles.append(t)
                truths.append(tr)
                guesses.append(g)
        else:
            for i in tqdm(range(self.size), desc=f"Evaluating {self.title}"):
                t, tr, g in self.run_item(i)
                titles.append(t)
                truths.append(tr)
                guesses.append(g)

        result = EvaluationResult(self.title, truths, guesses, titles)
        print("\n" + result.summary())
        return result


def evaluate(predictor, data: list, title: str | None = None, size: int = DEFAULT_EVAL_SIZE, workers: int = 1) -> EvaluationResult:
    """Convenience function matching the capstone interface."""
    evaluator = BenchmarkEvaluator(predictor, data, title=title, size=size, workers=workers)
    return evaluator.evaluate()
