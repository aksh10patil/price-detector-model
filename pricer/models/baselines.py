import os
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
from pricer.data.schema import Item
from pricer.config import DATA_DIR


class ConstantPricer:
    """Predicts the mean price from training data."""

    def __init__(self, train_items: list[Item]):
        prices = [item.price for item in train_items]
        self.mean_price = float(np.mean(prices)) if prices else 50.0

    def predict(self, item: Item) -> float:
        return self.mean_price

    def __call__(self, item: Item) -> float:
        return self.predict(item)


class LinearRegressionPricer:
    """Predicts price using tabular metadata features (weight, text length)."""

    def __init__(self):
        self.model = LinearRegression()

    @staticmethod
    def extract_features(item: Item) -> dict:
        summary_text = item.summary or item.full
        return {
            "weight": item.weight or 0.0,
            "weight_unknown": 1 if (item.weight or 0) == 0 else 0,
            "text_length": len(summary_text),
        }

    def fit(self, train_items: list[Item]):
        X = pd.DataFrame([self.extract_features(it) for it in train_items])
        y = np.array([it.price for it in train_items])
        self.model.fit(X, y)

    def predict(self, item: Item) -> float:
        X = pd.DataFrame([self.extract_features(item)])
        return float(self.model.predict(X)[0])

    def __call__(self, item: Item) -> float:
        return self.predict(item)


class NLPLearningPricer:
    """Predicts price using top N bag-of-words text features."""

    def __init__(self, max_features: int = 1000):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = Ridge(alpha=1.0)

    def fit(self, train_items: list[Item]):
        texts = [it.summary or it.full for it in train_items]
        y = np.array([it.price for it in train_items])
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)

    def predict(self, item: Item) -> float:
        text = item.summary or item.full
        X = self.vectorizer.transform([text])
        return float(self.model.predict(X)[0])

    def __call__(self, item: Item) -> float:
        return self.predict(item)


class RandomForestPricer:
    """Random Forest regressor combining text token features and tabular metadata."""

    def __init__(self, n_estimators: int = 50, max_features: int = 500):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    def fit(self, train_items: list[Item]):
        texts = [it.summary or it.full for it in train_items]
        X_text = self.vectorizer.fit_transform(texts).toarray()
        X_tab = np.array([[it.weight or 0, len(it.summary or it.full)] for it in train_items])
        X = np.hstack([X_text, X_tab])
        y = np.array([it.price for it in train_items])
        self.model.fit(X, y)

    def predict(self, item: Item) -> float:
        text = item.summary or item.full
        X_text = self.vectorizer.transform([text]).toarray()
        X_tab = np.array([[item.weight or 0, len(text)]])
        X = np.hstack([X_text, X_tab])
        return float(self.model.predict(X)[0])

    def __call__(self, item: Item) -> float:
        return self.predict(item)


class XGBoostPricer:
    """Gradient boosted decision trees for price prediction."""

    def __init__(self, n_estimators: int = 100, max_features: int = 500):
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = XGBRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)

    def fit(self, train_items: list[Item]):
        texts = [it.summary or it.full for it in train_items]
        X_text = self.vectorizer.fit_transform(texts).toarray()
        X_tab = np.array([[it.weight or 0, len(it.summary or it.full)] for it in train_items])
        X = np.hstack([X_text, X_tab])
        y = np.array([it.price for it in train_items])
        self.model.fit(X, y)

    def predict(self, item: Item) -> float:
        text = item.summary or item.full
        X_text = self.vectorizer.transform([text]).toarray()
        X_tab = np.array([[item.weight or 0, len(text)]])
        X = np.hstack([X_text, X_tab])
        return float(self.model.predict(X)[0])

    def __call__(self, item: Item) -> float:
        return self.predict(item)


class HumanPricer:
    """Pricer backed by recorded human expert predictions."""

    def __init__(self, test_items: list[Item], predictions_file: str = "human_out.csv"):
        self.test_items = test_items
        path = DATA_DIR / predictions_file
        if not path.exists():
            path = Path(predictions_file)
        self.predictions = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) > 1:
                        try:
                            self.predictions.append(float(row[1]))
                        except ValueError:
                            pass

    def predict(self, item: Item) -> float:
        try:
            idx = self.test_items.index(item)
            if idx < len(self.predictions):
                return self.predictions[idx]
        except (ValueError, IndexError):
            pass
        return 50.0

    def __call__(self, item: Item) -> float:
        return self.predict(item)
