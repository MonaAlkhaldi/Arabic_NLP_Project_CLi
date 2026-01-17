from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainResult:
    name: str
    accuracy: float
    report: str
    cm: np.ndarray
    model: Any


def get_models(models: list[str] | None = None) -> dict[str, Any]:
    """
    Return a dictionary of sklearn models to train.
    """
    all_models = {
        "lr": LogisticRegression(max_iter=2000),
        "svm": LinearSVC(),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    if not models or models == ["all"]:
        return all_models

    chosen = {}
    for m in models:
        if m not in all_models:
            raise ValueError(f"Unknown model '{m}'. Choose from: {list(all_models.keys())} or 'all'.")
        chosen[m] = all_models[m]
    return chosen


def train_and_eval(
    X,
    y,
    model_dict: dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train and evaluate multiple models; return results sorted by accuracy (best first).
    """
    # stratify keeps class distribution in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results: list[TrainResult] = []

    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        cm = confusion_matrix(y_test, preds)
        rep = classification_report(y_test, preds, digits=4)

        results.append(TrainResult(name=name, accuracy=acc, report=rep, cm=cm, model=model))

    results.sort(key=lambda r: r.accuracy, reverse=True)
    return results
