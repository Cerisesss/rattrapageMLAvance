from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def save_model(
    model_path: str | Path,
    model: Any,
    feature_names: list[str],
    target_names: list[str],
    comparison: dict[str, Any],
) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "feature_names": feature_names,
            "target_names": target_names,
            "comparison": comparison,
        },
        path,
    )


def load_model(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {path}. Lancez d'abord: python scripts/train.py"
        )
    return joblib.load(path)


def predict_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    target_names: list[str],
) -> dict[str, Any]:
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
