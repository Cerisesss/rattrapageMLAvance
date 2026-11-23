from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_NAMES = [
    "setosa",
    "versicolor",
    "virginica",
]


def load_iris_frame() -> tuple[pd.DataFrame, pd.Series]:
    iris = load_iris(as_frame=True)
    frame = iris.frame
    frame.columns = FEATURE_NAMES + ["target"]
    return frame[FEATURE_NAMES], frame["target"]
