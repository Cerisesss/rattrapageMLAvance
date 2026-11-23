from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse

from src.model import load_model


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "iris_model.joblib"


class FlowerInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionOutput(BaseModel):
    predicted_species: str
    probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.modele = load_model(MODEL_PATH)
    except FileNotFoundError:
        app.state.modele = None
    yield


app = FastAPI(
    title="Iris API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if app.state.modele is not None else "model_not_loaded",
        # "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(flower: FlowerInput) -> PredictionOutput:
    modele = app.state.modele
    if modele is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    frame = pd.DataFrame(
        [flower.model_dump()],
        columns=modele["feature_names"],
    )
    predicted_index = int(modele["pipeline"].predict(frame)[0])
    probabilities = modele["pipeline"].predict_proba(frame)[0]
    return PredictionOutput(
        predicted_species=modele["target_names"][predicted_index],
        probabilities={
            name: float(prob)
            for name, prob in zip(modele["target_names"], probabilities)
        },
    )
