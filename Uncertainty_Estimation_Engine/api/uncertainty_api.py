"""
FastAPI REST interface for the Uncertainty Estimation Engine.
Exposes prediction and calibration endpoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.uncertainty_engine import UncertaintyEngine
from models.gaussian_process import GaussianProcessModel
from models.dropout_model import MCDropoutModel

app = FastAPI(
    title="Uncertainty Estimation Engine",
    description="Production-grade API for uncertainty-aware ML predictions.",
    version="1.0.0",
)

# Initialise engine with default models
engine = UncertaintyEngine(config={"mc_samples": 50, "confidence_level": 0.1})
gp = GaussianProcessModel(name="gp")
dropout = MCDropoutModel(input_dim=5, hidden_dims=[32, 16], name="dropout")
engine.register_model("gp", gp)
engine.register_model("dropout", dropout)


class PredictRequest(BaseModel):
    model_name: str = Field(..., example="gp")
    features: List[List[float]] = Field(..., example=[[1.0, 2.0, 3.0, 4.0, 5.0]])


class PredictResponse(BaseModel):
    model: str
    predictions: List[float]
    epistemic_uncertainty: List[float]
    aleatoric_uncertainty: List[float]
    total_uncertainty: List[float]
    mean_epistemic: float
    mean_aleatoric: float


class ModelInfoResponse(BaseModel):
    registered_models: List[str]
    engine_config: Dict[str, Any]


@app.get("/health")
def health(): return {"status": "ok", "version": "1.0.0"}


@app.get("/models", response_model=ModelInfoResponse)
def list_models():
    return ModelInfoResponse(
        registered_models=engine.list_models(),
        engine_config=engine.config,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.model_name not in engine.list_models():
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found.")
    X = np.array(request.features, dtype=np.float64)
    try:
        result = engine.predict(request.model_name, X)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictResponse(
        model=request.model_name,
        predictions=result.predictions.tolist(),
        epistemic_uncertainty=result.epistemic_uncertainty.tolist(),
        aleatoric_uncertainty=result.aleatoric_uncertainty.tolist(),
        total_uncertainty=result.total_uncertainty.tolist(),
        mean_epistemic=result.mean_epistemic,
        mean_aleatoric=result.mean_aleatoric,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
