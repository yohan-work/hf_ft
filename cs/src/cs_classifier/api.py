"""FastAPI application for local CS-classifier inference."""

from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .inference import ClassifierPredictor, PredictionResult

MAX_TEXT_CHARS = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS, description="Korean customer inquiry")

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value.strip()


class Score(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    label_name: str
    confidence: float
    scores: list[Score]
    model_version: str


def default_predictor() -> ClassifierPredictor:
    return ClassifierPredictor(
        model_path=Path(os.getenv("CS_MODEL_PATH", "models/klue-roberta-cs-v1")),
        labels_path=Path(os.getenv("CS_LABELS_PATH", "configs/labels.yaml")),
        max_length=int(os.getenv("CS_MAX_LENGTH", "128")),
        device=os.getenv("CS_DEVICE", "auto"),
    )


def create_app(predictor_factory: Callable[[], ClassifierPredictor] = default_predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor_factory()
        yield

    app = FastAPI(title="Korean CS Classifier API", version="0.1.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {"error": {"code": "validation_error", "message": "Invalid prediction request", "details": exc.errors()}}
            ),
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        predictor = request.app.state.predictor
        return {"status": "ok", "model_version": os.getenv("CS_MODEL_VERSION", "v1"), "device": str(predictor.device)}

    @app.get("/lab/metrics")
    def lab_metrics() -> dict:
        metrics_path = PROJECT_ROOT / "artifacts/metrics/final-test-metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Final evaluation artifacts are not available")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    @app.get("/", include_in_schema=False)
    def model_lab() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        result: PredictionResult = request.app.state.predictor.predict(payload.text)
        return PredictResponse(
            label=result.label,
            label_name=result.label_name,
            confidence=result.confidence,
            scores=[Score(**score) for score in result.scores],
            model_version=os.getenv("CS_MODEL_VERSION", "v1"),
        )

    return app


app = create_app()
