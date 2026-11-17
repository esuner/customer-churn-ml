"""API de inferencia de churn."""
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.logging_config import configure_logging, request_id_var
from src.inference.predictor import ChurnPredictor
from src.inference.schemas import CustomerFeatures, HealthResponse, PredictionResponse

configure_logging()
logger = logging.getLogger("churn-api")

VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ChurnPredictor()
    logger.info(
        "service_started",
        extra={"fields": {"model_name": "churn-classifier", "model_version": app.state.predictor.model_version}},
    )
    yield
    logger.info("service_stopped")


app = FastAPI(
    title="Churn Prediction API",
    description="Estima la probabilidad de abandono de un cliente y devuelve un nivel de riesgo.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if VALID_REQUEST_ID.match(incoming) else str(uuid.uuid4())
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled_error")
        response = JSONResponse(
            status_code=500, content={"detail": "Error interno del servidor", "request_id": request_id}
        )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    predictor = getattr(request.app.state, "predictor", None)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(duration_ms)
    logger.info(
        "request",
        extra={
            "fields": {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "model_version": predictor.model_version if predictor else None,
            }
        },
    )
    request_id_var.reset(token)
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", model_version=request.app.state.predictor.model_version)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request) -> dict:
    predictor = request.app.state.predictor
    try:
        result = predictor.predict(customer.model_dump())
    except Exception:
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="Error interno al ejecutar la predicción")
    logger.info(
        "prediction",
        extra={
            "fields": {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "risk_level": result["risk_level"],
                "model_version": result["model_version"],
            }
        },
    )
    return {**result, "request_id": request_id_var.get()}
