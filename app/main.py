"""API de inferencia de churn."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.inference.predictor import ChurnPredictor
from src.inference.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ChurnPredictor()
    logger.info("Modelo cargado: churn-classifier v%s", app.state.predictor.model_version)
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Estima la probabilidad de abandono de un cliente y devuelve un nivel de riesgo.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", model_version=request.app.state.predictor.model_version)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request) -> dict:
    predictor = request.app.state.predictor
    try:
        result = predictor.predict(customer.model_dump())
    except Exception:
        logger.exception("Falló la predicción")
        raise HTTPException(status_code=500, detail="Error interno al ejecutar la predicción")
    logger.info(
        "predict prediction=%s probability=%s risk_level=%s model_version=%s",
        result["prediction"], result["probability"], result["risk_level"], result["model_version"],
    )
    return result
