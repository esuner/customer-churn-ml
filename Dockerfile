FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Solo las dependencias de la API (sin Jupyter, MLflow ni DVC): imagen liviana
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY src/__init__.py ./src/__init__.py
COPY src/inference ./src/inference
# Modelo exportado del Model Registry (ver scripts/export_model.py); no hay secretos en la imagen
COPY models/churn-classifier ./models/churn-classifier

RUN useradd --create-home appuser
USER appuser

ENV MODEL_DIR=/app/models/churn-classifier
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
