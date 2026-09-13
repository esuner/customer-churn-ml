# customer-churn-ml

Sistema de predicción de abandono de clientes (customer churn) para una empresa de
telecomunicaciones. Proyecto Integrador — Laboratorio de Minería de Datos, ISTEA,
segundo cuatrimestre 2026.

El proyecto evoluciona en tres entregas desde un modelo experimental hasta un
servicio de Machine Learning desplegado y monitoreado, según la consigna del curso.

## Problema de negocio

Estimar la probabilidad de abandono (`Churn`: Yes/No) de un cliente a partir de sus
características contractuales y de uso del servicio, y devolver un nivel de riesgo
(`LOW` / `MEDIUM` / `HIGH`) utilizable por otros sistemas (CRM, campañas de retención).

## Dataset

Provisto por la cátedra (sintético, sin datos personales reales). Ver
[`data/README_DATOS.md`](data/README_DATOS.md) y
[`data/metadata/data_dictionary.csv`](data/metadata/data_dictionary.csv) para el
detalle de campos.

- `data/raw/customer_churn_historical.csv` — 7.043 clientes históricos con target. Usado
  para EDA, entrenamiento y evaluación.
- `data/production/customer_churn_current.csv` — 2.500 clientes sin target, reservado
  para el análisis de data drift de la entrega final.
- `data/scoring/scoring_batch.csv` — 100 clientes sin target, para pruebas del servicio
  de inferencia.

Reglas: `customerID` no se usa como predictor; `TotalCharges` tiene faltantes
intencionales a imputar en el pipeline; la partición train/test la define el equipo.

## Estructura del repositorio

```
customer-churn-ml/
├── data/            # Dataset (versionado con DVC, no directamente en Git)
├── notebooks/        # Exploración (EDA); no es el flujo productivo
├── src/
│   ├── data/          # Carga y partición de datos
│   ├── features/       # Preprocessing (Pipeline / ColumnTransformer)
│   ├── training/       # Entrenamiento de modelos + registro en MLflow
│   ├── evaluation/      # Métricas y comparación de modelos
│   └── inference/       # Carga del modelo registrado y predicción
├── tests/            # pytest
├── models/            # Artefactos locales (gitignored; el registro real es MLflow)
└── scripts/           # Entry points ejecutables (ej. entrenar desde CLI)
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Reproducir el entrenamiento

_Pendiente: se documenta en la Entrega 1 junto con el pipeline de entrenamiento._

## Estado del proyecto

- [x] Dataset recibido, verificado (checksums) y organizado.
- [ ] EDA
- [ ] Pipeline de preprocessing (scikit-learn)
- [ ] Entrenamiento y comparación de modelos (baseline / lineal / árbol)
- [ ] Versionado de datos con DVC + DagsHub
- [ ] Tracking de experimentos con MLflow + Model Registry
