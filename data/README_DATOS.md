# Paquete de datos — Proyecto Integrador
## Laboratorio de Minería de Datos — ISTEA

Este paquete acompaña el proyecto de Customer Churn del curso.

### Archivos principales

- `data/raw/customer_churn_historical.csv`
  - 7.043 observaciones históricas.
  - Incluye la variable objetivo `Churn`.
  - Debe utilizarse para EDA, partición train/test, entrenamiento, evaluación y versionado con DVC.

- `data/production/customer_churn_current.csv`
  - 2.500 observaciones que simulan un período posterior de producción.
  - No incluye `Churn`.
  - Debe reservarse para la etapa final de monitoreo y análisis de drift.
  - No utilizar este archivo para entrenar el modelo inicial.

- `data/scoring/scoring_batch.csv`
  - 100 observaciones sin target.
  - Puede utilizarse para pruebas batch, ejemplos de inferencia y validación del servicio.

- `metadata/data_dictionary.csv`
  - Diccionario de campos, tipos y dominios.

- `metadata/schema.json`
  - Esquema general y metadatos del paquete.

- `examples/predict_request_valid.json`
  - Ejemplo de payload válido para `POST /predict`.

- `examples/predict_request_invalid.json`
  - Ejemplo deliberadamente inválido para probar validaciones de Pydantic.

### Reglas de uso

1. `customerID` es un identificador y no debe utilizarse directamente como feature predictiva.
2. Existen faltantes intencionales en `TotalCharges`; deben ser tratados dentro del pipeline.
3. Los datos históricos deben ser incorporados a DVC y no versionados directamente como dataset dentro de Git.
4. El conjunto `production/current` se reserva para la etapa de monitoreo. No debe utilizarse para mejorar el modelo antes de esa etapa.
5. El equipo debe generar su propia división train/test a partir del histórico, usando un procedimiento reproducible.
6. No se entrega una partición "correcta" ni un conjunto de hiperparámetros predefinido: esas decisiones forman parte del trabajo.
7. El dataset es sintético y no contiene datos personales reales.

### Objetivo pedagógico

Los datos fueron construidos para permitir:
- variables numéricas y categóricas;
- valores faltantes;
- clasificación binaria no trivial;
- comparación de varios modelos tradicionales;
- uso de `Pipeline` y `ColumnTransformer`;
- trazabilidad con DVC y MLflow;
- serving mediante FastAPI;
- pruebas de contratos;
- análisis posterior de data drift.
