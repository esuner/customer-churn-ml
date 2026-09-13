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
git clone https://github.com/esuner/customer-churn-ml.git
cd customer-churn-ml
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

### Recuperar los datos (DVC)

Los CSV no están en este repo — se versionan con DVC y viven en DagsHub
(`https://dagshub.com/esuner/customer-churn-ml`). Para traerlos:

1. Pedí que te agreguen como colaborador en el repo de DagsHub (pestaña
   **Collaboration**).
2. Generá tu propio token: en DagsHub, ícono de tu perfil → **My Settings** →
   **Tokens** → **Generate New Token**.
3. En el repo, botón verde **Data** → copiá los bloques **"Add DVC remote"** y
   **"Setup credentials"** y pegalos en tu terminal (con `.venv` activado). El
   segundo bloque guarda tu token en `.dvc/config.local`, que está
   gitignoreado — nunca se comparte ni se sube.
4. Traé los datos reales:
   ```bash
   dvc pull
   ```

## Reproducir el entrenamiento

_Pendiente: se documenta cuando el pipeline de entrenamiento (`src/training/`)
esté armado — ver "Próximos pasos" abajo._

## Estado del proyecto (al 13/09/2026)

### Hecho
- [x] Dataset recibido, verificado (checksums) y organizado en `data/`.
- [x] EDA — [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb). Hallazgos clave:
      ~26% churn (dataset desbalanceado → no usar Accuracy sola), los 26
      faltantes de `TotalCharges` son clientes con `tenure==0` (no se
      descartan, se imputan), `customerID` sin duplicados y excluido como
      predictor, `Contract` y `tenure` fuertemente asociados al churn.
- [x] Repositorio en GitHub, estructura de carpetas recomendada creada.
- [x] Dataset versionado con **DVC**, remote funcional en **DagsHub**
      (`dvc push`/`dvc pull` probados).

### Pendiente para Entrega 1 (22/09/2026 — 19:00 h)

1. **Partición train/test reproducible** — módulo `src/data/load_data.py`
   (arrancado: función `load_raw_data()` que centraliza la conversión de
   `TotalCharges` a numérico, la misma corrección que se hizo a mano en el
   EDA). Falta la función `split_data()`: `train_test_split` con
   `random_state` fijo y `stratify=y` sobre la columna `Churn` (importante
   por el desbalance ~26/74 que se vio en el EDA).
2. **Pipeline de preprocessing** en `src/features/` con
   `ColumnTransformer` de scikit-learn: `SimpleImputer` para
   `TotalCharges`, `OneHotEncoder` para las categóricas, `StandardScaler`
   para las numéricas (al menos para el modelo lineal). Debe ser el mismo
   pipeline el que se use en training e inferencia.
3. **Entrenamiento y comparación de modelos** en `src/training/` +
   `src/evaluation/`: al menos `DummyClassifier` (baseline),
   `LogisticRegression` (lineal) y `RandomForestClassifier` (árbol).
   Métricas a reportar: Precision, Recall, F1, ROC-AUC y matriz de
   confusión (Accuracy sola no alcanza). Justificar qué métrica pesa más
   dado que un falso negativo (cliente que iba a abandonar y no se
   detectó) tiene mayor costo de negocio.
4. **MLflow**: loggear cada corrida (parámetros, métricas, artefactos,
   modelo) contra el tracking server de DagsHub. Se necesitan **al menos 6
   runs relevantes** (ej. variando hiperparámetros de cada modelo, no
   ejecuciones arbitrarias).
5. **Model Registry**: registrar el modelo candidato elegido, con
   justificación de por qué se eligió ese y no otro.
6. **Entrenamiento ejecutable por script**, no por notebook a mano — la
   estructura en `src/` + `scripts/` ya está pensada para esto.
7. Tag Git **`entrega-1`** sobre el commit final que se presente.

### Cómo seguir (para quien retome esto)

- El módulo de partición de datos quedó a mitad de camino: falta agregar
  `split_data()` a `src/data/load_data.py` (ver punto 1 arriba) y un
  bloque `if __name__ == "__main__":` que imprima el shape y el % de churn
  de train/test para verificar que la estratificación funcionó.
- Después de eso, el orden lógico es: pipeline de preprocessing → script
  de entrenamiento con los 3 modelos → conectar MLflow (tracking URI del
  proyecto en DagsHub, se consigue en la pestaña **Experiments** → **Remote**)
  → 6+ runs → registrar el mejor modelo.
