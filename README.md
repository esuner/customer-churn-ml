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
├── app/              # API FastAPI (/health, /predict)
├── data/             # Dataset (DVC, no directamente en Git) + ejemplos de payload en data/examples
├── models/           # Modelo exportado del Registry (churn-classifier/), versionado en Git (~5 KB)
├── notebooks/        # Exploración (EDA); no es el flujo productivo
├── scripts/          # Entry points: export_model.py, register_model.py, ...
├── src/
│   ├── data/         # Carga y partición de datos
│   ├── features/     # Preprocessing (Pipeline / ColumnTransformer)
│   ├── training/     # Entrenamiento de modelos + registro en MLflow
│   ├── evaluation/   # Métricas y comparación de modelos
│   └── inference/    # Predictor, niveles de riesgo y contratos Pydantic
├── tests/            # pytest (API, predictor y niveles de riesgo)
├── Dockerfile, compose.yaml, .dockerignore
└── requirements.txt (desarrollo) · requirements-api.txt (imagen Docker)
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
3. El remote ya viene configurado en el repo (`.dvc/config`); solo faltan tus
   credenciales. Con el `.venv` activado, corré:
   ```bash
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <tu usuario de DagsHub>
   dvc remote modify origin --local password <tu token>
   ```
   Se guardan en `.dvc/config.local`, que está gitignoreado — nunca se
   comparte ni se sube. El mismo token sirve para MLflow (ver abajo).
4. Traé los datos reales:
   ```bash
   dvc pull
   ```


## Reproducir el entrenamiento

Con el entorno instalado y los datos recuperados (secciones anteriores):

1. **Credenciales de MLflow.** Creá un archivo `.env` en la raíz del proyecto
   (está en `.gitignore`, nunca se sube) con tu usuario y token de DagsHub:
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/esuner/customer-churn-ml.mlflow
   MLFLOW_TRACKING_USERNAME=<tu usuario de DagsHub>
   MLFLOW_TRACKING_PASSWORD=<tu token de DagsHub>
   ```
2. **Entrenar y comparar los 7 modelos** (tarda ~20 minutos: cada run hace
   validación cruzada y sube el modelo a DagsHub):
   ```bash
   python -m src.training.train
   ```
   Cada run queda en el experimento `churn-model-comparison` de MLflow, con
   parámetros, métricas, modelo y tags de trazabilidad (`git_commit`,
   `git_dirty`, `data_version`).
3. **Registrar el modelo candidato** en el Model Registry (correr una sola
   vez; repetirlo crea una versión nueva):
   ```bash
   python scripts/register_model.py
   ```

Semillas: `RANDOM_STATE = 42` en la partición, en los modelos y en la
validación cruzada, así que los resultados son reproducibles.

## Experimentos y selección del modelo

Se compararon 7 configuraciones sobre el mismo pipeline de preprocessing
(46 features tras imputar, escalar y aplicar one-hot). La selección se hizo con
**validación cruzada de 5 folds sobre train** (`cv_*`); el conjunto de test se
usó solo para reportar. Ordenados por `cv_f1`:

| Run | cv_f1 | cv_recall | cv_roc_auc | test_recall | test_precision | Falsos negativos (de 372) |
|---|---|---|---|---|---|---|
| **`logreg_C1_balanced`** | **0.610** | 0.768 | **0.828** | 0.718 | 0.478 | 105 |
| `logreg_C0.1_balanced` | 0.606 | 0.764 | 0.828 | 0.723 | 0.481 | 103 |
| `rf_200_depth8_balanced` | 0.603 | 0.710 | 0.817 | 0.677 | 0.500 | 120 |
| `rf_300_depth5_balanced` | 0.592 | 0.771 | 0.814 | 0.742 | 0.475 | 96 |
| `logreg_C1` | 0.561 | 0.484 | 0.828 | 0.446 | 0.659 | 206 |
| `rf_200` | 0.522 | 0.438 | 0.805 | 0.406 | 0.614 | 221 |
| `baseline_dummy` | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 372 |

Qué se aprendió de la comparación:

- **`class_weight="balanced"` fue la mejora más grande**: en regresión
  logística el recall pasa de 0.45 a ~0.72 sin perder ROC-AUC. Limitar la
  profundidad del Random Forest también ayudó (recall de 0.41 a 0.68-0.74).
- **El baseline** acierta el 73.6% de los casos sin detectar a ningún cliente
  que se va: por eso Accuracy no se usa como métrica principal.
- **Random Forest no superó a la regresión logística** en F1 ni en ROC-AUC.

### Modelo candidato: `logreg_C1_balanced`

Métricas en test (1.409 clientes): recall 0.718, precision 0.478, F1 0.574,
ROC-AUC 0.812. Matriz de confusión: 746 verdaderos negativos, 291 falsos
positivos, 105 falsos negativos, 267 verdaderos positivos.

Por qué este y no otro:

- Mejor `cv_f1` (0.610) y mejor `cv_roc_auc` (0.828, empatado). El test
  confirma el mismo orden.
- Es simple, interpretable y liviano para servir en la API. Los tres
  primeros modelos están dentro del ruido de la validación cruzada
  (0.610 / 0.606 / 0.603), así que se desempató por simplicidad.
- Alternativa: `rf_300_depth5_balanced` tiene el mejor recall (0.742, 96
  falsos negativos) pero peor F1 y ROC-AUC; sería la opción si el negocio
  priorizara todavía más no perder clientes.

### Impacto del falso negativo y trade-off

Un falso negativo es un cliente que iba a abandonar y el modelo lo clasificó
como estable: la empresa no lo contacta y lo pierde sin oportunidad de
retención. Un falso positivo es contactar a un cliente que no se iba a ir:
cuesta una oferta o una llamada innecesaria. Bajo el supuesto (razonable en
telecomunicaciones, a validar con el negocio) de que perder un cliente cuesta
bastante más que una campaña de retención, conviene **priorizar el recall**
sobre la precisión. Por eso se usa `class_weight="balanced"`, que sube el
recall de 0.45 a 0.72 a cambio de una precisión de ~0.48 (aproximadamente 1 de
cada 2 alertas es una falsa alarma). El umbral de decisión se mantiene en 0.5;
ajustarlo según el costo real de cada error es una mejora futura.

## Trazabilidad

Desde el modelo registrado se puede reconstruir su origen completo:

| Eslabón | Valor |
|---|---|
| Model Registry | `churn-classifier`, versión 1, alias `candidate` |
| Run de MLflow | `21ae5258b6b1498d80ce6f46ac47992d` (`logreg_C1_balanced`) |
| Commit de código | `fcf2ef1c8e6b45eaa9d93477d92eabbcfa4809bd` |
| Versión de datos (hash DVC del histórico) | `178571792c566f067e3d16b5fb913026` |

El commit y el hash de datos están guardados como tags tanto en el run como en
la versión del modelo. Todos los runs se ejecutaron con el árbol de trabajo
limpio (`git_dirty=False`). Para recuperar exactamente ese código y esos datos:
`git checkout fcf2ef1` y `dvc pull`.

## Servicio de inferencia (Entrega 2)

La API expone el modelo registrado (`churn-classifier`, alias `candidate`) como
servicio REST con FastAPI. Las transformaciones (imputación, escalado, one-hot)
viajan dentro del pipeline guardado, así que la API no repite ninguna: aplica
exactamente las mismas del entrenamiento.

### Levantar el servicio con Docker Compose

Solo hace falta Docker. No se necesita Python, ni credenciales de DagsHub.

```bash
git clone https://github.com/esuner/customer-churn-ml.git
cd customer-churn-ml
docker compose up --build
```

- API en `http://localhost:8000` y documentación Swagger en `http://localhost:8000/docs`.
- En segundo plano: `docker compose up -d --build`. Estado: `docker compose ps`.
  Logs: `docker compose logs`. Detener: `docker compose down`.

### Probar la API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/examples/predict_request_valid.json
```

En PowerShell:

```powershell
$body = Get-Content data\examples\predict_request_valid.json -Raw
Invoke-RestMethod -Method Post -ContentType "application/json" -Body $body http://localhost:8000/predict
```

Respuesta esperada:

```json
{"prediction": "CHURN", "probability": 0.9144, "risk_level": "HIGH", "model_version": "1"}
```

| Endpoint | Qué hace | Respuestas |
|---|---|---|
| `GET /health` | Indica que el servicio está operativo y qué versión del modelo cargó | `200` `{"status": "ok", "model_version": "1"}` |
| `POST /predict` | Recibe las características de un cliente y devuelve la predicción | `200` correcta · `422` entrada inválida · `500` error interno (mensaje genérico; el detalle queda solo en los logs) |

**Contrato de entrada** (validado con Pydantic; el esquema completo está en Swagger):

- Son las 19 características del dataset, sin `customerID` (no es un predictor).
- Las variables categóricas aceptan solo los valores que existen en los datos
  (por ejemplo `Contract`: `Month-to-month`, `One year` o `Two year`); `SeniorCitizen` es 0 o 1.
- `tenure` es un entero entre 0 y 72, y `MonthlyCharges` debe ser mayor que 0.
- `TotalCharges` es opcional: en clientes nuevos puede omitirse y se imputa igual que en el entrenamiento.
- Los campos desconocidos o faltantes se rechazan con `422`. El ejemplo
  `data/examples/predict_request_invalid.json` devuelve un `422` que señala
  `tenure`, `Contract` y `MonthlyCharges`.

**Respuesta:** `prediction` (`CHURN` si la probabilidad es ≥ 0.5, el mismo corte
con el que se evaluó el modelo), `probability`, `risk_level` y `model_version`.

### Niveles de riesgo

Los cortes se eligieron con datos: se calcularon probabilidades por validación
cruzada sobre **train** (sin tocar el test) y se midió cuántos clientes de cada
rango abandonaron realmente.

| Nivel | Probabilidad | % de clientes | Churn real observado |
|---|---|---|---|
| `LOW` | < 0.30 | 41 % | 5.8 % |
| `MEDIUM` | 0.30 a 0.60 | 28 % | ~24 % |
| `HIGH` | ≥ 0.60 | 31 % | 56 % |

`LOW` tiene unas 4 veces menos churn que el promedio (26 %) y concentra solo el
9 % de los clientes que se van; `HIGH` tiene más del doble del promedio y
captura aproximadamente 2 de cada 3 clientes que abandonan. Con
`class_weight="balanced"` la probabilidad **no está calibrada** (un cliente con
0.55 abandona en la práctica ~31 % de las veces), por eso los cortes se fijaron
mirando el churn observado y no el valor crudo. Las constantes están en
`src/inference/risk.py`.

### Cómo llega el modelo a la imagen

1. `scripts/export_model.py` descarga del Model Registry la versión con alias
   `candidate` y la guarda en `models/churn-classifier/`, junto con
   `model_info.json` (versión, run de MLflow, commit y hash de los datos).
2. Esa carpeta pesa ~5 KB y está versionada en Git, así que construir la imagen
   no requiere credenciales ni acceso a DagsHub.
3. La API carga el pipeline con `pickle` al arrancar (sin MLflow) y responde
   `model_version` con el valor de `model_info.json`. De ahí se llega al run, al
   commit y a los datos (ver "Trazabilidad").

Para publicar un modelo nuevo: registrarlo, mover el alias `candidate`, correr
`python scripts/export_model.py` y reconstruir la imagen.

### Configuración y seguridad

- Variables de entorno: `LOG_LEVEL` (por defecto `INFO`) y `MODEL_DIR` (por
  defecto `models/churn-classifier`); `compose.yaml` las define.
- La imagen parte de `python:3.11-slim`, instala solo `requirements-api.txt`
  (sin Jupyter, MLflow ni DVC) y pesa ~700 MB. Corre con un usuario sin
  privilegios y tiene un `HEALTHCHECK` que consulta `/health`.
- `.dockerignore` deja afuera `.env`, la configuración de DVC (donde está el
  token), los datos, los tests y `.git`: no hay secretos en la imagen.
- Cada predicción se registra en el log (resultado, probabilidad, nivel de
  riesgo y versión del modelo), sin incluir los datos del cliente.

### Tests

```bash
pip install -r requirements.txt
python -m pytest
```

Son 28 tests que corren en un par de segundos y no necesitan DVC ni credenciales
(usan el modelo exportado y los ejemplos que están en Git):

- Carga del modelo, con sus metadatos de trazabilidad.
- Predicción válida: estructura y tipos de la respuesta, determinismo, y un
  cliente propenso a irse que debe puntuar más alto que uno fiel.
- `/health` y `/predict`, incluida la coincidencia de `model_version` entre ambos.
- Entradas inválidas: el ejemplo de la cátedra, valores fuera de contrato, campo
  faltante, campo desconocido y cuerpo vacío (todos `422`).
- Error interno simulado: responde `500` sin filtrar detalles.
- Documentación Swagger accesible, y los cortes de `risk_level` en sus bordes.

### Evidencia de punta a punta

```bash
git clone https://github.com/esuner/customer-churn-ml.git
cd customer-churn-ml
pip install -r requirements.txt
python -m pytest
docker build -t churn-api:entrega-2 .
docker compose up -d
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/examples/predict_request_valid.json
```

## Uso de asistentes de desarrollo

Parte del código de este repositorio se escribió con la asistencia de un LLM
(Claude, Anthropic), lo cual queda declarado en el historial de Git mediante
la línea `Co-Authored-By` en los mensajes de commit — por eso puede aparecer
listado en el panel "Contributors" de GitHub. Esto está permitido
explícitamente por la consigna del curso:

> "Se permite utilizar documentación, bibliotecas y asistentes de desarrollo
> siempre que el equipo mantenga comprensión real de la solución, respete las
> políticas institucionales y pueda defender técnicamente lo implementado."

El equipo puede explicar y defender cada decisión técnica del proyecto: por
qué se particiona con `stratify`, por qué el pipeline usa `ColumnTransformer`,
por qué se comparan 7 configuraciones con validación cruzada, por qué se
prioriza el recall vía `class_weight="balanced"` dado el costo de un falso
negativo, y por qué se eligió `logreg_C1_balanced` como candidato (ver
"Experimentos y selección del modelo" arriba). El asistente se usó para
explicar conceptos y guiar la escritura del código; el diseño, las decisiones
de negocio y la verificación de cada resultado fueron del equipo.

## Estado del proyecto (al 01/11/2026)

### Hecho
- [x] Dataset de la cátedra verificado (checksums) y organizado en `data/`.
- [x] EDA — [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb): ~26% churn,
      los 26 faltantes de `TotalCharges` son clientes con `tenure==0` (se
      imputan con 0), `customerID` excluido como predictor.
- [x] Repositorio en GitHub con la estructura recomendada.
- [x] Dataset versionado con **DVC**, remote en **DagsHub**.
- [x] Partición train/test reproducible (`src/data/load_data.py`, estratificada,
      `random_state=42`): 5.634 train / 1.409 test.
- [x] Pipeline de preprocessing con `ColumnTransformer`
      (`src/features/preprocessing.py`).
- [x] Entrenamiento por script (`python -m src.training.train`) con baseline,
      regresión logística y Random Forest; métricas en `src/evaluation/`.
- [x] 7 runs registrados en MLflow/DagsHub con parámetros, métricas, modelo y
      tags de trazabilidad.
- [x] Modelo candidato registrado en el Model Registry.
- [x] Prueba de reproducción por un segundo integrante: Thomas Palacio clonó
      el repo, ejecutó `dvc pull` y corrió `python -m src.training.train` de
      forma independiente, confirmando resultados equivalentes.
- [x] Tag Git `entrega-1` sobre el commit final.
- [x] Entrega 1 cerrada.
- [x] Modelo exportado del Registry a `models/churn-classifier/` con su
      metadato de trazabilidad.
- [x] Capa de inferencia (`src/inference/`) y API FastAPI con `/health` y
      `/predict`, contratos Pydantic y Swagger.
- [x] Niveles de riesgo `LOW` / `MEDIUM` / `HIGH` con cortes justificados con datos.
- [x] 28 tests con pytest.
- [x] `Dockerfile`, `.dockerignore` y `compose.yaml`: imagen construida y
      servicio levantado con `docker compose up`, con `/predict` respondiendo.

### Pendiente para Entrega 2 (03/11/2026 — 19:00 h)
- [ ] Prueba de reproducción por un segundo integrante (clonar, `pytest`,
      `docker build`, `docker compose up`, `POST /predict`).
- [ ] Tag Git `entrega-2` sobre el commit final.

### Limitaciones conocidas
- La comparación usa una única partición train/test; con más datos o tiempo se
  reportarían intervalos de confianza.
- El corte de decisión (0.5) no está optimizado contra costos reales de cada error.
- La probabilidad devuelta no está calibrada (efecto de `class_weight="balanced"`);
  por eso los niveles de riesgo se definieron con el churn observado.
- La API no tiene autenticación, `request_id` ni logging estructurado, y corre con
  un solo proceso: queda para la entrega final junto con CI/CD, despliegue y monitoreo.
