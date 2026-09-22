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

## Estado del proyecto (al 22/09/2026)

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

### Pendiente para Entrega 1
_Ninguno — entrega cerrada._

### Limitaciones conocidas
- La comparación usa una única partición train/test; con más datos o tiempo se
  reportarían intervalos de confianza.
- El umbral de decisión (0.5) no está optimizado contra costos reales.
- Aún no hay tests automáticos ni API (Entrega 2).
- MLflow advierte que `SeniorCitizen` se infiere como entero: al definir el
  contrato de la API habrá que decidir cómo tratar valores faltantes.
