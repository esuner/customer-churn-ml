"""Builds notebooks/01_eda.ipynb programmatically so the EDA notebook
lives under version control as reviewable source, not just binary output."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# EDA — Customer Churn

Exploración del dataset histórico provisto por la cátedra
(`data/raw/customer_churn_historical.csv`, 7.043 clientes, variable objetivo `Churn`).

Objetivo: entender dimensiones, tipos, faltantes, distribución del target y
problemas de calidad antes de diseñar el pipeline de preprocessing.""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

df = pd.read_csv("../data/raw/customer_churn_historical.csv")
df.shape""")

md("## 1. Dimensiones y tipos")

code("""df.info()""")

code("""df.head()""")

md("""Observación: todas las columnas se leen como `object` salvo `SeniorCitizen`,
`tenure` y `MonthlyCharges`. `TotalCharges` debería ser numérica — revisamos por qué
no lo es en la sección de calidad de datos.""")

md("## 2. Valores faltantes")

code("""missing = df.isna().sum()
missing[missing > 0]""")

code("""# TotalCharges se lee como object porque los faltantes vienen como strings vacíos/espacios,
# no como NaN reales. Los convertimos y volvemos a contar.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].isna().sum()""")

code("""# ¿Quiénes son los clientes con TotalCharges faltante?
df[df["TotalCharges"].isna()][["customerID", "tenure", "MonthlyCharges", "Contract", "Churn"]]""")

md("""**Hallazgo:** los 26 faltantes de `TotalCharges` corresponden a clientes con
`tenure == 0` (clientes nuevos, todavía sin facturación acumulada). No es un dato
corrupto sino la ausencia lógica de historial — el pipeline debe imputarlo
(ej. con 0, o con `MonthlyCharges` como aproximación) en lugar de descartarlos.""")

md("## 3. Distribución del target (`Churn`)")

code("""churn_counts = df["Churn"].value_counts()
churn_rate = df["Churn"].value_counts(normalize=True)
print(churn_counts)
print(churn_rate)

fig, ax = plt.subplots(figsize=(4, 4))
churn_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_title("Distribución de Churn")
ax.set_xlabel("")
ax.set_ylabel("Clientes")
plt.tight_layout()
plt.savefig("figures/target_distribution.png", dpi=120)
plt.show()""")

md("""**Hallazgo:** dataset moderadamente desbalanceado (~26% Churn=Yes / ~74% No).
No es extremo, pero conviene reportar Precision/Recall/F1/ROC-AUC y matriz de
confusión en vez de depender de Accuracy — una métrica dominada por la clase
mayoritaria daría una imagen artificialmente buena.""")

md("## 4. Variables numéricas")

code("""numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[numeric_cols].describe()""")

code("""fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(data=df, x=col, hue="Churn", bins=30, ax=ax, element="step", stat="density", common_norm=False)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("figures/numeric_distributions.png", dpi=120)
plt.show()""")

md("""**Hallazgo:** `tenure` bajo se asocia claramente con mayor churn (clientes nuevos
abandonan más); `MonthlyCharges` alto también se asocia con más churn. Son señales
fuertes para el modelo.""")

md("## 5. Variables categóricas")

code("""categorical_cols = [c for c in df.columns if c not in numeric_cols + ["customerID", "Churn", "SeniorCitizen"]]
for col in categorical_cols:
    print(col, "->", df[col].unique())""")

code("""churn_by_contract = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack()
churn_by_contract""")

code("""fig, ax = plt.subplots(figsize=(6, 4))
churn_by_contract["Yes"].sort_values().plot(kind="barh", ax=ax, color="#C44E52")
ax.set_xlabel("Tasa de churn")
ax.set_title("Tasa de churn por tipo de contrato")
plt.tight_layout()
plt.savefig("figures/churn_by_contract.png", dpi=120)
plt.show()""")

md("""**Hallazgo:** el contrato `Month-to-month` concentra la mayor tasa de churn frente
a `One year`/`Two year`. Variables como `Contract`, `InternetService` y los servicios
"No internet service"/"No phone service" son categorías legítimas (no faltantes) que
el `OneHotEncoder` debe preservar tal cual.""")

md("## 6. Problemas de calidad detectados")

code("""dupes = df["customerID"].duplicated().sum()
print("customerID duplicados:", dupes)
print("SeniorCitizen dtype:", df["SeniorCitizen"].dtype, "-> valores:", df["SeniorCitizen"].unique())""")

md("""- **`customerID`**: único por fila (0 duplicados), identificador — se excluye
  explícitamente como predictor.
- **`TotalCharges`**: 26 faltantes que en realidad son clientes con `tenure == 0`;
  requiere imputación explícita en el `Pipeline` (no eliminar filas).
- **`SeniorCitizen`**: ya viene codificada como 0/1 (no como Yes/No como el resto de
  las binarias) — hay que tratarla de forma consistente en el `ColumnTransformer`.
- **Categorías "No internet service" / "No phone service"**: no son faltantes, son
  información válida (el cliente no tiene ese servicio contratado).

## 7. Conclusiones para el pipeline

- Target: `Churn` (Yes/No) → binarizar a 1/0.
- Excluir `customerID` de las features.
- Imputar `TotalCharges` (missing ⇒ clientes nuevos, `tenure == 0`).
- `ColumnTransformer`: `OneHotEncoder` para categóricas (incluyendo `SeniorCitizen`
  tratada como categórica u ordinal binaria), `StandardScaler` para numéricas
  (`tenure`, `MonthlyCharges`, `TotalCharges`) en el modelo lineal.
- Accuracy no alcanza como métrica única por el desbalance ~26/74 — priorizar
  Recall/F1/ROC-AUC dado que un falso negativo (cliente que iba a irse y no se
  detectó) tiene mayor costo de negocio que un falso positivo.""")

nb["cells"] = cells
with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/01_eda.ipynb")
