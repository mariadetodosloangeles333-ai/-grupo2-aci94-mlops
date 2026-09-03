# Grupo 2 — ACI94: Predicción de Ingreso (Adult / Census Income)

> Estado: 🚧 en construcción — Etapa 1 (repositorio) completada.

## 1. Business Problem

El problema de negocio es **predecir si el ingreso anual de una persona supera los US$50,000**, a partir de sus características socioeconómicas y laborales (edad, ocupación, nivel educativo, horas trabajadas, estado civil, etc.).

**Pregunta de negocio:** ¿qué características socioeconómicas y laborales permiten identificar patrones asociados con personas cuyos ingresos anuales superan los US$50,000?

**Por qué es relevante:** el modelo es aplicable a análisis socioeconómico, políticas públicas, planificación educativa y de empleo, y estudios de desigualdad. El objetivo no es solo predecir, sino que el modelo sea **útil, interpretable y evaluado con responsabilidad** — no busca establecer causalidad, solo identificar patrones asociados en los datos históricos de 1994.

## 2. Dataset

**Adult / Census Income Dataset** — UCI Machine Learning Repository, extraído por Barry Becker (Censo de EE.UU., 1994).

| | |
|---|---|
| Total de registros | 48,842 (entrenamiento: 32,561 · prueba: 16,281) |
| Variables | 14 predictoras + 1 variable objetivo (`income`) |
| Tipo de problema | Clasificación binaria (`≤50K` vs `>50K`) |
| Balance de clases | ~76.07% `≤50K` vs ~23.93% `>50K` (desbalanceado) |

**Condiciones de extracción originales del dataset:** `(AAGE > 16) && (AGI > 100) && (AFNLWGT > 1) && (HRSWK > 0)` — representa población adulta con ciertas condiciones de participación económica/laboral, no a toda la población.

> **Nota:** el dataset crudo trae 14 variables predictoras. El modelo de producción usa un subconjunto de **9 variables** (`v2_without_sensitive`) — ver sección 7 (Training) y 11 (Monitoring) para el detalle de qué se excluyó y por qué.

Por el desbalance de clases, la evaluación del modelo no puede basarse solo en *accuracy*; se usan métricas como G-Mean, ROC AUC, Recall y Especificidad (ver sección 7).

## 3. Architecture

El flujo end-to-end del proyecto se divide en 3 etapas:

### Pipeline de datos
![Pipeline de datos](docs/architecture/dark_01_pipeline_datos.png)

### Entrenamiento y registro en MLflow
![Entrenamiento y MLflow](docs/architecture/dark_02_entrenamiento_mlflow.png)

### Producción y monitoreo
![Producción y monitoreo](docs/architecture/dark_03_produccion_monitoreo.png)

Cada componente del diagrama corresponde a código real del repositorio: `src/ingestion/`, `src/cleaning/`, `src/features/`, `src/training/`, `src/api/`, y `src/monitoring/`.

## 4. Repository Structure
```
.
├── data/
│   ├── raw/            # datos crudos (no versionados, ver .gitignore)
│   └── processed/       # datos procesados (no versionados)
├── notebooks/            # EDA y notebooks exploratorios
├── src/
│   ├── ingestion/        # script de ingesta reproducible
│   ├── cleaning/         # limpieza y validación de datos
│   ├── features/         # feature engineering reutilizable
│   ├── training/          # entrenamiento y tracking en MLflow
│   ├── api/               # API de inferencia (FastAPI)
│   └── monitoring/        # monitoreo de datos, modelo y sistema
├── tests/                 # pruebas de datos, modelo y API
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Installation
```bash
git clone <URL_DEL_REPO>
cd aci94-repo
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Data Ingestion

Script reproducible en `src/ingestion/ingest.py`. Descarga el dataset directamente desde el UCI ML Repository (paquete `ucimlrepo`, `id=2`) — no depende de un CSV guardado a mano en ninguna computadora.

El script:
1. Descarga features + target desde UCI.
2. Une ambos en un único dataframe crudo (`income` como columna objetivo).
3. Valida el mínimo esperado: al menos 48,000 filas y 15 columnas (14 features + `income`).
4. Guarda el resultado en `data/raw/adult_raw.csv`, con un log de la ingesta en `logs/ingestion.log`.

Correr:
```bash
python src/ingestion/ingest.py
```

## 7. Training

El modelo de producción es un **Random Forest** con `class_weight="balanced"` (para compensar el desbalance de clases del dataset, ~76% ≤50K vs ~24% >50K), entrenado sobre el conjunto de features **`v2_without_sensitive`** (9 variables productivas, ver sección Feature Engineering en `src/features/build_features.py`).

**Métrica principal: G-Mean** (media geométrica de sensibilidad y especificidad), preferida sobre accuracy por el desbalance de clases.

Resultados del modelo aprobado (test set):

| Métrica | Valor |
|---|---|
| G-Mean | 0.8374 |
| ROC AUC | 0.9219 |
| Recall | 0.8660 |
| Especificidad | 0.8098 |

Ver comparación completa entre modelos en la sección 12 (Results).

Correr:
```bash
python src/training/train.py
```

## 8. MLflow

Cada corrida de entrenamiento se registra como un run en MLflow, con parámetros (algoritmo, hiperparámetros, `feature_set_version`, semilla), métricas (G-Mean, ROC AUC, Recall, Especificidad) y artefactos (modelo, matriz de confusión, configuración).

El modelo aprobado quedó registrado en el **Model Registry**:
- Nombre: `adult-income-classifier`
- Versión: `1`
- Alias: `production`
- Run final: `11c8bc44969d40938763b78ead476dfd`

Levantar la UI de MLflow:
```bash
mlflow ui
```

## 9. Docker
```bash
docker build -t grupo2-aci94-mlops .
docker run -p 8000:8000 grupo2-aci94-mlops
```
La imagen ejecuta la API correctamente; el contenedor fue probado de punta a punta (`docker build` + `docker run` + request real a `/predict`).

## 10. API

Servida con **FastAPI**, cargando el modelo desde `models/production/model.pkl` (versión aprobada en el Model Registry). Endpoints disponibles:

- `GET /health` — healthcheck del servicio.
- `GET /model-info` — versión y metadatos del modelo cargado.
- `POST /predict` — predicción sobre las **9 variables productivas**: `age`, `education-num`, `hours-per-week`, `capital-gain`, `capital-loss`, `workclass`, `marital-status`, `occupation`, `relationship`. El campo `age` se valida en el rango **17–90**, el mismo rango usado en la simulación de Monitoring.

Ejemplo de respuesta de `POST /predict`:
```json
{
  "prediction": "<=50K",
  "probability": 0.12,
  "model_version": "1"
}
```

Documentación interactiva disponible en `/docs` una vez el servicio está corriendo.

## 11. Monitoring

El sistema de monitoreo cubre tres frentes: **detección de drift**, **calidad de datos en producción**, y **decisión de reentrenamiento**. El código vive en `src/monitoring/`, y opera exclusivamente sobre las **9 variables productivas** de `v2_without_sensitive` (no sobre las 15 columnas RAW).

### 11.1 Drift Detection (`src/monitoring/drift_detection.py`)

Se construyó un **reference/baseline** (40% del dataset histórico) y 3 lotes de "producción" simulados a partir del mismo dataset, cada uno con una magnitud de cambio distinta:

| Lote | Descripción | PSI máximo | Estado |
|---|---|---|---|
| Lote 1 (normal) | Sin alterar | 0.0031 | 🟢 OK |
| Lote 2 (moderado) | Cambio leve en edad, horas y educación | 0.1529 | 🟡 WARNING |
| Lote 3 (fuerte) | Cambio grande en la población de entrada | 3.6700 | 🔴 ALERT |

**Métrica usada:** PSI (Population Stability Index).
- Numéricas monitoreadas: `age`, `education-num`, `hours-per-week`
- Categórica monitoreada: `occupation` (en lugar de `education`, que no es una entrada productiva)

**Umbrales (estándar de la industria, documentados en el código, no tratados como ley universal):**
- `PSI < 0.10` → sin cambio significativo → **OK**
- `0.10 ≤ PSI < 0.25` → cambio moderado, vigilar de cerca → **WARNING**
- `PSI ≥ 0.25` → cambio fuerte, la distribución cambió → **ALERT**

El rango de `age` en la simulación está alineado con el que valida la API: **17 a 90 años**.

Correr:
```bash
python src/monitoring/drift_detection.py
```

### 11.2 Data Quality Gates (`src/monitoring/data_quality_gates.py`)

Se simula contaminación **sobre una copia** de un batch (nunca se modifica `data/raw/` ni `data/processed/`), introduciendo 6 problemas típicos de producción:

1. Missing values (`occupation`)
2. Filas duplicadas
3. Outlier extremo (`age = 999`)
4. Tipo de dato incorrecto (`age = "treinta"`)
5. Categoría desconocida en `occupation`
6. Cambio de esquema (columna nueva no esperada)

Las validaciones corren sobre las **9 columnas productivas** (no las 15 de RAW), y detectan tanto **columnas faltantes** como **columnas extra** no esperadas en el esquema. La prueba de "categoría desconocida" se hace sobre `occupation` (variable que sí llega a producción), no sobre `native-country`, que quedó excluida en la versión productiva.

El pipeline sigue el flujo: **Detecta → Bloquea/Advierte → Registra**. Los problemas críticos (esquema, tipo de dato, rango) bloquean el batch; los demás generan advertencia. Todo queda registrado en `logs/data_quality_incidents.log`.

Correr:
```bash
python src/monitoring/data_quality_gates.py
```

### 11.3 Decisión de reentrenamiento (`src/monitoring/retraining_decision.py`)

Combina la señal de drift (PSI) con el desempeño del modelo, medido con **G-Mean**, para decidir entre **MANTENER**, **REVISAR** o **CONSIDERAR REENTRENAMIENTO**, partiendo de la premisa de que **Drift ≠ Model Degradation**: un cambio en la distribución de los datos no implica automáticamente que el modelo esté fallando.

El G-Mean aprobado de Production v1 es **0.8374** (`PRODUCTION_G_MEAN`).

| Escenario | Drift | G-Mean (simulado) | Decisión |
|---|---|---|---|
| Lote 1 | Sin drift | Estable | MANTENER |
| Lote 2 | Drift moderado | Estable | REVISAR |
| Lote 3 | Drift fuerte | Se deterioró | CONSIDERAR REENTRENAMIENTO |

> **Nota:** los valores de G-Mean usados para los lotes de ejemplo son **simulados** con fines demostrativos — no provienen de una evaluación real del modelo sobre datos de producción etiquetados. Solo se recomienda reentrenar cuando **ambas** señales (drift relevante + caída de G-Mean) aparecen juntas.

Correr:
```bash
python src/monitoring/retraining_decision.py
```

## 12. Results

### Comparación de modelos ajustados

La selección del modelo se realizó con **validación cruzada** sobre el conjunto de entrenamiento, usando **G-Mean** como métrica principal debido al desbalance de clases. El conjunto de test no se utilizó para seleccionar el modelo.

| Modelo | G-Mean CV | Desv. CV | G-Mean OOF | F1 CV | Recall CV | Especificidad CV | ROC AUC CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8221 | 0.0072 | 0.8221 | 0.6800 | 0.8490 | 0.7960 | 0.9058 |
| Decision Tree | 0.8214 | 0.0049 | 0.8215 | 0.6747 | **0.8661** | 0.7792 | 0.9043 |
| **Random Forest** | **0.8338** | 0.0058 | **0.8338** | **0.6990** | 0.8529 | **0.8151** | **0.9186** |

**Random Forest** fue seleccionado porque obtuvo el mayor G-Mean promedio de validación y OOF, junto con el mejor F1, especificidad y ROC AUC. Aunque Decision Tree presentó un recall ligeramente mayor, su especificidad y equilibrio general entre clases fueron inferiores.

### Evaluación final del modelo seleccionado

Después de seleccionar Random Forest mediante validación cruzada, el modelo fue evaluado **una única vez** sobre el conjunto de test reservado:

| Métrica | Resultado |
|---|---:|
| G-Mean | 0.8374 |
| F1 | 0.7011 |
| Recall | 0.8660 |
| Especificidad | 0.8098 |
| ROC AUC | 0.9219 |
| Diferencia absoluta validación-test | 0.0036 |

La diferencia reducida entre validación y test (0.0036) indica que el desempeño se mantuvo sobre datos no utilizados en la selección del modelo, sin evidencia importante de sobreajuste.

MLflow almacenó los criterios de selección y validación, la matriz de confusión, la curva ROC, la importancia de variables y el resultado de la validación final, permitiendo reproducir y auditar la evaluación completa.

El modelo fue registrado en MLflow Model Registry con el nombre `adult-income-classifier`; la versión `1` recibió el alias `production`, identificándola como la versión aprobada para el servicio de inferencia (ver sección 8, MLflow).

## 13. Team
- Gi — Coordinación y análisis
- Naomy — Modelado / análisis
- Dalay (Maria Angeles) — _(rol a definir)_
