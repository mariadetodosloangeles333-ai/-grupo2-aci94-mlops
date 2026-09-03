# Grupo 2 — ACI94: Predicción de Ingreso (Adult / Census Income)

> Estado: 🚧 en construcción — Etapa 1 (repositorio) completada.

## 1. Business Problem
_(Pendiente)_ Describir el problema de negocio: predecir si el ingreso anual de una
persona supera los US$50,000 a partir de sus características socioeconómicas y
laborales, y por qué esto es relevante (segmentación, políticas públicas,
estudios de desigualdad, etc.).

## 2. Dataset
_(Pendiente)_ Adult / Census Income Dataset (UCI Machine Learning Repository,
Barry Becker, censo de EE.UU. de 1994). 48,842 registros, 14 variables
predictoras + 1 variable objetivo (`income`). Enlace a la fuente original.

## 3. Architecture
_(Pendiente)_ Diagrama de arquitectura MLOps end-to-end: Fuente de datos → Data
Ingestion → Raw/Bronze → Data Validation → Data Cleaning → Feature Pipeline →
Training → Evaluation → MLflow (Tracking + Model Registry) → Best Candidate →
Dockerize → Model API → Producción → Monitoring (Data Drift / Model
Performance / System Metrics) → Retrain Trigger.

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
## 6. Feature Engineering

El proyecto soporta dos versiones del conjunto de features, controladas por `feature_set_version` en `src/features/build_features.py`:

- **`v1_baseline`**: usa todas las variables aprobadas.
- **`v2_without_sensitive`** *(versión productiva, usada por la API y por Monitoring)*: excluye las variables sensibles `sex`, `race` y `native-country`.

### Las 9 variables productivas (v2)

| Tipo | Variables |
|---|---|
| Numéricas | `age`, `education-num`, `hours-per-week` |
| Numéricas sesgadas | `capital-gain`, `capital-loss` |
| Categóricas | `workclass`, `marital-status`, `occupation`, `relationship` |

### Variables excluidas y justificación

- **`education`**: redundante con `education-num`, que ya representa el nivel educativo de forma numérica.
- **`fnlwgt`**: es un peso estadístico del muestreo censal, no una característica individual — no aporta señal predictiva real.
- **`race`, `sex`, `native-country`**: variables sensibles. Se excluyen de la versión productiva (`v2_without_sensitive`) para evitar que el modelo las use como predictoras directas; se conservan disponibles para análisis de equidad (`SENSITIVE_FEATURES`).

Transformaciones aplicadas: `capital-gain`/`capital-loss` se convierten en `*_flag` (indicador de presencia) + `*_log` (log1p); `native-country` agrupa países poco frecuentes en `"Other"` (solo aplica en v1). El mismo pipeline (`construir_pipeline_features`) se usa en entrenamiento y en la API — no existe una versión distinta para producción.

Correr:
```bash
python src/features/build_features.py
```

---

## 11. Monitoring

Actualizado tras la auditoría de código para alinearse con la versión productiva del modelo (`v2_without_sensitive`, 9 variables).

### 11.1 Drift Detection

El drift se mide sobre variables que **sí forman parte de las 9 entradas productivas** (antes se monitoreaba `education`, que ya no es un input del modelo; se sustituyó por `occupation`).

- Numéricas monitoreadas: `age`, `hours-per-week`
- Categórica monitoreada: `occupation` (en lugar de `education`)
- Métrica: PSI (Population Stability Index)
- Umbrales: OK < 0.10 · WARNING 0.10–0.25 · ALERT ≥ 0.25

El rango de `age` en la simulación de lotes está alineado con el que valida la API: **17 a 90 años**.

### 11.2 Data Quality Gates

Las validaciones corren sobre las **9 entradas productivas**, no sobre las 15 columnas del dataset RAW. Se detectan tanto **columnas faltantes** como **columnas extra** no esperadas en el esquema.

La prueba de "categoría desconocida" se hace sobre `occupation` y `workclass` (variables que sí llegan a producción), no sobre `native-country`, que quedó excluida en la versión productiva.

Flujo: **Detecta → Bloquea/Advierte → Registra**, sobre una copia del batch; los datos originales nunca se modifican. Log en `logs/data_quality_incidents.log`.

### 11.3 Decisión de reentrenamiento

La métrica principal para evaluar desempeño es **G-Mean** (coherente con la métrica usada en el entrenamiento del modelo, ver sección 5). La lógica de decisión (MANTENER / REVISAR / CONSIDERAR REENTRENAMIENTO) no cambió: sigue exigiendo que drift relevante **y** deterioro de G-Mean aparezcan juntos para recomendar reentrenar.

> **Nota:** los valores de desempeño (G-Mean) usados para los lotes de ejemplo en `retraining_decision.py` son **simulados** con fines demostrativos — no provienen de una evaluación real del modelo sobre datos de producción etiquetados.

Correr:
```bash
python src/monitoring/drift_detection.py
python src/monitoring/data_quality_gates.py
python src/monitoring/retraining_decision.py
```
