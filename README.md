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

## 6. Data Ingestion
_(Pendiente — Etapa 2)_ Ejecutar el script reproducible de ingesta:
```bash
python src/ingestion/ingest.py
```
Documentar aquí de dónde se obtiene el dataset y cómo se valida al llegar.

## 7. Training
_(Pendiente)_
```bash
python src/training/train.py
```

## 8. MLflow
_(Pendiente)_ Cómo levantar la UI de MLflow y qué se registra en cada run
(parámetros, métricas, artefactos, modelo).
```bash
mlflow ui
```

## 9. Docker
_(Pendiente)_
```bash
docker build -t grupo2-aci94-mlops .
docker run -p 8000:8000 grupo2-aci94-mlops
```

## 10. API
_(Pendiente)_ Endpoint(s) disponibles, ejemplo de request/response de
`POST /predict`.

## 11. Monitoring
El sistema de monitoreo cubre tres frentes: **detección de drift**, **calidad de datos en producción**, y **decisión de reentrenamiento**. El código vive en `src/monitoring/`.

### 11.1 Drift Detection (`src/monitoring/drift_detection.py`)

Se construyó un **reference/baseline** (40% del dataset histórico) y 3 lotes de "producción" simulados a partir del mismo dataset, cada uno con una magnitud de cambio distinta:

| Lote | Descripción | PSI máximo | Estado |
|---|---|---|---|
| Lote 1 (normal) | Sin alterar | 0.0031 | 🟢 OK |
| Lote 2 (moderado) | Cambio leve en edad, horas y educación | 0.1529 | 🟡 WARNING |
| Lote 3 (fuerte) | Cambio grande en la población de entrada | 3.6700 | 🔴 ALERT |

**Métrica usada:** PSI (Population Stability Index), calculado tanto para variables numéricas (age, hours-per-week) como categóricas (education).

**Umbrales (estándar de la industria, documentados en el código, no tratados como ley universal):**
- `PSI < 0.10` → sin cambio significativo → **OK**
- `0.10 ≤ PSI < 0.25` → cambio moderado, vigilar de cerca → **WARNING**
- `PSI ≥ 0.25` → cambio fuerte, la distribución cambió → **ALERT**

Correr:
```bash
python src/monitoring/drift_detection.py
```

### 11.2 Data Quality Gates (`src/monitoring/data_quality_gates.py`)

Se simula contaminación **sobre una copia** de un batch (nunca se modifica `data/raw/` ni `data/processed/`), introduciendo 6 problemas típicos de producción:

1. Missing values
2. Filas duplicadas
3. Outlier extremo (edad = 999)
4. Tipo de dato incorrecto (edad = "treinta")
5. Categoría desconocida (`native-country = "UNKNOWN_NEW_COUNTRY"`)
6. Cambio de esquema (columna nueva no esperada)

El pipeline sigue el flujo: **Detecta → Bloquea/Advierte → Registra**. Los problemas críticos (esquema, tipo de dato, rango) bloquean el batch; los demás generan advertencia. Todo queda registrado en `logs/data_quality_incidents.log`.

Correr:
```bash
python src/monitoring/data_quality_gates.py
```

### 11.3 Decisión de reentrenamiento (`src/monitoring/retraining_decision.py`)

Combina la señal de drift (PSI) con el desempeño del modelo para decidir entre **MANTENER**, **REVISAR** o **CONSIDERAR REENTRENAMIENTO**, partiendo de la premisa de que **Drift ≠ Model Degradation**: un cambio en la distribución de los datos no implica automáticamente que el modelo esté fallando.

| Escenario | Drift | Desempeño | Decisión |
|---|---|---|---|
| Lote 1 | Sin drift | Estable | MANTENER |
| Lote 2 | Drift moderado | Estable | REVISAR |
| Lote 3 | Drift fuerte | Se deterioró | CONSIDERAR REENTRENAMIENTO |

Solo se recomienda reentrenar cuando **ambas** señales (drift relevante + caída de desempeño) aparecen juntas.

Correr:
```bash
python src/monitoring/retraining_decision.py

## 12. Results
_(Pendiente)_ Métricas finales del mejor modelo y comparación entre los
modelos evaluados.

## 13. Team
- Gi — Coordinación y análisis
- Naomy — Modelado / análisis
- Dalay (Maria Angeles) — _(rol a definir)_
