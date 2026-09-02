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
los umbrales de PSI usados (OK<0.10, WARNING 0.10-0.25, ALERT>=0.25) y por qué (estándar de industria), los 6 tipos de problemas simulados en data quality gates, y la lógica de MANTENER/REVISAR/CONSIDERAR REENTRENAMIENTO con la explicación de Drift ≠ Model Degradation. 
Comandos para correr los 3 scripts.
*python src/monitoring/retraining_decision.py
*python src/monitoring/drift_detection.py
*python src/monitoring/data_quality_gates.py
## 12. Results
_(Pendiente)_ Métricas finales del mejor modelo y comparación entre los
modelos evaluados.

## 13. Team
- Gi — Coordinación y análisis
- Naomy — Modelado / análisis
- Dalay (Maria Angeles) — _(rol a definir)_
