# Guía de uso — Repositorio ACI94 (Grupo 2)

Esta guía explica cómo clonar, configurar y trabajar en el repositorio del
proyecto de clasificación de ingreso (Adult / Census Income), para que
cualquiera del equipo pueda arrancar sin depender de que alguien más le
explique todo de nuevo.

---

## 1. Requisitos antes de empezar

Antes de tocar el repo, cada persona necesita tener instalado:

- **Git** (para GitHub Desktop) o **GitHub Desktop** directamente — [desktop.github.com](https://desktop.github.com)
- **Python 3.10+** — descargado de [python.org/downloads](https://www.python.org/downloads/),
  marcando la opción **"Add python.exe to PATH"** durante la instalación.
  (En Windows, si `python --version` abre la Microsoft Store en vez de mostrar
  un número de versión, significa que no está realmente instalado — hay que
  bajarlo de python.org, no usar el acceso directo que trae Windows.)
- Opcional pero recomendado: **Visual Studio Code**, con la extensión de
  Python instalada.

Para confirmar que todo quedó bien, abrí una terminal y corré:

```bash
git --version
python --version
python -m pip --version
```

Los tres deberían mostrar un número de versión, sin errores.

---

## 2. Clonar el repositorio

### Con GitHub Desktop (sin usar comandos)
1. Abrí GitHub Desktop → `File` → `Clone repository`.
2. Buscá `grupo2-aci94-mlops` en la lista (tiene que estar dentro de tu cuenta
   de GitHub, o el dueño del repo te tuvo que agregar como colaborador).
3. Elegí dónde guardarlo en tu compu y dale `Clone`.

### Con terminal
```bash
git clone https://github.com/<usuario>/grupo2-aci94-mlops.git
cd grupo2-aci94-mlops
```

---

## 3. Estructura del repositorio

```
.
├── data/
│   ├── raw/            # datos crudos (NO se suben a git, ver .gitignore)
│   └── processed/       # datos procesados (tampoco se suben)
├── notebooks/            # EDA y notebooks exploratorios
├── src/
│   ├── ingestion/        # script de ingesta reproducible (ingest.py)
│   ├── cleaning/         # limpieza y validación de datos
│   ├── features/         # feature engineering reutilizable
│   ├── training/          # entrenamiento y tracking en MLflow
│   ├── api/               # API de inferencia (FastAPI)
│   └── monitoring/        # monitoreo de datos, modelo y sistema
├── tests/                 # pruebas de datos, modelo y API
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Obtener el dataset (ingesta)

El dataset **no está guardado en el repositorio** — se descarga con un script
reproducible, tal como pide el proyecto. Nadie debería tener que pedir el
CSV por WhatsApp: cualquiera puede generarlo así:

```bash
python -m pip install -r requirements.txt
python src/ingestion/ingest.py
```

Esto descarga el Adult Dataset directo del UCI Machine Learning Repository
y lo guarda en `data/raw/adult_raw.csv` (48,842 filas, 15 columnas). El
script deja un log en `logs/ingestion.log`.

> Si el script falla por problemas de instalación de Python, como solución
> temporal se puede compartir manualmente el `adult_raw.csv` ya generado —
> pero la meta es que cada quien pueda correr el script por su cuenta.

---

## 5. Flujo de trabajo con ramas (Git Workflow)

- `main` → rama estable, solo se actualiza con código ya revisado.
- `develop` → rama base de trabajo del equipo.
- `feature/<algo>` → cada tarea específica sale de `develop`.

Ejemplos de nombres de rama:
```
feature/data-ingestion
feature/data-cleaning
feature/model
feature/api
feature/monitoring
```

Pasos típicos para trabajar en algo nuevo:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/mi-tarea
# ... hacer cambios ...
git add .
git commit -m "feat: descripción corta de lo que hice"
git push origin feature/mi-tarea
```

Después se abre un Pull Request hacia `develop` para que el resto del
equipo revise antes de mezclar.

---

## 6. Convención de mensajes de commit

Usar prefijos descriptivos, en minúscula, explicando qué se hizo:

```
feat: add data validation pipeline
fix: handle missing CustomerID
model: register random forest in MLflow
monitor: add PSI drift detection
chore: initial repo structure and gitignore
```

Evitar mensajes como `cambio final`, `prueba`, `ahora_si_final` — el
historial de commits es parte de lo que se evalúa.

---

## 7. Qué NO se sube al repositorio

El `.gitignore` ya bloquea esto automáticamente, pero es bueno saberlo:

- Cualquier CSV/dataset dentro de `data/raw/` o `data/processed/`
- Entornos virtuales (`.venv/`, `venv/`)
- Archivos de caché de Python (`__pycache__/`, `.ipynb_checkpoints/`)
- Modelos entrenados pesados (`.pkl`, `.joblib`, `.h5`)
- Artefactos de MLflow (`mlruns/`, `mlartifacts/`)

Si `git status` muestra alguno de estos como "nuevo archivo" antes de un
commit, revisar el `.gitignore` antes de subir.

---

## 8. Dudas o problemas comunes

| Problema | Causa probable | Solución |
|---|---|---|
| `pip` no se reconoce | Python no está en el PATH | Reinstalar Python marcando "Add to PATH" |
| `python` abre la Microsoft Store | Alias falso de Windows | Desactivar en "Alias de ejecución de aplicaciones" |
| El CSV aparece en los cambios de Git | `.gitignore` no lo está bloqueando | Confirmar la ruta exacta del archivo dentro de `data/raw/` |
| Conflictos al hacer push | Alguien más subió cambios antes | `git pull origin develop` antes de seguir trabajando |
