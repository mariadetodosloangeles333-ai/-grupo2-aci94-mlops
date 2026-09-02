# Cómo obtener el dataset (Adult / Census Income) — ACI94

Este documento es solo para levantar el dataset en tu compu. Para todo lo
demás del repositorio (ramas, commits, estructura), ver `GUIA_DE_USO.md`.

---

## Antes de empezar: ¿tenés Python instalado?

Abrí una terminal y corré:

```bash
python --version
```

- Si te muestra un número (ej. `Python 3.12.0`) → seguí al siguiente paso.
- Si te dice que el comando no existe, o se abre la Microsoft Store → todavía
  no tenés Python real instalado. Bajalo de
  [python.org/downloads](https://www.python.org/downloads/) y **marcá la
  casilla "Add python.exe to PATH"** durante la instalación. Sin eso, nada
  de lo siguiente va a funcionar.

---

## Paso a paso

**1. Ubicate en la carpeta del repositorio** (la que clonaste con GitHub
Desktop o `git clone`). Abrí una terminal ahí dentro.

**2. Instalá las dependencias:**

```bash
python -m pip install -r requirements.txt
```

(Si `pip` solo, sin `python -m` adelante, te da error de "no se reconoce",
usá siempre la versión con `python -m pip`.)

**3. Corré el script de ingesta:**

```bash
python src/ingestion/ingest.py
```

Vas a ver mensajes en pantalla mostrando el progreso de la descarga. Al
final debería decir algo como "Ingesta finalizada con éxito".

**4. Verificá el resultado:** entrá a la carpeta `data/raw/` dentro del
repo — debería haber un archivo nuevo llamado `adult_raw.csv`, con
48,842 filas y 15 columnas.

---

## Qué hace el script (por si preguntan)

- Descarga el Adult Dataset directo desde el repositorio oficial de UCI
  (usando el paquete `ucimlrepo`), no depende de que alguien tenga el CSV
  guardado a mano.
- Junta la variable objetivo (`income`) con las 14 variables predictoras.
- Guarda todo en `data/raw/adult_raw.csv`.
- Deja un registro en `logs/ingestion.log` con la fecha y hora de la
  descarga.

Ese `adult_raw.csv` **no se sube a GitHub** — el `.gitignore` ya lo
bloquea a propósito, porque el proyecto pide que el dataset se obtenga
por script, no que viaje copiado dentro del repositorio.

---

## Si el script falla

| Problema | Qué hacer |
|---|---|
| `pip` no se reconoce | Usar `python -m pip install -r requirements.txt` en vez de `pip install ...` |
| Error de conexión / no descarga | Revisar que haya internet; reintentar |
| Python no instalado | Bajarlo de python.org, marcando "Add to PATH" |
| Sigue sin funcionar | Como solución temporal, pedile a Dalay el `adult_raw.csv` ya generado y colocalo manualmente en `data/raw/` mientras se resuelve la instalación — pero la meta es que cada quien pueda correr el script por su cuenta |

## Solución de problemas

### Error de certificado SSL en Windows

En algunos entornos de Windows/Python, la descarga desde UCI puede fallar con un error similar a:

```text
SSLCertVerificationError: certificate verify failed
```

Este error puede ocurrir cuando Python no está utilizando correctamente el bundle de certificados disponible en `certifi`.

Para verificar dónde se encuentra el archivo de certificados de `certifi`, ejecutar:

```powershell
python -c "import certifi; print(certifi.where())"
```

Si `certifi` está disponible, en PowerShell se puede indicar temporalmente a Python que utilice ese bundle de certificados:

```powershell
$env:SSL_CERT_FILE = python -c "import certifi; print(certifi.where())"
```

Luego volver a ejecutar la ingesta:

```powershell
python src/ingestion/ingest.py
```

Una ejecución correcta debería finalizar mostrando mensajes similares a:

```text
Descarga completa: 48842 filas, 14 columnas de features
Validación mínima de ingesta: OK (48842 filas, 15 columnas)
Ingesta finalizada con éxito
```

> **Nota:** esta solución mantiene activa la verificación SSL utilizando el bundle de certificados de `certifi`. No se recomienda deshabilitar la validación de certificados SSL.