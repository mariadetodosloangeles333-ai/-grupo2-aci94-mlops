# Validación funcional — Data Quality y Quality Gates

**Fecha:** 25/08/2026  
**Rama validada:** `feature/data-quality`

## Objetivo

Validar en un segundo entorno que el pipeline de Data Quality es reproducible,
que las reglas automáticas identifican condiciones esperadas y que bloquean
condiciones críticas.

## Prueba 1 — Ejecución con datos reales

Entrada:
- 48,842 filas
- 15 columnas

### Gates RAW

- PASS: 5
- WARNING: 3
- FAIL: 0

Warnings esperados:
- valores faltantes conocidos;
- 29 duplicados conocidos;
- etiquetas de `income` pendientes de normalización.

### Resultado de limpieza

- 52 duplicados eliminados.
- Dataset resultante: 48,790 filas × 15 columnas.

### Gates CLEAN

- PASS: 7
- WARNING: 0
- FAIL: 0

Resultado:
`adult_clean.csv` generado correctamente.

## Prueba 2 — Valor imposible

Se modificó temporalmente en memoria:

`age = -5`

Resultado:

`numeric_ranges → FAIL`

Estado general:

`FAIL`

El Gate detectó correctamente un valor fuera del dominio permitido.

## Prueba 3 — Esquema inválido

Se eliminó temporalmente en memoria la columna:

`income`

Resultado:

- `schema → FAIL`
- `target_completeness → FAIL`
- `target_values → FAIL`
- `data_types → FAIL`

Estado general:

`FAIL`

El sistema detectó correctamente la ausencia de una columna obligatoria.

## Integridad del repositorio

Las pruebas negativas se realizaron únicamente sobre DataFrames temporales
en memoria.

Verificación final:

`nothing to commit, working tree clean`

## Conclusión

La versión evaluada del pipeline:

- procesa correctamente el dataset real;
- distingue WARNING de FAIL;
- valida nuevamente los datos después de la limpieza;
- detecta valores imposibles;
- detecta alteraciones del esquema;
- genera el dataset procesado únicamente cuando las validaciones finales
  son satisfactorias.

**Resultado de validación: APROBADO para integración a `develop`.**