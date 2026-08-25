"""
Data Quality Gates - Validaciones automáticas

Cada regla produce uno de estos estados:

- PASS: la regla se cumple.
- WARNING: se detecta una situación que puede manejarse con la limpieza o revisión posterior.
- FAIL: el problema es crítico y el pipeline debe detenerse.
"""

# Importar librerias
from pandas.api.types import is_numeric_dtype, is_string_dtype

# Reglas automáticas implementadas

# Regla 1 - Validación del esquema de los datos

# Columnas que debe contener el Adult Dataset
EXPECTED_COLUMNS = {
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
}

# Se comprueba que el dataset cumpla con la estructura esperada
# Existen 3 resultados posibles: FAIL, WARNING y PASS
def check_schema(df):
    """
    Justificación de la regla:

    Esta regla es necesaria ya que al no validar el esquema de los datos no se puede garantizar 
    el correcto funcionamiento del pipeline posterior, en caso de no contar con alguna de las
    columnas se genera un fallo pero si se detectan columnas adicionales solo será necesario que
    se genere una advertencia.
    """

    # Convertir las columnas recibidas en un conjunto facilita su comparación
    received_columns = set(df.columns)

    # Columnas faltantes
    missing_columns = EXPECTED_COLUMNS - received_columns
    # Columnas adicionales
    unexpected_columns = received_columns - EXPECTED_COLUMNS

    # La ausencia de una columna obligatoria impide continuar - Se produce FAIL
    if missing_columns:
        return {
            "gate": "schema",
            "status": "FAIL",
            "message": (
                f"Faltan columnas obligatorias: {sorted(missing_columns)}. "
                "El pipeline debe detenerse."
            ),
        }

    # Las columnas adicionales se detectan y registran - Se produce WARNING
    if unexpected_columns:
        return {
            "gate": "schema",
            "status": "WARNING",
            "message": (
                f"Se encontraron columnas adicionales: "
                f"{sorted(unexpected_columns)}. "
                "Estas columnas no serán utilizadas por el pipeline."
            ),
        }

    # No faltan ni sobran columnas - Se produce PASS
    return {
        "gate": "schema",
        "status": "PASS",
        "message": "El dataset contiene exactamente las 15 columnas esperadas.",
    }

# Regla 2 - Validación de filas esperadas

# Cantidad oficial de registros del Adult Dataset
EXPECTED_ROWS = 48_842

# Se comprueba que el dataset esté completo
# Existen 3 resultados posibles: FAIL, WARNING y PASS
def check_row_count(df):
    """
    Justificación de la regla

    Esta regla es necesaria debido a que si no se verifca que el dataset esté completo se puede 
    perder información valiosa para el proyecto, además en caso de tener más filas de las esperadas
    se puede revisar la fuente de los datos para verificar si se realizó algún cambio o los datos
    se duplicaron en algún momento.
    """

    # Obtener la cantidad de filas del DataFrame
    row_count = df.shape[0]

    # Menos filas significa dataset incompleto - Se produce FAIL
    if row_count < EXPECTED_ROWS:
        return {
            "gate": "row_count",
            "status": "FAIL",
            "message": (
                f"El dataset contiene {row_count} filas; se esperaban "
                f"{EXPECTED_ROWS}. El dataset está incompleto."
            ),
        }

    # Las filas adicionales se detectan y registran - Se produce WARNING
    if row_count > EXPECTED_ROWS:
        return {
            "gate": "row_count",
            "status": "WARNING",
            "message": (
                f"El dataset contiene {row_count} filas; se esperaban "
                f"{EXPECTED_ROWS}. Se debe revisar si existen duplicados "
                "o una nueva versión de los datos."
            ),
        }

    # La cantidad coincide exactamente con la fuente oficial - Se produce PASS
    return {
        "gate": "row_count",
        "status": "PASS",
        "message": f"El dataset contiene las {EXPECTED_ROWS} filas esperadas.",
    }

# Regla 3 - Validación de la variable objetivo

# Se comprueba que la variable objetivo income no tenga valores faltantes
# Solo existen 2 resultados posibles: FAIL o PASS
def check_target_completeness(df):
    """
    Justificación de la regla

    Esta regla es necesaria debido a que la variable income es la variable objetivo que se desea
    predecir, si se tuvieran valores faltantes en esta variable no se podría utilizar para hacer 
    el entrenamiento supervisado requerido en el modelado. Por esta razón esta regla es más estricta
    y solo admite 2 resultados: FAIL o PASS
    """

    # Verificar que la variable objetivo exista
    if "income" not in df.columns:
        return {
            "gate": "target_completeness",
            "status": "FAIL",
            "message": "No se encontró la columna objetivo income.",
        }

    # Convertir temporalmente la columna a texto y eliminar espacios laterales
    normalized_target = df["income"].astype("string").str.strip()

    # Detectar las distintas representaciones de valores faltantes
    missing_mask = (
        normalized_target.isna()
        | normalized_target.eq("?")
        | normalized_target.eq("")
    )

    # Contar cuántas filas no tienen una etiqueta válida
    missing_count = int(missing_mask.sum())

    # Cualquier valor faltante en el target bloquea el pipeline - Se produce FAIL
    if missing_count > 0:
        return {
            "gate": "target_completeness",
            "status": "FAIL",
            "message": (
                f"La columna income contiene {missing_count} valores faltantes. "
                "El pipeline debe detenerse."
            ),
        }

    # La variable objetivo está completa - Se produce PASS
    return {
        "gate": "target_completeness",
        "status": "PASS",
        "message": "La columna income no contiene valores faltantes.",
    }

# Regla 4 - Validación de valores faltantes

# Máximo de faltantes aceptados según el diagnóstico de calidad hecho previamente
ALLOWED_MISSING_COUNTS = {
    "occupation": 2_809,
    "workclass": 2_799,
    "native-country": 857,
}

# Se comprueba que no existan valores faltantes desconocidos
# Existen 3 resultados posibles: FAIL, WARNING y PASS
def check_predictor_missing_values(df):
    """
    Justificación de la regla

    Esta regla comprueba que no existan más valores faltantes que los encontrados durante el 
    diagnóstico de calidad. En caso de existir más faltantes que los identificados previamente se 
    detiene el pipeline y en caso de detectar solo los faltantes ya conocidos se genera el WARNING 
    para indicar que se tratarán en limpieza.
    """

    # Todas las columnas esperadas excepto la variable objetivo
    predictor_columns = EXPECTED_COLUMNS - {"income"}
    # Comprobar que todas las variables predictoras estén presentes
    missing_predictor_columns = predictor_columns - set(df.columns)

    # Variables faltantes - Se produce FAIL
    if missing_predictor_columns:
        return {
            "gate": "predictor_missing_values",
            "status": "FAIL",
            "message": (
                f"Faltan variables predictoras: "
                f"{sorted(missing_predictor_columns)}."
            ),
        }

    # Guardar las cantidades encontradas en cada columna
    detected_missing = {}

    # Guardar los problemas que deben bloquear el pipeline
    violations = {}

    for column in predictor_columns:
        # Convertir temporalmente a texto y eliminar espacios laterales
        normalized_values = df[column].astype("string").str.strip()

        # Detectar NaN, símbolos "?" y textos vacíos
        missing_mask = (
            normalized_values.isna()
            | normalized_values.eq("?")
            | normalized_values.eq("")
        )

        # Cantidad total de faltantes en la columna
        missing_count = int(missing_mask.sum())

        if missing_count > 0:
            detected_missing[column] = missing_count

        # Obtener el máximo permitido; para las demás columnas el máximo es 0
        allowed_count = ALLOWED_MISSING_COUNTS.get(column, 0)

        # Registrar cualquier cantidad que supere el máximo aprobado
        if missing_count > allowed_count:
            violations[column] = {
                "detected": missing_count,
                "allowed": allowed_count,
            }

    # Se detectan más valores faltantes - Se produce FAIL
    if violations:
        return {
            "gate": "predictor_missing_values",
            "status": "FAIL",
            "message": (
                f"Se superaron los límites de faltantes: {violations}. "
                "El pipeline debe detenerse."
            ),
        }

    # Se detectan los faltantes conocidos que serán tratados en la limpieza - Se produce WARNING
    if detected_missing:
        return {
            "gate": "predictor_missing_values",
            "status": "WARNING",
            "message": (
                f"Se detectaron faltantes conocidos y tratables en la limpieza: "
                f"{detected_missing}."
            ),
        }

    # No se encontró ningún tipo de valor faltante - Se produce PASS
    return {
        "gate": "predictor_missing_values",
        "status": "PASS",
        "message": "Las variables predictoras no contienen valores faltantes.",
    }

# Regla 5 - Validación de duplicados

# Cantidad de filas duplicadas según el diagnóstico hecho previamente
EXPECTED_RAW_DUPLICATES = 29

# Se comprueba que no existan más filas duplicadas
# Existen 3 resultados posibles: FAIL, WARNING y PASS
def check_duplicates(df):
    """
    Justificación de la regla

    Esta regla es necesaria debido a que detecta valores duplicados desconocidos, en caso de 
    detectar los duplicados ya conocidos solo genera una advertencia de que se limpiarán más tarde, 
    pero si se detectan más duplicados debido a errores o algo parecido se produce un FAIL y se 
    detiene el pipeline.
    """

    # Contar las filas repetidas después de su primera aparición
    duplicate_count = int(df.duplicated().sum())

    # Más duplicados bloquean el pipeline - Se produce FAIL
    if duplicate_count > EXPECTED_RAW_DUPLICATES:
        return {
            "gate": "duplicates",
            "status": "FAIL",
            "message": (
                f"Se detectaron {duplicate_count} filas duplicadas; "
                f"el máximo conocido es {EXPECTED_RAW_DUPLICATES}. "
                "El pipeline debe detenerse."
            ),
        }

    # Se detectan duplicados conocidos - Se produce WARNING
    if duplicate_count > 0:
        return {
            "gate": "duplicates",
            "status": "WARNING",
            "message": (
                f"Se detectaron {duplicate_count} filas duplicadas conocidas. "
                "Serán tratadas durante la limpieza."
            ),
        }

    # No se encontraron registros repetidos - Se produce PASS
    return {
        "gate": "duplicates",
        "status": "PASS",
        "message": "No se detectaron filas duplicadas.",
    }

# Regla 6 - Validación de clases de la variable objetivo

# Etiquetas finales esperadas
CLEAN_TARGET_VALUES = {"<=50K", ">50K"}

# Etiquetas inconsistentes detectadas durante el diagnóstico
KNOWN_DIRTY_TARGET_VALUES = {"<=50K.", ">50K."}

# Todas las etiquetas aceptadas antes de ejecutar la limpieza
ALLOWED_RAW_TARGET_VALUES = (
    CLEAN_TARGET_VALUES | KNOWN_DIRTY_TARGET_VALUES
)

# Se comprueba que las clases de la variable objetivo sean correctas
# Existen 3 resultados posibles: FAIL, WARNING y PASS
def check_target_values(df):
    """
    Justificación de la regla

    Esta regla es necesaria debido a que es la variable objetivo que se desea predecir, si se 
    tuvieran más clases de las permitidas para realizar el modelo no se podría realizar el proyecto,
    en caso de detectar las clases vistas en el diagnóstico se realiza un WARNING para indicar que 
    se tratarán posteriormente, pero si se detectan más problemas con las clases se produce un FAIL
    y se detiene el pipeline.
    """

    # Verificar que la variable objetivo exista
    if "income" not in df.columns:
        return {
            "gate": "target_values",
            "status": "FAIL",
            "message": "No se encontró la columna objetivo income.",
        }

    # Convertir temporalmente las etiquetas a texto y eliminar espacios
    normalized_target = df["income"].astype("string").str.strip()

    # Detectar faltantes para evitar que la función falle o los ignore
    missing_mask = (
        normalized_target.isna()
        | normalized_target.eq("?")
        | normalized_target.eq("")
    )

    # Se detectan faltantes - Se produce un fail
    if missing_mask.any():
        return {
            "gate": "target_values",
            "status": "FAIL",
            "message": (
                "La columna income contiene valores faltantes y no se pueden "
                "validar todas sus categorías."
            ),
        }

    # Obtener el conjunto de categorías presentes
    detected_values = set(normalized_target.unique())

    # Identificar categorías que no pertenecen al dominio aprobado
    unknown_values = detected_values - ALLOWED_RAW_TARGET_VALUES

    # Se detectan categorías desconocidas - Se produce FAIL
    if unknown_values:
        return {
            "gate": "target_values",
            "status": "FAIL",
            "message": (
                f"Se encontraron categorías desconocidas en income: "
                f"{sorted(unknown_values)}."
            ),
        }

    # Identificar las etiquetas conocidas que requieren normalización
    dirty_values = detected_values & KNOWN_DIRTY_TARGET_VALUES

    # Se detectan las clases que requieren limpieza - Se produce WARNING
    if dirty_values:
        return {
            "gate": "target_values",
            "status": "WARNING",
            "message": (
                f"Se encontraron etiquetas que requieren limpieza: "
                f"{sorted(dirty_values)}. Serán normalizadas por clean.py."
            ),
        }

    # Solo aparecen las dos categorías finales esperadas - Se produce PASS
    return {
        "gate": "target_values",
        "status": "PASS",
        "message": "income contiene únicamente las categorías <=50K y >50K.",
    }

# Regla 7 - Validación de tipos de datos

# Variables numéricas
NUMERIC_COLUMNS = {
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
}

# Variables categóricas
CATEGORICAL_COLUMNS = EXPECTED_COLUMNS - NUMERIC_COLUMNS

# Se comprueba que el tipo de datos sea correcto
# Solo existen 2 resultados: FAIL y PASS
def check_data_types(df):
    """
    Justificación de la regla

    Esta regla es necesaria ya que los tipos de datos son fundamentales al considerar que 
    transformaciones se deben hacer, las distribuciones de las variables, el manejo de los datos, 
    etc. En caso de que algún tipo de dato sea erróneo se produce FAIL y se detiene el pipeline, 
    solo se puede continuar si todos los tipos de datos son correctos
    """

    # Comprobar que estén disponibles todas las columnas necesarias
    required_columns = NUMERIC_COLUMNS | CATEGORICAL_COLUMNS
    missing_columns = required_columns - set(df.columns)

    # Columnas faltantes - Se produce FAIL
    if missing_columns:
        return {
            "gate": "data_types",
            "status": "FAIL",
            "message": (
                f"No se pueden validar los tipos porque faltan columnas: "
                f"{sorted(missing_columns)}."
            ),
        }

    # Guardar las columnas cuyo tipo no coincide con lo esperado
    invalid_types = {}

    # Verificar las variables que deben ser numéricas
    for column in NUMERIC_COLUMNS:
        if not is_numeric_dtype(df[column]):
            invalid_types[column] = {
                "detected": str(df[column].dtype),
                "expected": "numeric",
            }

    # Verificar las variables que deben contener texto
    for column in CATEGORICAL_COLUMNS:
        if not is_string_dtype(df[column]):
            invalid_types[column] = {
                "detected": str(df[column].dtype),
                "expected": "string",
            }

    # Cualquier tipo incorrecto bloquea el pipeline - Se produce FAIL
    if invalid_types:
        return {
            "gate": "data_types",
            "status": "FAIL",
            "message": (
                f"Se detectaron tipos de datos incorrectos: {invalid_types}."
            ),
        }

    # Todas las variables tienen el tipo esperado - Se produce PASS
    return {
        "gate": "data_types",
        "status": "PASS",
        "message": "Todas las columnas tienen los tipos de datos esperados.",
    }

# Regla 8 - Validación de rangos posibles

# Rangos encontrados durante el diagnóstico de calidad
OBSERVED_NUMERIC_RANGES = {
    "age": (17, 90),
    "fnlwgt": (12_285, 1_490_400),
    "education-num": (1, 16),
    "capital-gain": (0, 99_999),
    "capital-loss": (0, 4_356),
    "hours-per-week": (1, 99),
}

# Se comprueba que el rango de las variables sea posible
# Existen 3 resultados: FAIL, WARNING y PASS
def check_numeric_ranges(df):
    """
    Justificación de la regla

    Esta regla es necesaria ya que los valores tienen que ser posibles para ser útiles, en caso de 
    que los valores estén dentro del rango observado durante el diagnóstico de calidad se continúa 
    el proceso, pero si se encuentran valores fuera de ese rango se tiene que generar WARNING o 
    FAIL según si el valor es posible o imposible.
    """

    # Comprobar que estén presentes las columnas numéricas
    missing_columns = NUMERIC_COLUMNS - set(df.columns)

    # Faltan columnas - Se produce FAIL
    if missing_columns:
        return {
            "gate": "numeric_ranges",
            "status": "FAIL",
            "message": (
                f"No se pueden validar los rangos porque faltan columnas: "
                f"{sorted(missing_columns)}."
            ),
        }

    # Evitar errores si la función se ejecuta antes del gate de tipos
    non_numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if not is_numeric_dtype(df[column])
    ]
    # Columnas no numéricas - Se produce FAIL
    if non_numeric_columns:
        return {
            "gate": "numeric_ranges",
            "status": "FAIL",
            "message": (
                f"No se pueden validar los rangos porque estas columnas "
                f"no son numéricas: {sorted(non_numeric_columns)}."
            ),
        }

    # Guardar valores imposibles y valores posibles fuera de la referencia
    failures = {}
    warnings = {}

    # AGE
    # Menores de 17 están fuera de la población y mayores de 120 se consideran imposibles
    age_failures = (df["age"] < 17) | (df["age"] > 120)
    age_warnings = (df["age"] > 90) & (df["age"] <= 120)

    if age_failures.any():
        failures["age"] = int(age_failures.sum())

    if age_warnings.any():
        warnings["age"] = int(age_warnings.sum())

    # FNLWGT
    # Un peso censal debe ser positivo
    fnlwgt_failures = df["fnlwgt"] <= 0
    fnlwgt_warnings = (
        (df["fnlwgt"] > 0)
        & (
            (df["fnlwgt"] < OBSERVED_NUMERIC_RANGES["fnlwgt"][0])
            | (df["fnlwgt"] > OBSERVED_NUMERIC_RANGES["fnlwgt"][1])
        )
    )

    if fnlwgt_failures.any():
        failures["fnlwgt"] = int(fnlwgt_failures.sum())

    if fnlwgt_warnings.any():
        warnings["fnlwgt"] = int(fnlwgt_warnings.sum())

    # EDUCATION-NUM
    # Es un código definido entre 1 y 16
    education_failures = (
        (df["education-num"] < 1)
        | (df["education-num"] > 16)
    )

    if education_failures.any():
        failures["education-num"] = int(education_failures.sum())

    # CAPITAL-GAIN
    # No puede ser negativo, pero podría superar el máximo observado
    gain_failures = df["capital-gain"] < 0
    gain_warnings = df["capital-gain"] > 99_999

    if gain_failures.any():
        failures["capital-gain"] = int(gain_failures.sum())

    if gain_warnings.any():
        warnings["capital-gain"] = int(gain_warnings.sum())

    # CAPITAL-LOSS
    # No puede ser negativo, pero podría superar el máximo observado
    loss_failures = df["capital-loss"] < 0
    loss_warnings = df["capital-loss"] > 4_356

    if loss_failures.any():
        failures["capital-loss"] = int(loss_failures.sum())

    if loss_warnings.any():
        warnings["capital-loss"] = int(loss_warnings.sum())

    # HOURS-PER-WEEK
    # Una semana tiene como máximo 168 horas
    hours_failures = (
        (df["hours-per-week"] < 1)
        | (df["hours-per-week"] > 168)
    )
    hours_warnings = (
        (df["hours-per-week"] > 99)
        & (df["hours-per-week"] <= 168)
    )

    if hours_failures.any():
        failures["hours-per-week"] = int(hours_failures.sum())

    if hours_warnings.any():
        warnings["hours-per-week"] = int(hours_warnings.sum())

    # Los valores imposibles tienen prioridad y bloquean el pipeline - Se produce FAIL
    if failures:
        return {
            "gate": "numeric_ranges",
            "status": "FAIL",
            "message": (
                f"Se encontraron valores imposibles o fuera del dominio: "
                f"{failures}."
            ),
        }

    # Los valores posibles fuera de la referencia requieren revisión - Se  produce WARNING
    if warnings:
        return {
            "gate": "numeric_ranges",
            "status": "WARNING",
            "message": (
                f"Se encontraron valores fuera de los rangos observados, "
                f"pero posiblemente válidos: {warnings}."
            ),
        }

    # Todos los valores están dentro de los rangos de referencia - Se produce PASS
    return {
        "gate": "numeric_ranges",
        "status": "PASS",
        "message": "Todas las variables numéricas están dentro de los rangos observados.",
    }

# Función para ejecutar todas las reglas definidas para obtener un diagnóstico completo

def build_gate_report(results):
    """
    Reúne los resultados individuales y calcula el estado general.
    """

    # Extraer los estados individuales
    statuses = [result["status"] for result in results]

    # Calcular el estado general
    if "FAIL" in statuses:
        overall_status = "FAIL"

    elif "WARNING" in statuses:
        overall_status = "WARNING"

    else:
        overall_status = "PASS"

    # Contar la cantidad de resultados de cada tipo
    summary = {
        "PASS": statuses.count("PASS"),
        "WARNING": statuses.count("WARNING"),
        "FAIL": statuses.count("FAIL"),
    }

    return {
        "overall_status": overall_status,
        "summary": summary,
        "results": results,
    }


def run_reference_gates(df):
    """
    Ejecuta todos los Data Quality Gates del dataset crudo.

    Esta función ejecuta todas las reglas de entrada antes de la limpieza.
    Los problemas conocidos pueden producir WARNING y permitir que el
    pipeline continúe hacia clean.py.
    """

    # Ejecutar las reglas correspondientes al dataset raw
    results = [
        check_schema(df),
        check_row_count(df),
        check_target_completeness(df),
        check_predictor_missing_values(df),
        check_duplicates(df),
        check_target_values(df),
        check_data_types(df),
        check_numeric_ranges(df),
    ]

    return build_gate_report(results)


def run_clean_gates(df):
    """
    Ejecuta los Data Quality Gates después de aplicar la limpieza.

    Los WARNING relacionados con esquema, faltantes, duplicados y etiquetas
    de income se convierten en FAIL, porque clean.py debía corregirlos.
    """

    # La cantidad oficial de filas no se revisa después de eliminar duplicados
    results = [
        check_schema(df),
        check_target_completeness(df),
        check_predictor_missing_values(df),
        check_duplicates(df),
        check_target_values(df),
        check_data_types(df),
        check_numeric_ranges(df),
    ]

    # Gates que deben producir PASS después de la limpieza
    strict_clean_gates = {
        "schema",
        "predictor_missing_values",
        "duplicates",
        "target_values",
    }

    clean_results = []

    for result in results:
        # Crear una copia para no modificar el resultado original
        clean_result = result.copy()

        # Una advertencia en estos gates indica que la limpieza no terminó bien
        if (
            clean_result["gate"] in strict_clean_gates
            and clean_result["status"] == "WARNING"
        ):
            clean_result["status"] = "FAIL"
            clean_result["message"] = (
                "La validación posterior a la limpieza falló. "
                + clean_result["message"]
            )

        clean_results.append(clean_result)

    return build_gate_report(clean_results)