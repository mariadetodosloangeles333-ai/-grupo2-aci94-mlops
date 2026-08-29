# Revisión y análisis final del modelo

## Identificación y trazabilidad

- Problema: clasificación binaria de `income`.
- Algoritmo: `RandomForestClassifier`.
- Feature set: `v2_without_sensitive`.
- Versión de datos: `adult_clean_v1`.
- Semilla aleatoria: `42`.
- División de test: `20%`, estratificada.
- Run final de MLflow: `11c8bc44969d40938763b78ead476dfd`.
- Modelo registrado: `adult-income-classifier`.
- Versión registrada: `1`.
- Alias de MLflow: `production`.
- Métrica principal de selección: G-Mean.
- Tratamiento del desbalance: `class_weight="balanced"`.

## Desempeño y generalización

- G-Mean: 0.8374
- Recall: 0.8660
- Especificidad: 0.8098
- F1: 0.7011
- ROC AUC: 0.9219
- Diferencia absoluta validación-test: 0.0036

El modelo obtuvo un buen desempeño general. La diferencia de 0.0036 entre el G-Mean de 
validación y el de test indica que el desempeño se mantuvo sobre los datos reservados, por lo que no se observa evidencia importante de sobreajuste.

## Matriz de confusión y tipos de error

La matriz de confusión muestra que el modelo predijo correctamente 2.023 de los 2.336 casos reales de la clase `>50K` y produjo 313 falsos negativos.

Este comportamiento es coherente con el tratamiento del desbalance de clases. Así 1.412 casos de la clase `<=50K` fueron clasificados como `>50K`. Este fue el compromiso asumido al utilizar `class_weight="balanced"` para mejorar la detección de la clase minoritaria y reducir los falsos negativos.

## Análisis por subgrupos

El modelo no presenta el mismo comportamiento en los dos subgrupos analizados.

| Métrica | Female | Male | Brecha absoluta |

| Recall | 0.7572 | 0.8874 | 0.1302 |
| Especificidad | 0.9453 | 0.7207 | 0.2246 |
| G-Mean | 0.8460 | 0.7997 | 0.0463 |
| Tasa real `>50K` | 0.1152 | 0.3036 | 0.1884 |
| Tasa predicha `>50K` | 0.1356 | 0.4639 | 0.3283 |

En el subgrupo Female, de los 383 casos reales de la clase `>50K`, 290 fueron clasificados correctamente y 93 fueron falsos negativos. De los 2.943 casos reales de la clase `<=50K`, 2.782 fueron identificados correctamente y 161 fueron falsos positivos.

En el subgrupo Male, de los 1.953 casos reales de la clase `>50K`, 1.733 fueron clasificados correctamente y 220 fueron falsos negativos. De los 4.479 casos reales de la clase `<=50K`, 3.228 fueron identificados correctamente y 1.251 fueron falsos positivos.

El modelo es más conservador al clasificar a una mujer en la clase `>50K` e identifica mejor los casos de hombres pertenecientes a esa clase. Por otro lado, presenta una mayor especificidad para Female y una cantidad considerablemente mayor de falsos positivos para Male.

Estas diferencias deben documentarse y monitorearse durante las siguientes etapas para detectar y mitigar posibles comportamientos desiguales del modelo. Los resultados no demuestran por sí solos que exista discriminación causal o jurídica, pero sí evidencian diferencias importantes entre los tipos de error de ambos subgrupos.

También debe considerarse que los subgrupos y sus clases internas presentan distribuciones desiguales. Male representa aproximadamente el 65.9% del conjunto de test y contiene una proporción considerablemente mayor de casos `>50K` que Female. Por ello, sus métricas se calculan con más observaciones, suelen ser más estables e influyen más en el resultado general.

Además, el dataset procede del censo estadounidense de 1994, por lo que refleja condiciones económicas, laborales y sociales de esa época. En consecuencia, no puede asumirse que sus distribuciones representen adecuadamente la población laboral actual.

## Feature Importance

El análisis de importancia confirma que el modelo utiliza principalmente información relacionada con estado civil, educación, edad, relación familiar, ganancias de capital y horas trabajadas. También confirma que las variables sensibles `sex`, `race` y `native-country` no se utilizaron directamente como variables predictoras.

Sin embargo, features como `marital-status`, `relationship` y `occupation` podrían actuar como variables proxy, debido a su relación indirecta con el sexo, la situación económica o determinadas condiciones sociales. Esto podría contribuir al comportamiento diferente observado entre los subgrupos.

Por lo tanto, no se puede concluir que el modelo esté libre de sesgos únicamente porque las variables sensibles fueron excluidas. Las diferencias identificadas deben documentarse y monitorearse posteriormente.

La importancia calculada por el Random Forest representa contribución predictiva dentro del modelo, pero no demuestra causalidad ni indica que una feature aumente o reduzca directamente la probabilidad de pertenecer a la clase `>50K`.

## Curva ROC

La curva ROC se mantiene ampliamente por encima del clasificador aleatorio y alcanza un ROC AUC de 0.9219. Esto demuestra que el modelo tiene una buena capacidad general para separar y ordenar los casos de las clases `<=50K` y `>50K`.

Sin embargo, un ROC AUC alto no demuestra igualdad de comportamiento entre los subgrupos ni ausencia de sesgos. Por esta razón, debe interpretarse junto con la matriz de confusión, la importancia de features y las métricas obtenidas para Female y Male.

## Conclusión final

El modelo Random Forest superó los criterios de validación, fue registrado como la versión 1 de adult-income-classifier y recibió el alias production dentro del entorno técnico del proyecto.

No obstante, se identificaron diferencias de recall, especificidad, G-Mean y tasa de predicción positiva entre Female y Male. Estas diferencias pueden estar relacionadas con la composición desigual de la muestra, las condiciones históricas representadas en el dataset y el posible uso indirecto de variables proxy.