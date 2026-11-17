# Artículos "no encontrados" y fichas inválidas (v11.2)

Diagnóstico sin anotación humana. Corpus penal de 1,412 sentencias; fichas de `llama3.1:8b`.

## Artículos citados en la ficha que no aparecen en la sentencia

Muestra de 40 de 429 (semilla 42). En 39 de los 40, el artículo **sí** está en la sentencia y el que fallaba era el
extractor, no el LLM:

| Por qué no se encontraba | Ejemplo en la sentencia |
|---|---|
| Lista con fracciones o párrafos intercalados | "artículos 20 fracción i, 133 fracción iii, 461, 467 fracción vii, 475, 478 y 479" |
| Otra palabra en lugar de "artículo" | "los arábigos 461, 473 y 475"; "los diversos 61, 71, 471"; "en relación al 219" |
| Palabras partidas por el PDF | "art ículo s 154, 155"; "numeral es 250, 251" |
| Número antes del ordenamiento | "317 y 327 del código de procedimientos"; "16 constitucional" |

Solo uno (el artículo 262, en una sentencia con páginas omitidas) no aparece en el texto.

**Cambio:** `fichas.articulos_en_sentencia` lee la sentencia de forma tolerante. Toma los números que siguen a esas
palabras, hasta un punto y coma o un punto (como máximo 160 caracteres), y los que van justo antes de "del código",
"de la ley" o "constitucional". Excluye fechas, años, días y meses. La ficha se sigue leyendo de forma estricta.

**Control de la tolerancia:** se tomaron los artículos de la ficha de otra sentencia elegida al azar y se contó
cuántos se "encuentran" en esta:

| Lectura | Artículos de otra ficha que se encuentran |
|---|---|
| Estricta | 874 de 4,709 (18.6 %; los artículos procesales comunes aparecen en casi todas) |
| Tolerante | 1,187 de 4,709 (25.2 %) |

Es decir, alrededor de 8 % de los artículos que de verdad no están en la sentencia pasarían por encontrados.

**Resultado:**

| | Antes | Después |
|---|---|---|
| Artículos no encontrados | 429/5,096 (8.4 %) | 23/5,096 (0.5 %) |
| Fichas con alguno | 259/1,346 (19.2 %) | 22/1,346 (1.6 %) |

Corridas de `fichas-diagnostico`: antes `20261116-213316_1fc6dd4`, después `20261116-213758_19f3672`.

De los 23 que quedan, unos 10 no aparecen en ninguna parte del texto: son posibles inventos o están en páginas
omitidas. El resto son errores de lectura poco frecuentes ("arículo174"; una lista que sigue al nombre de una ley).
Esta tasa entra al informe solo como diagnóstico.

## Fichas con JSON inválido (66 de 1,412)

| Error tras el reintento | Fichas |
|---|---|
| Falta `delitos` | 58 |
| Faltan `delitos` y `resumen` | 4 |
| Falta `resumen` | 1 |
| JSON mal formado | 2 |
| Sentido no interpretable | 1 |

**Cambio:** si después del reintento solo faltan campos obligatorios, un tercer intento corto pide únicamente esos
campos (`prompts.PROMPT_FICHA_FALTANTES`) y los une a la respuesta anterior. No se rellenan con la categoría del
portal, porque esa es la referencia con la que se mide la familia. Las fichas inválidas se vuelven a intentar en
el siguiente `fichas-batch`. Cuántas quedan inválidas se sabrá cuando se regeneren, en la siguiente tanda de GPU.
