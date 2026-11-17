# Sentido del fallo: desacuerdos entre el LLM y las reglas (v11.2)

Diagnóstico sin anotación humana. Corpus penal de 1,412 sentencias; fichas de `llama3.1:8b`.

## Desacuerdos (muestra de 40 de 189, semilla 42)

Se leyó la sección de puntos resolutivos de cada sentencia y se clasificó el desacuerdo:

| Categoría | Casos | Ejemplo |
|---|---|---|
| Error del LLM | 24 | "Primera. Se revoca la resolución…" y la ficha dice CONFIRMA; también "se modifica" o "se ordena la reposición" leídos como CONFIRMA |
| REPONE o REVOCA (por definición) | 7 | "Se deja insubsistente… se ordena la reposición del procedimiento": la regla da REPONE, como está definido; el LLM, REVOCA |
| Acto distinto (cumplimiento de amparo) | 3 | El primer punto deja insubsistente la sentencia de la propia sala por orden de un amparo |
| Resolución mixta | 1 | Confirma la prescripción de un delito y la revoca para otro |
| Regla dudosa o resolutivo cortado | 5 | El verbo cayó en un salto de página o el resolutivo no se ve completo |

El error más frecuente es del LLM, y siempre en el mismo sentido: pone CONFIRMA cuando el resolutivo revoca o modifica.
Por eso la app ya muestra el sentido de las reglas cuando lo encuentran y marca el desacuerdo como "verificar".

## Sentencias sin sentido por reglas (855 antes del cambio)

- 618 son de las sentencias con páginas omitidas. En 443 de ellas, los resolutivos no están en el texto, así que
  lo que las corrige es volver a descargarlas, no las reglas.
- En las 237 completas (muestra de 25, semilla 1) había:
  - redacciones no cubiertas: "lo procedente es confirmar/revocar…" y el verbo sin "se" ("…revoca lo resuelto en la audiencia");
  - resultados que no son ninguno de los cuatro sentidos: recurso sin materia, inadmisible o extemporáneo,
    prescripción, el sentenciado dejó de ser sujeto de la ley de adolescentes. Estos deben quedar "no determinado";
  - verbos perdidos en la extracción del PDF.

## Cambio en las reglas (`fichas.sentido_por_reglas`)

1. Se agrega "lo procedente es confirmar / revocar / modificar / reponer".
2. Se agrega el verbo sin "se" (revoca, modifica o confirma, seguido de "la", "el" o "lo"), solo si la sección no
   tiene ninguno de los verbos anteriores. En la sección también se narra: en un caso, "el colegiado confirma el
   amparo" había cambiado un REVOCA correcto.
3. Se probó y se descartó "se deja insubsistente" como REVOCA. En una muestra de 4 casos, 2 eran incorrectos: la
   sentencia de la propia sala ("este ad quem") y una cuestión de competencia.
4. No se agregó un valor MIXTO. Hubo un solo caso en la muestra y obligaría a cambiar el esquema de la ficha y de
   los jueces.

Revisión manual de las lecturas nuevas: 21 de una muestra al azar, todas correctas después de los ajustes 2 y 3.

## Resultado

| | Antes | Después | Corridas |
|---|---|---|---|
| Sentido leído por reglas (todas las penales) | 557/1,412 | 608/1,412 | `20261113-000725_137c238` → `20261116-213210_4e60549` (metadatos-cobertura) |
| Sentido leído por reglas (fichas válidas) | 538/1,346 | 589/1,346 | `20261113-105003_fb1f3fe` → `20261116-213316_1fc6dd4` (fichas-diagnostico) |
| Sentido del LLM = reglas | 349/538 (64.9 %) | 391/589 (66.4 %) | ídem |

Ningún valor que ya leían las reglas cambió; solo se agregaron 51 lecturas. La concordancia no es exactitud: en
la muestra, la mayor parte del desacuerdo es error del LLM. La medición contra la referencia automática
(`fichas-referencia`) dirá cuánto acierta cada uno.
