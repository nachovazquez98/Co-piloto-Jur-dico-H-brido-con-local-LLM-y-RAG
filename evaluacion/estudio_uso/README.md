# Estudio de uso: búsqueda manual vs. Co-piloto

**Pregunta:** ¿encuentra un abogado precedentes útiles en menos tiempo con la app que buscando a mano en el portal
de sentencias del STJ Jalisco?

## Antes de empezar (quien organiza)

1. Un abogado revisa los 5 casos de `casos.jsonl` (salen de los ejemplos de la app) y cambia `borrador` a `false`.
   Mientras haya casos en borrador, el resumen sale marcado como no reportable.
2. `registro.csv` ya trae el orden y el método de 4 participantes (P01–P04). Para otro número:
   `python -m copiloto_juridico.evaluacion estudio --plantilla N` (no sobrescribe registros llenos).
3. La app debe estar en el mismo equipo, con la configuración congelada y **sin** resultados guardados de esos
   casos (borra `data/cache/resultados/` o arranca con `COPILOTO_CACHE_RESULTADOS=false`): el participante debe
   esperar el análisis real.

## Para cada participante

- Haz los casos **en el orden de tus filas** de `registro.csv`, cada uno con el método indicado:
  - `manual`: buscador público del STJ Jalisco y lo que uses normalmente, sin la app;
  - `app`: solo el Co-piloto (puedes abrir las sentencias que muestra).
- Lee el caso, pon el cronómetro y busca **precedentes de las Salas Penales que te servirían para la apelación**.
  Detente cuando tengas los que usarías o a los 20 minutos, lo que ocurra primero.
- Anota en tu fila: `minutos` (con decimales si hace falta), `precedentes_utiles` (cuántos usarías de verdad),
  `utilidad` (1 = nada útil … 5 = muy útil, sobre lo que encontraste con ese método) y un `comentario` breve.
- No compartas los casos ni lo que encontraste con otros participantes hasta que todos terminen.

Cada participante hace la mitad de los casos a mano y la otra mitad con la app; el método alterna caso a caso, los
participantes pares recorren los casos al revés y empiezan con la app, así ningún caso ni método va siempre primero.

## Resumen

`python -m copiloto_juridico.evaluacion estudio` guarda en `evaluacion/resultados/`: mediana de minutos por método,
diferencia pareada por caso (manual − app) con IC bootstrap y efecto mínimo detectable, precedentes útiles por minuto
y utilidad media. Con menos de 10 participantes se reporta como **estudio exploratorio**: describe a quienes
participaron, no permite generalizar.
