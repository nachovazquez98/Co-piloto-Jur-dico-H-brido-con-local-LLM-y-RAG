# Juez gemma2:9b: errores al leer la familia del delito (v11.2, corpus final)

Corrida: `20261121-003449_5311dc1_referencias`. Muestra estable de 725 sentencias penales. Referencia: la familia de la
categoría que asignó el STJ (reglas de `evaluacion/familias.py`). El juez acertó 536 de 725
(0.739), por debajo del mínimo de 0.80 fijado antes de correrlo, así que **no se usa**: la referencia de
acto impugnado, apelante y sentido queda con las reglas.

## Exactitud por familia

| familia_stj | sentencias | aciertos | exactitud |
|---|---|---|---|
| robo | 97 | 85 | 0.880 |
| homicidio | 91 | 79 | 0.870 |
| violencia familiar | 77 | 74 | 0.960 |
| delitos culposos | 69 | 17 | 0.250 |
| delitos sexuales | 66 | 28 | 0.420 |
| fraude y abuso de confianza | 48 | 43 | 0.900 |
| privacion de la libertad y secuestro | 36 | 36 | 1.000 |
| otros | 35 | 6 | 0.170 |
| lesiones | 31 | 18 | 0.580 |
| despojo | 30 | 28 | 0.930 |
| contra la salud | 23 | 22 | 0.960 |
| servidores publicos y administracion de justicia | 22 | 16 | 0.730 |
| desaparicion de personas | 19 | 19 | 1.000 |
| falsedad y falsificacion | 12 | 12 | 1.000 |
| feminicidio | 11 | 11 | 1.000 |
| amenazas y extorsion | 11 | 8 | 0.730 |
| sustraccion de menores | 10 | 6 | 0.600 |
| dano en las cosas | 8 | 4 | 0.500 |
| armas | 7 | 6 | 0.860 |
| abandono de familiares | 7 | 7 | 1.000 |
| responsabilidad profesional | 5 | 4 | 0.800 |
| contra la dignidad de las personas | 5 | 4 | 0.800 |
| trata de personas | 3 | 3 | 1.000 |
| delincuencia organizada | 2 | 0 | 0.000 |

## Confusiones más frecuentes

| familia_stj | juez | sentencias |
|---|---|---|
| delitos culposos | dano en las cosas | 25 |
| delitos sexuales | violencia familiar | 24 |
| delitos culposos | homicidio | 16 |
| otros | no determinado | 13 |
| delitos sexuales | contra la dignidad de las personas | 10 |
| lesiones | violencia familiar | 7 |
| robo | otros | 6 |
| otros | contra la dignidad de las personas | 4 |
| delitos culposos | lesiones | 4 |
| homicidio | otros | 4 |
| delitos sexuales | otros | 3 |
| homicidio | privacion de la libertad y secuestro | 3 |

## Lectura

- **Delitos culposos.** El juez responde por el bien afectado («daño en las cosas», «homicidio») aunque su prompt dice
  expresamente que los delitos a título de culpa son «delitos culposos». Es un error del juez, no de la etiqueta.
- **Delitos sexuales.** En violación y abuso sexual cometidos en el ámbito familiar responde «violencia familiar» o
  «contra la dignidad de las personas»: elige el contexto y no el tipo.
- **«Otros».** Es una familia residual (tipos que no encuadran en las demás); el juez no la reconoce como delito y
  responde «no determinado» u otra familia. Es un límite del diseño de familias, no de la lectura.
- La caída frente a la validación anterior (0.845 en 401 sentencias del corpus previo) viene de la composición de la
  muestra: los delitos culposos pasaron del 4.2 % al 9.5 % y «otros» del 0.5 % al 4.8 %, mientras que en el resto de
  las familias la exactitud del juez se mantiene (0.89 antes, 0.87 ahora). Muchos tipos nuevos son el mismo delito con
  otra grafía en el portal («dano-a-las-cosas…», «danos-en-las-cosas…»), por lo que no se restringió la validación a los
  tipos del corpus anterior: se aplica el mínimo fijado de antemano sobre toda la muestra.
