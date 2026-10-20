# Experimento paralelo: ¿quién escribe la regla del juego, el LLM o el motor?

Rama `exp-reglas-motor`, separada de las entregas (no toca `kaggle/` ni las ramas candidatas).
Sin GPU: usa las partidas ya jugadas por el LLM en las prácticas (11 corridas, 8 juegos
públicos, 12.307 transiciones) guardadas en el repo privado de datos.

## Teoría

Una **regla** es un simulador del juego: dado el tablero y una acción, predice el tablero
siguiente. Con una regla verificada, el agente planifica dentro del simulador en vez de
probar en el juego, y ahorra acciones y, sobre todo, tiempo de deliberación.

Hay dos fuentes posibles de reglas:

1. **El LLM la escribe** (`step`, lo que hoy verifica el injerto). Es flexible, pero cuesta
   tokens, que es el recurso que limita el puntaje: el modelo reescribe `step` muchas veces
   por nivel.
2. **El motor la aprende solo** (`vivos/mundo.py`: sprite rígido, parche relativo de un click
   y memoria exacta). No cuesta tokens, pero su lenguaje de reglas es limitado.

**Hipótesis de cobertura complementaria:** las dos fuentes cubren tipos de juego distintos.
Si es así, un híbrido (el motor entrega gratis las reglas que sabe aprender y el LLM solo
completa el resto) cubriría más niveles que cualquiera de las dos, sin sumar tokens.

**Predicción que la refutaría:** que el motor no logre reglas completas en niveles donde el
LLM tampoco las logra.

## Método

- `convertir.py`: eventos del harness → trayectorias (cuadros, acciones, cortes de nivel).
- `medir_modelo.py`: por nivel jugado, si el LLM escribió un `step` y si el auto-chequeo dijo
  que explicaba **todas** las transiciones del nivel con al menos 3 transiciones. Solo
  corridas con `--vivos-auto` o posteriores.
- `medir_motor.py`: el motor aprende en línea, nivel por nivel, y predice la transición
  siguiente antes de verla. Se compara el interior (sin la franja de 4 celdas y sin la banda
  de relojes). Se considera **regla completa** si, desde la 4.ª transición con cambio, acierta
  al menos el 90 % (con al menos 3 medidas).

Las dos métricas no son idénticas:

- La del LLM es en muestra: el `step` se escribe después de ver las transiciones.
- La del motor es fuera de muestra: predice antes de ver.

Esto favorece al LLM. Las unidades también difieren un poco: el LLM cuenta niveles jugados y
el motor cuenta tramos entre cortes, porque un RESET o un game over parten el nivel.

## Resultados (medidos, 20-oct-2026)

| Juego | LLM: algún `step` | LLM: regla completa | Motor: regla completa | Motor: acierto por transición |
|---|---|---|---|---|
| ft09 | 0 % | 0 % | **53 %** | 70 % |
| tu93 | 81 % | **15 %** | 0 % | 5 % |
| lp85 | 41 % | **11 %** | 0 % | 5 % |
| ls20 | 17 % | 0 % | 0 % | 35 % |
| vc33 | 2 % | 2 % | 0 % | 5 % |
| ar25 | 0 % | 0 % | 0 % | 4 % |
| cd82 | 0 % | 0 % | 0 % | 13 % |
| sb26 | 0 % | 0 % | 0 % | 5 % |
| **Total** | 28 % (89/320) | **6 %** (18/320) | **7 %** (34/483) | 18 % |

1. **Se confirma la complementariedad.** El motor logra reglas completas justo donde el LLM
   nunca escribe una (ft09: 53 % frente a 0 %). El LLM las logra donde el motor no
   (tu93 y lp85). No hay ningún juego cubierto por los dos.
2. **La cobertura total sigue siendo chica.** Combinadas, 3 de 8 juegos tienen alguna regla
   completa. En ar25, cd82 y sb26 no la logra ninguna de las dos, y en vc33 solo hay una
   adopción aislada del modelo de estado.
3. **Las reglas del LLM no muestran ahorro de tiempo** en los datos (tu93 y lp85, niveles
   completados):

   | Regla del LLM en el nivel | Niveles | Mediana por nivel |
   |---|---|---|
   | Completa | 16 | 3,9 min |
   | Incompleta | 59 | 3,6 min |
   | Sin regla | 47 | 3,5 min |

   Es una correlación con confusión (la regla aparece en los niveles difíciles), no un
   efecto causal, pero no hay señal de ahorro.

## Porcentajes de éxito estimados (suposiciones, no datos)

Escenario híbrido: el motor le entrega al LLM sus reglas completas, por ejemplo con un
`step` ya verificado en el prompt.

| Pregunta | Estimación | Base |
|---|---|---|
| Niveles con regla completa (públicos) | ~13 % (6 % + 7 %, sin solaparse) | suma de lo medido |
| Que el LLM use la regla que le da el motor | 30–60 % | "ofrecer no es usar" (estado 0/5), pero un `step` hecho es más fácil de adoptar que una API nueva |
| Tiempo ahorrado en esos niveles | 0–50 % | hoy, 0 % medido con reglas del LLM |
| Mejora en puntaje oculto antes del 2-nov | **10–15 % de probabilidad de superar +1 punto** | ~13 % × adopción × ahorro, frente a un ruido de ~2 puntos por entrega |
| Mejora si se amplía el lenguaje del motor (objetos con comportamiento condicional, contadores, empujar) | desconocida; es la palanca real | el motor está en 0 % en 5 de 8 juegos |

## Conclusión

- **La teoría se sostiene:** las dos fuentes de reglas son complementarias.
- **No alcanza para el cierre:** la cobertura combinada es baja (~13 % de los niveles) y no hay
  evidencia de que una regla ahorre tiempo.
- **Lo que daría el salto:** un motor con un lenguaje de reglas más rico, que aprenda solo en
  la mayoría de los juegos. Es trabajo posterior al cierre y una sección de "trabajo futuro"
  del paper, con estos números como justificación.

## Reproducir

```
python3 experimentos/reglas/convertir.py <living-mesh-datos/salidas> trayectorias/practicas
python3 experimentos/reglas/medir_modelo.py <living-mesh-datos/salidas>
python3 experimentos/reglas/medir_motor.py trayectorias/practicas
```

`trayectorias/` contiene tableros de juegos ARC y no se versiona.
