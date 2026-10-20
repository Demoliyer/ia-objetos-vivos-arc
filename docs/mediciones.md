# Mediciones ARC-AGI-3 — IA de Objetos Vivos

## Kaggle (set oculto, leaderboard público ≈ 50 % de los juegos)

| Fecha (UTC) | Notebook / versión | Public score | Nota |
|---|---|---|---|
| 2026-10-14 | Starter (agente propio, 41238b4) v1 | 0.07 | puesto ~3868 |
| 2026-10-15 | Starter v1 (re-entrega por error de selección) | 0.06 | confirma estabilidad: ruido entre corridas ≈ 0.01 |
| 2026-10-16 10:23 | `arc-agi-3-vivos`: Franzen hito 2 + `--vivos` (ref 57265504) | **26.39** | en la práctica es la base de Franzen: el modelo nunca llamó las herramientas. Franzen reportó 27,89 para su base (dato de él) |
| 2026-10-17 00:40 | `arc-agi-3-vivos`: iba a ser la base, corrió con `--vivos` (ref 57289126) | **25.38** | 2.ª muestra de la base. Ruido oculto ≈ ±1 punto (26,39 / 25,38; solo 2 muestras) |
| 2026-10-18 00:39 | `--vivos-auto` (4e5fb5f, con arreglos de la revisión) (ref 57331916) | **29.28** | **+3,4 sobre la media de la base (25,9)**; supera el umbral fijado de 28. Primera mejora propia en ocultos (una sola muestra). **Candidata a entrega final** |
| 2026-10-19 00:41 | `vivos-v3` 81e458f, `--vivos-auto` (ref 57375180) | **26.70** | **No supera 29,28**: queda pegado a la base (25,38 / 26,39). O algo de v3 resta (freno, costo, avisos) o el 29,28 fue en parte ruido. Con una muestra por variante no se puede separar |
| 2026-10-20 00:40 | **2.ª muestra de `--vivos-auto`** (4e5fb5f, idéntico, ref 57415286) | **25.71** | **El 29,28 fue en buena parte suerte.** Mismo código: 29,28 y 25,71 → el ruido oculto es de ±2 puntos, no ±1. Medias: vivos-auto 27,5 (2 muestras), base 25,9 (2), v3 26,7 (1). Ninguna variante se separa claramente de la base |
| 2026-10-21 00:10 (programada) | `vivos-v4s` f34aaf8 (v4 sin freno, tope 2,5 s), `--vivos-auto` | pendiente | suma una muestra de la línea auto + mejoras |

Lecciones de Kaggle:
- En el diálogo Submit, verificar **notebook y versión** antes de enviar (el selector de notebook vuelve al último usado).
- `workflow_dispatch` no distingue un input vacío del default: para "sin opciones" usar `args=-`. Verificar siempre la descripción de la entrega.
- **El guardado rápido de 15 min no mide comportamiento** (el servidor tarda ~12 min en arrancar). Para medir: `sin_entregar` con `--juegos=... --minutos=40`; la salida completa se guarda sola en el repo privado `Demoliyer/living-mesh-datos` (workflow `kaggle-salida`) y se resume con `kaggle/analizar_salida.py` (puntaje desde `benchmark.json`).
- **Al contar uso de herramientas, sacar primero el texto que inyectamos nosotros** (ayuda, recordatorio, auto-chequeo). Un conteo sin esa limpieza dijo "4 de 4 juegos" y era 1 de 4. Para `observe`/`advance` contar `def observe(` (la ayuda nunca lo contiene).
- El puntaje oculto tarda ~8-14 h en aparecer después de la entrega.
- El "Save & Run" del notebook de Franzen dura 9 h de GPU: usar siempre el workflow `kaggle-experimento`.
- **Ruido oculto (corregido el 20-oct):** el mismo código dio 29,28 y 25,71. Con 1 muestra por variante no se detectan diferencias menores a ~3-4 puntos; con ~11 entregas restantes, cualquier mejora chica es indetectable en el puntaje oculto.
- Leaderboard 2026-10-16: 1.º Tufa Labs 68,91; 2.º 67,49; 10.º 41,19; 20.º 37,27. Bronce hasta el puesto 428 de ~4.300. Probable (inferencia) un montón de copias de Franzen en 25-28.
- GPU usada la semana del 17-oct: ~19 h de 30 (estimado: medir-v3, medir-v4, medir-v4b y medir-v4c ≈ 1,1 h cada una).

## Banco de juegos públicos (sin entregar, mismos 8 juegos, 40 min)

Juegos: ls20, tu93, ft09, vc33, lp85, ar25, sb26, cd82. Puntaje = `final_score` del harness (fórmula oficial ×100). Infla mucho respecto de los ocultos (base ~50-69 aquí, ~26 en ocultos).

| Fecha (UTC) | Variante | Media | Tokens generados | ar25 | cd82 | ft09 | lp85 | ls20 | sb26 | tu93 | vc33 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-17 13:58 | base | 68,62 | 1,13 M | 100 | 48,9 | 47,6 | 100 | 12,7 | 97,6 | 67,1 | 75,0 |
| 2026-10-17 17:32 | base (2.ª corrida) | **50,36** | 0,93 M | 41,7 | 2,5 | 100 | 58,3 | 9,1 | 94,2 | 21,9 | 75,0 |
| 2026-10-17 14:50 | `--vivos-auto` | 79,98 | 1,12 M | 100 | 100 | 100 | 100 | 6,5 | 100 | 33,3 | 100 |
| 2026-10-17 16:11 | `vivos-v3` (1.ª versión) | 57,26 | 0,94 M | 41,7 | 100 | 100 | 58,3 | 17,4 | 97,0 | 22,2 | 21,4 |

**Ruido del banco (medido):** la misma base dio 68,6 y 50,4; diferencia media por juego 31 puntos, máxima 58. **Una corrida del banco no sirve para ordenar variantes por puntaje.**

**Factor de confusión: velocidad de la máquina.** "Rápidas" (~1,12-1,13 M tokens en 40 min: base 68,6, auto 80,0) vs "lentas" (~0,93-0,94 M: base2 50,4, v3 57,3). Dentro de cada par, la variante con herramientas quedó arriba (+11,4 y +6,9). El juez es la entrega oculta.

Uso del simulador en el banco: auto → lp85 (18 `def step`, 26 `check_model`, 9 perfectos, 4 planes) y tu93 (10 `def step`, 2 planes); v3 → ls20 (4 `def step`, 2 planes).

tu93 nivel por nivel (base vs auto, misma velocidad): auto usó menos acciones por nivel (N2 15 vs 21, N3 19 vs 28, N4 22 vs 42) pero 1,5-3× más tokens por nivel y pasó 5 niveles de 9 en vez de 8: **el costo del simulador es tiempo, no acciones**. El modelo reescribió `step` en cada nivel (~10 definiciones en 5 niveles).

## Corridas de medición de comportamiento (sin entregar)

| Fecha (UTC) | Corrida | Juegos × min | Resultado |
|---|---|---|---|
| 2026-10-17 01:10 | `medir-vivos`: `--vivos` | ls20, ft09, vc33, tu93 × 45 | 457 pedidos, 977 acciones. **0 llamadas, 0 `def step`** en los 4 juegos |
| 2026-10-17 02:26 | `estres-rend`: 14 streams + `SPEC_STEPS=4` | 14 juegos × 20 | **el servidor no arranca** (QSA ≤ 4 tokens de borrador). `preparar_franzen` rechaza `SPEC_STEPS > 3` |
| 2026-10-17 03:00 | `estres-14`: 14 streams | 14 juegos × 20 | 396 respuestas, **550 acciones** |
| 2026-10-17 03:35 | `estres-10`: CFG original | mismos 14 juegos × 20 | 461 respuestas, **872 acciones** → 14 streams descartado |
| 2026-10-17 04:10 | `medir-fuerte`: `--vivos-fuerte` | ls20, ft09, vc33, tu93 × 45 | 1121 acciones. Solo tu93: 9 `def step`, 8 `check_model`, 2 planes |
| 2026-10-17 10:20 | `medir-auto`: `--vivos-auto` | ls20, ft09, vc33, tu93 × 45 | 1069 acciones. 2/4: ls20 y tu93 (148 auto-chequeos, 4 planes) |
| 2026-10-17 12:15 | `humo-auto`: `--vivos-auto` arreglado | ls20, tu93 × 25 | sin errores; ls20 8 planes (largos 8, 16); ls20 21,4, tu93 33,3 |
| 2026-10-18 11:14 | `medir-v3`: `vivos-v3` antes de H1–H5, `--vivos-auto` | ft09, lp85, ls20, tu93, vc33 × 45 | 1067 acciones. **`step` en 3/5** (lp85 14 `def step`, 48 auto-chequeos, 9 perfectos; tu93 12 `def step`, 4 planes, uno de 16 acciones; vc33 6 `def step`, 248 transiciones chequeadas). 0 errores del injerto (los 7 tracebacks son del código del modelo). Freno: **7 en vc33** (con un `step` que explicaba todo: 11 "explains everything") y 2 en lp85 (el "0" anterior era un error del analizador, corregido en 900d0be); vc33 tuvo 12 auto-chequeos, no 2. Avisos de global / nivel nuevo: 0. NameError del modelo por constantes sueltas (BARS, P, rows). Puntajes (ruido): ft09 100, lp85 100, ls20 28,0, tu93 46,7, vc33 52,0 |
| 2026-10-18 13:03 | `medir-v4`: `vivos-v4` 22309b6 (ayuda con `step` primero), `--vivos-auto` | mismos 5 × 45 | **`observe`/`advance`: 0/5.** `def step`: lp85 12, tu93 3, ls20 2. Anti-repetición: 12 veces en lp85 y 12 en tu93. Freno: 4 en lp85, 5 en tu93. 0 errores del injerto. Puntajes (ruido): media 81,4 |
| 2026-10-18 14:22 | `medir-v4b`: 9d8e861 (ayuda con el **modelo de estado primero** + cambios de Argon) | mismos 5 × 45 | **Modelo de estado: 0/5** (la ayuda llegó a los 5; en vc33 el modelo escribió `def observe` 2 veces pero como función auxiliar para leer el tablero, sin `advance`). `step` cae: lp85 1 `def step` (contra 12-14); tu93 14 `def step`, 10 planes (largos 15-19). 0 errores del injerto. Puntajes (ruido): media 77,6. **Orden de la ayuda revertido** (una sola corrida: observación, no prueba) |
| 2026-10-18 15:40 | `medir-v4c`: eec41d5 (`step` primero otra vez + todos los arreglos) | mismos 5 × 45 | **vc33 adopta `observe`/`advance` solo: 97/97 transiciones correctas** (campos L, g, y), luego "captures nothing yet" al cambiar de nivel y fallas por campo localizadas; tu93 8 `def step`, 2 planes. Aviso de constantes perdidas: 4 en vc33 (`a`), 2 en tu93 (`PASS`), ambos correctos. UNCHANGED: 3 + 3. 0 errores del injerto. Puntajes (ruido): media 73,8 |

| 2026-10-19 22:02 | `medir-v4s`: f34aaf8 (v4 sin freno, tope 2,5 s) | mismos 5 × 45 | Freno 0, 0 errores, 0 cortes por tiempo. `step` en lp85 y tu93 (tu93: 25 auto-chequeos "explica todo"); UNCHANGED 32 en tu93 y 10 en lp85. Puntaje de práctica 61,2 (ruido) |
| 2026-10-20 00:56 | `medir-v5`: c503bfa (v5: error del `step` en componentes, correcciones de Astra) | mismos 5 × 45 | 0 errores, freno 0. Solo tu93 usó `step` (2 `def step`). 44 avisos "In components:" revisados a mano: **correctos** (p. ej. "the b component moved by (+0,+6), but your step leaves the board unchanged" = `step` con coordenadas fijas en el nivel nuevo), pero **el modelo no reaccionó** (40 UNCHANGED, no reescribió `step`). No cumple la regla (≥ 2/5): v5 no va al oculto |

**Oportunidad de v6 (cortar una tanda en la primera divergencia), medida sin GPU en 13 prácticas:** 924 turnos con tandas de ≥ 2 acciones (7.603 acciones); solo **24** con un `step` verificado antes (148 acciones); 11 divergieron; como máximo **74 acciones evitables** en total (< 2 por juego, ~0,5 %). **v6 descartada**: casi nunca hay un simulador verificado cuando el modelo juega.

## Dónde se pierde el puntaje (análisis del 20-oct, sin GPU, 64 partidas de práctica guardadas)

Fórmula oficial (`taaf.game.GameRun._compute_final_score`, espejo de `arc_agi.scorecard` v0.9.8): por nivel `min(115, (humano/acciones)² × 100)` si se completó, si no 0; peso = número de nivel; total = suma ponderada / suma de pesos, **con tope en el porcentaje ponderado de niveles completados**. Reproduce los 64 `final_score` guardados con error 0.

| Pérdida media por juego (64 partidas) | Puntos |
|---|---|
| Niveles no completados (100 − techo) | **27,5** |
| Acciones de más, a completitud constante (techo − puntaje) | **2,7** |

- Eliminar toda la penalización por acciones, con los mismos niveles completados, recuperaría como máximo ~2,7 puntos en estas prácticas. Una mejor planificación podría igual ayudar a completar niveles: lo acotado es su beneficio directo sobre las acciones (precisión de Astra).
- Las 37 partidas no ganadas terminaron todas por tiempo (`cancelled`), ninguna por game over.
- **No suelen estar trabadas:** mediana de 5 min en el último nivel (13 % del tiempo de la partida) contra 4,8 min por nivel completado (p90 15). Trabas largas en un solo nivel: ft09 dos veces (30 y 38 min, 75-85 %), ar25 y ls20 en menor medida (2-3 de 37).
- **Presupuesto oficial:** 110 juegos, `concurrency=120`, 532 min, 10 flujos activos (`ARC3_MAX_ACTIVE_STREAMS`): ~48 min-flujo por juego. Banco de 8 juegos ≈ comparable; prácticas de 5 juegos dan ~2× cómputo por juego.
- **Tokens por nivel completado** (incluye lo gastado en el nivel incompleto): base ~27.800, nuestras versiones ~26.500 de media; más caras en vc33 y tu93, más baratas en cd82 y sb26. Descriptivo (base con 2 partidas por juego).
- **Costo del auto-chequeo:** 10-16 chequeos con texto por juego × ≤ 2,5 s = < 1 min de 45, en CPU (mientras tanto la GPU atiende a otros juegos). No explica nada.
- **Lectura:** el cuello de botella es el ritmo (tokens generados por nivel bajo un presupuesto compartido), no la eficiencia en acciones. Nuestras versiones no lo cambiaron.

## Tokens por turno y tope (análisis del 20-oct, pedido de Astra, sin GPU)
- Unidades: solicitud (tope duro `max_tokens` 12 288), invocación (cede a los 2048 tokens con "llamá a action()"; se revisa entre solicitudes), intervalo entre acciones (tokens exactos del `history`, sin tope salvo el tiempo).
- Intervalos (2284, 64 juegos): mediana 2753, p95 14 313, máx 86 238. Los >8k son el 16 % de los intervalos y el 51,5 % de los tokens (exceso sobre 8k: 22,4 %). Base/banco8: 5,9 % de tokens por encima de 12k; vivos-auto/banco8: 13,9 % (2 contra 1 corrida, sin separar del ruido).
- Solicitudes: 20 de 4867 cortadas por `length` (0,4 %). Un intervalo >12k son mediana 6 solicitudes (la mayor pesa 39 %) y 3 cesiones ignoradas: no es un pensamiento largo, es una cadena de investigación.
- Los >8k completan niveles por intervalo como los cortos (14–15 %), pero por token mucho menos; el 85 % de sus tokens está en niveles que sí se completaron. Correlación entre juegos de la fracción >8k con niveles por 100k tokens: r = −0,59 (dirección causal no identificable). Repetición literal de código: 0 %; snippets con error: 12,9 % frente a 7,7 %.
- Decisión: **no probar tope**. `max_tokens` no alcanza los intervalos largos, el tope blando ya se ignora y un tope duro cortaría planes verificados (caso vc33 de 45k). Entregas restantes: ver la auditoría de Astra abajo (el plan de 7 entregas era incorrecto: no existe muestra de Franzen sin injerto). Si se prueba algo: solo `LOCAL_ANALYZER_YIELD_TOKENS` 1024 contra v4s, 2 prácticas por brazo, quedarse si niveles/100k tokens ≥ +15 % en ambas.
- Paquete de revisión para Astra: `revision-tokens-2026-10-20.zip` (código base/vivos-auto/v4s, config del motor, benchmark y transcripciones, scripts que reproducen los 64 puntajes con error 0).

## Auditoría de Astra y congelado por huella (20-oct)
- **Las dos "base" (26,39 y 25,38) fueron Franzen + `--vivos`**, no Franzen sin cambios (dato: descripción de las entregas en Kaggle). `--vivos` agrega la ayuda de las herramientas al prompt aunque el modelo no las llame. La entrega del 15-oct con descripción "Base hito 2 Franzen, sin cambios" (0,06) fue en realidad el Starter propio re-entregado por error de selección. **No hay ninguna muestra oculta de Franzen sin injerto.**
- Reconstrucción sin GPU: Kaggle no deja bajar versiones viejas de un notebook por API (403, también las propias). Pero el notebook entregado = `preparar_franzen.py`(notebook de Franzen, commit, opciones) es determinista: regenerar `medir-v5` da las mismas 25 celdas que el notebook que corrió. Con el commit vigente en cada lanzamiento (inferido por hora; el input `ref` no quedó registrado) y las opciones de la descripción:

| Entrega | Commit | Opciones | Huella del notebook |
|---|---|---|---|
| 57265504 (26,39) | 044f1c5 | `--vivos` | be36c273659a |
| 57289126 (25,38) | 68e99b0 | `--vivos` | be36c273659a (idéntica) |
| 57331916 (29,28) | 9010ed8 | `--vivos-auto` | ff42d7f18a5a |
| 57415286 (25,71) | 900d0be | `--vivos-auto` | ff42d7f18a5a (idéntica) |
| 57375180 (26,70) | 81e458f | `--vivos-auto` | 5e0b90762a34 |
| Franzen sin injerto (nueva) | e98ebe2 | — | 5c66785823b6 |
| v4s (esta noche) | c95f937 | `--vivos-auto` | 413a0b289a64 |

- Franzen sin injerto difiere del original solo en código que corre si `not TRUE_SUBMISSION` (chequeo de GPU y guardado rápido): en la corrida oficial es Franzen sin cambios.
- Notebook de Franzen: fuentes idénticas el 15-oct y el 20-oct (huella 4056fbfc69bb). Riesgo residual no controlado: los datasets `dfranzen/taaf-kaggle-source-bundle-copy` y `dfranzen/pennyroyal-v253` se montan sin versión fija; los modelos sí van fijos (`/1`) y la imagen de docker por sha.
- Congelado desde el 20-oct (`experimento.sh` en ia-objetos-vivos e98ebe2 y vivos-v4s c95f937): exige la huella del origen de Franzen, imprime la del notebook preparado, la agrega a la descripción de la entrega (`[nb …]`) y `--huella=<h>` corta antes de subir si no coincide. Nuevo workflow `kaggle-versiones` (solo lectura) guarda el notebook actual de cada kernel y la lista de entregas en living-mesh-datos/notebooks/.
- Umbral: no hay umbral fijo de superioridad; con 2–4 muestras por brazo el desvío está mal estimado. Se elige por medias sin afirmar que la diferencia está demostrada; la incertidumbre se calcula con los resultados finales. El 26,70 de v3 no prueba que el freno perjudicó.
- **Plan:** 9 entregas intercaladas; ver calendario abajo. Identidad de las históricas: notebook recuperado de Kaggle por versión y coincidente con la reconstrucción; el vínculo entrega→versión es deducido (orden + huella).
- Dependencias: no se pudo verificar que la API permita fijar versión de los datasets en los metadatos del kernel; se registra lo que cada corrida cargó (`kaggle/fuentes_salida.py`). Las 10 prácticas del 17 al 20-oct con registro cargaron lo mismo.
- Pruebas del congelado: `tests/test_huella.py` (8 pruebas; suite completa 234 OK).

## Entregas: trazabilidad por entrega (actualizar en cada una)

Tres fases separadas: **notebook enviado** (versión del kernel y su huella), **fuentes del guardado rápido** (lo que cargó la corrida visible, `kaggle/fuentes_salida.py`), **fuentes de la evaluación oculta** (siempre desconocidas: Kaggle no expone esa corrida; que coincidan con el guardado de la misma versión es inferencia).

Recuperación directa (20-oct, `kaggle-recuperar`, solo lectura): con `version_label="v<N>"` (Kaggle/kaggle-cli #1200/#1201) Kaggle devolvió las versiones 1–22 de demoliyer22/arc-agi-3-vivos con fuente, metadatos y salida; `current_version_number` coincide con lo pedido y hay 19 huellas distintas (control: la versión se respeta). Las versiones 23–40 no existen (404). Guardado en living-mesh-datos/historico/20261020-1608-demoliyer22__arc-agi-3-vivos.

Vínculo entrega → versión: la API de entregas no informa la versión del kernel. Se deduce del orden de los 22 lanzamientos exitosos del workflow (uno por versión) y se confirma con la huella: la versión deducida tiene la huella reconstruida desde el commit y las opciones de la descripción.

Fuentes de referencia: harness ARC3-Inference aa69123 (DIRTY: modificaciones sin identificar, se espera el parche de Franzen) + tufa fe9f7c4 + re-arc-3 57e46d619d; sglang 0.5.19+gd00d88efc8d6; torch 2.13.0+cu130; modelos `/1`. Las 22 versiones y las 10 prácticas con registro cargaron eso en el guardado rápido.

| Ref Kaggle | Fecha (UTC) | Condición | Versión kernel | Huella v1 / sha tipado | Notebook enviado | Fuentes guardado | Fuentes evaluación | Puntaje | Incidencias |
|---|---|---|---|---|---|---|---|---|---|
| 57196905 | 14-oct 22:39 | Starter propio v1 | otro kernel | — | — | — | desconocidas | 0,07 | otro sistema |
| 57230469 | 15-oct 12:36 | Starter propio v1 | otro kernel | — | — | — | desconocidas | 0,06 | descripción errónea "Base hito 2 Franzen, sin cambios" |
| 57265504 | 16-oct 10:23 | Franzen + `--vivos` | v1 | be36c273659a / 3b1e66a6… | recuperado; = reconstrucción (044f1c5) | ok (dirty sin identificar) | desconocidas | 26,39 | registrada como "base"; el prompt cambia aunque no llame herramientas |
| 57289126 | 17-oct 00:40 | Franzen + `--vivos` | v2 | be36c273659a / 3b1e66a6… | recuperado; = reconstrucción (68e99b0) | ok (dirty) | desconocidas | 25,38 | nombre "base" con `--vivos` |
| 57331916 | 18-oct 00:39 | vivos-auto | v14 | ff42d7f18a5a / 0595d5c8… | recuperado; = reconstrucción | ok (dirty) | desconocidas | 29,28 | — |
| 57375180 | 19-oct 00:41 | v3 (`--vivos-auto`) | v19 | 5e0b90762a34 / f30dc768… | recuperado; = reconstrucción (81e458f) | ok (dirty) | desconocidas | 26,70 | efecto del freno no identificado |
| 57415286 | 20-oct 00:40 | vivos-auto | v21 | ff42d7f18a5a / 0595d5c8… | recuperado; = reconstrucción | ok (dirty) | desconocidas | 25,71 | — |

Identidad = huella v1 (etiqueta) / `sha256_celdas_tipadas` (esquema de Astra, `kaggle/identidad_notebook.py`; incluye el tipo de celda). Esperados en `kaggle/identidades_candidatos.json` y exigidos por `experimento.sh` antes de subir: origen 4056fbfc69bb / 5b752682…; A v4s 413a0b289a64 / 53cdaf3e…; B vivos-auto ff42d7f18a5a / 0595d5c8…; C Franzen sin injerto 5c66785823b6 / 4f889ac2…. Ramas: candidatos ia-objetos-vivos da8cfda (B, C) y vivos-v4s 0838331 (A, notebook sin cambios); control/recolectores en ia-objetos-vivos.

Códigos de `fuentes_salida.py`: 0 coinciden los campos registrados; 1 difieren; 2 faltan; 3 ambos. Comparar campos registrados no certifica bytes.

## Calendario fijado (20-oct, antes de ver resultados)

Condiciones: **A** = v4s (`vivos-v4s` 0838331, `--vivos-auto`, huella 413a0b289a64); **B** = vivos-auto (`ia-objetos-vivos` da8cfda, `--vivos-auto`, huella ff42d7f18a5a); **C** = Franzen sin injerto (`ia-objetos-vivos` da8cfda, sin opciones, huella 5c66785823b6). Intercaladas, una por día a las 00:10 UTC (21:10 del día anterior en Uruguay).

| Día (UTC) | 21-oct | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |
|---|---|---|---|---|---|---|---|---|---|
| Condición | A | C | B | A | C | B | A | C | A |

Resultado: A 4 muestras nuevas, B 2 nuevas (4 con las reconstruidas), C 3 nuevas. Reserva: 30-oct, 31-oct, 1-nov, 2-nov.

**29-oct ~15:00 UTC: revisión de estado.** La decisión con nueve muestras se toma recién cuando estén los nueve puntajes (si el del 29 no salió, se espera y se decide en cuanto salga). Dos comparaciones por medias (sin afirmar superioridad; la incertidumbre se informa con los resultados finales): (1) solo entregas verificadas; (2) sumando las reconstruidas. Si las dos eligen los mismos 2 finalistas: la reserva repite a los finalistas o queda para contingencias. Si difieren: 30 y 31-oct = B (4 muestras verificadas de vivos-auto), y se vuelve a comparar. Cierre: arcprize.org dice "November 2, 2026 - Submissions due" sin hora; la hora de Kaggle no está confirmada. **Selección provisional de las 2 finales el 1-nov 15:00 UTC** (Franco la marca en la web); se actualiza solo si la entrega del 2-nov da un resultado admisible antes del cierre.

## Procedimiento de cada entrega

1. Verificar que no haya otra entrega ese día (kaggle-estado `competitions submissions`) y que la rama apunte al commit del calendario.
2. Lanzar kaggle-experimento con `ref`, `nombre` y `args` incluyendo `--huella=<h>`. Si falla por origen de Franzen cambiado o huella distinta: **no entregar, no forzar con `--origen`**, avisar a Franco.
3. Verificar la anotación de opciones y que la descripción en Kaggle termine en `[nb <h>]`.
4. Recuperar esa versión exacta con `kaggle-recuperar` (desde = hasta = N): notebook enviado, metadatos, estado y salida del guardado rápido. Correr `fuentes_salida.py` e `identidad_notebook.py` sobre lo recuperado y comprobar huella v1 y sha tipado. DIFIERE = incidencia, no se combina sin decisión explícita; INCOMPLETO se anota como tal. Fuentes de la evaluación oculta: no observables con el procedimiento actual.
5. Agregar la fila a la tabla (ref, condición, commit, huella, fuentes, identidad "verificada", incidencias) y, ~14 h después, el puntaje.


## Revisiones

- **Agente independiente (2026-10-17)** sobre `--vivos-auto`: 2 críticos (print dentro de `step` rompía el protocolo JSON del sandbox; `step` con bucle infinito → timeout de 30 s en cada snippet), 1 alto (`step` que llama `action()` gastaba acciones reales), 3 medios/bajos. Todos arreglados (4e5fb5f) con tests; los 16 casos hostiles pasan.
- **Agente independiente (2026-10-18)** sobre `vivos-v3`: veredicto "entregar con cambios". H1 (un `goal` que devuelve 1/0 se leía como costo; 5 o NaN como "ya llegaste"), H2 (`shift_test` descartaba casos por cambios del HUD exterior), H3 (aviso de global tapaba el informe), H4 (comparación fuera del try), H5 (NaN como costo). Arreglados en 81e458f con 8 tests nuevos.
- **Agente independiente (2026-10-18), diseño v4 (colonia de reglas `rule_*`)**: "adelante con cambios, pero no así". Huecos: precisión trivial (regla vacía = 100 %), `None` ambiguo, HUD en el residuo, planificar con colonia incompleta, orden de composición. Dato clave de las transcripciones: los `step` ya son de una sola mecánica; lo que los rompe son coordenadas fijas; el modelo ya arma estados abstractos solo (vc33 `getstate`/`render`, lp85 `tokpos`, tu93 `solve*`). → v4 pasó a **modelo de estado `observe`/`advance` verificado por campo** (Fable no se pudo usar: pide créditos).
- **Agente independiente (2026-10-18), código v4**: "entregar con cambios". `except:` pelado tragaba la alarma → 30 s (venía de v3) → hilo vigía que entrega lo hecho y cierra limpio; `plan_state` con meta no booleana; anti-repetición mentía tras un snippet sin chequeo; `step` no se chequeaba con estado retenido; claves `1`/`'1'`. Todo arreglado (92c55f8). Una regresión propia (alarma repetida) la detectaron los casos hostiles antes del commit.
- **Astra (GPT), revisión de v5 (19-oct)**: ejecutó el código y encontró 6 fallas; 5 reproducidas y corregidas en c503bfa (intercambio de igual abundancia, fragmentos de un componente partido, recoloreo de región grande, recorte antes de comparar, región > 900 celdas); forma final en la comparación; redacción "component". Corrigió una cifra mía (12.565 transiciones, no 25.000). Su chequeo de determinismo: 3 de ~11.000 pares con resultados distintos, los 3 solo en la barra de pasos del borde.
- **Astra, diseño de v6 (19-oct)**: devolver normal deja seguir al snippet (usar excepción); dividir la tanda cambia la ejecución → el corte va en el bucle por acción del host (`solver.py`, que ya tiene `guard_hook` por posición); medir la oportunidad antes de construir → medida y descartada.
- **Argon (Gemini), 3 rondas (2026-10-17)**: aciertos tomados → freno por turnos sin jugar, planificación por costo con plan parcial, aviso al cambiar de nivel solo si el `step` pasa la prueba de desplazamiento, aviso si `step` lee el tablero global. Errores descartados → su "guardián anti no-op" ya existe en Franzen; los diffs por objeto ya existen (`frame_diff`).
- **Argon, 3 rondas sobre v4 (2026-10-18)**: tomado → estado constante no cuenta como acierto; paredes en el estado (ayuda); prueba de desplazamiento para `observe`; prueba de `observe` sobre el tablero nuevo al cambiar de nivel; celdas ambientales (reloj) fuera de los puntos ciegos; `None` como falso en `goal`; `cost=True` en `plan_with_state`; memoria del último tipo de modelo; aviso de constantes que se pierden entre snippets (confirmado: el harness solo retiene `def`). Descartado → tratar números ≠ 0/1 como verdad (diría "llegaste" sin llegar); modificar el harness para retener constantes (código ajeno en la corrida oficial). Su hipótesis de poner el estado primero en la ayuda se midió y **no** subió la adopción (0/5) y bajó `step` en lp85 → revertida.

## Lectura del harness de Franzen (github.com/da-fr/arc-agi-3-solution)

Ya trae RESET no expuesto, guardas de no-op/estado viejo/terminal (`ARC3_BATCH_NOOP_BLOCK`, `ARC3_STALE_STATE_BLOCK`), explicación de la barra de pasos, `frame_diff` por objetos, funciones retenidas durante todo el juego (`ARC3_PERSISTENT_FUNCTIONS_SCOPE=game`). **Solo retiene `def`: nunca re-ejecuta asignaciones** (`retained_functions.restore_functions`). Un proceso nuevo por snippet; `initial["state"]` admite claves extra. Su CFG de servidor (10 streams, SPEC_STEPS=3) ya está en el óptimo que pudimos tocar.

## Injerto propio: simulador verificado

`vivos/simulador.py` + `kaggle/injerto_vivos.py`.

- `--vivos`: el modelo no lo usa (0/4). Ocultos ≈ base (26,39 / 25,38).
- `--vivos-fuerte`: + recordatorio por turno (1/4).
- `--vivos-auto` (4e5fb5f, rama `ia-objetos-vivos`): + `step` parcial `{(fila, col): color}` + auto-chequeo aislado después de cada snippet (2/4). **Ocultos 29,28.**
- `vivos-v3` (rama `vivos-v3`, 81e458f): + planificación por costo explícita (`cost=True`), freno tras 4 turnos sin jugar, aviso de nivel nuevo con prueba de desplazamiento, aviso de lectura de globales. Entrega 2026-10-19 00:10 UTC.
- `vivos-v4` (rama `vivos-v4`, 5682a7a): + modelo de estado opcional (`observe`/`advance`, `check_state` por campo, puntos ciegos, estado constante, celdas ambientales, `plan_with_state` con `cost=True`, coordenadas fijas en `observe`, prueba de `observe` al cambiar de nivel); informe de `step` por familia de acción y cambios no predichos / inventados; anti-repetición del lado del agente; aviso de constantes que se pierden; vigía contra `except:` pelado; memoria del último tipo de modelo. Ayuda con `step` primero. 56 tests + casos hostiles. Entrega 2026-10-20 00:10 UTC.
- **Hallazgo (para el paper, "ofrecer no es usar")**: el modelo no adopta una API nueva por más que se la expliquen (0/5 en dos órdenes de ayuda); cuando adopta el estado (vc33 en medir-v4c) es porque ya lo hacía por su cuenta. Lo que mueve el puntaje es verificar lo que el modelo ya escribe.
- Idea v5 (Argon + nuestro): que el modelo siga escribiendo solo `step` y que **nuestro motor de objetos** (`vivos/percepcion.py`, traslación rígida) traduzca los errores de celdas a objetos ("el objeto rojo se movió 1 a la derecha y tu `step` no lo movió"). Es donde el motor de Objetos Vivos aporta sin pedirle nada nuevo al modelo.

## Banco local del agente propio (juegos públicos, en pausa)

Base vigente 41238b4: 0,942 % media de 3 semillas (17/18/15 niveles). Historia completa en versiones anteriores de este doc y en `docs/paper/borrador.md` §4.

## Descartado (medido)

- Agente propio: conjetura del espejo; enmascarar barra con colores socios; memoria de clicks; cuerpo compuesto; diferencial de victoria mínimo.
- Guardas sobre Franzen (RESET en paso 0, máscara de HUD, veto): cubiertas o bajo el ruido.
- Paquete de rendimiento: `SPEC_STEPS=4` rompe el servidor; 14 streams rinde menos.
- `--vivos` solo con ayuda (0/4) y `--vivos-fuerte` (1/4): superados por `--vivos-auto`.
- Decidir variantes con 1 corrida del banco: el ruido medido (31 puntos por juego) lo invalida.
- Hipótesis de Argon (19-oct): el tope de 1,0 s del auto-chequeo de v3 cortaba el chequeo y apagaba "planificá". **Refutada con los datos**: 0 cortes por tiempo en 5 prácticas con ese tope. El sospechoso con evidencia es el **freno** (7 disparos en vc33 de medir-v3 con un `step` perfecto).
- Colonia de reglas `rule_*` por celdas (diseño v4 inicial): descartada por la revisión antes de programarla.
- Ayuda con el modelo de estado primero (medir-v4b): adopción 0/5 y menos `step` en lp85.

## Pendiente (orden)

1. Candidatas finales: con el ruido medido (±2), elegir por media de muestras, no por el máximo (el máximo de pocas muestras es suerte). Hoy: vivos-auto (27,5 en 2), v3 (26,7 en 1), base (25,9 en 2).
2. 2026-10-21 00:10 UTC: `vivos-v4s`.
3. Análisis sin GPU: dónde se pierde el puntaje (niveles no completados vs. acciones de más, por juego).
4. Paper (8 de noviembre): "ofrecer no es usar" (estado 0/5, diagnóstico en componentes ignorado, oportunidad de v6 ~0,5 %) y el ruido oculto medido.
5. Reservar las últimas 2-3 entregas para repetir candidatas finales.
