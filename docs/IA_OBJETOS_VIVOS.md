# IA de Objetos Vivos — especificación de diseño

Estado: decidido para la línea 0.6.x. Como `OBJETO_VIVO.md`, este documento
manda sobre el código cuando discrepan.

## 1. Qué es

Una inteligencia hecha de una **colonia de objetos vivos chicos, legibles y
verificados** que aprende cómo funciona un mundo **experimentando**, en lugar de
depender de un modelo monolítico. La arena para medirla es ARC-AGI-3 (ARC Prize
2026): juegos por turnos de 64×64 celdas y 16 colores, sin instrucciones, donde
hay que descubrir las reglas y la meta jugando, y donde el puntaje premia ganar
con **pocas acciones** (eficiencia frente a una base humana).

La regla que ordena todo, trasladada desde el Objeto Vivo:

> **La evidencia es de confianza; las reglas no.** Lo observado (cada transición
> que el juego devolvió) es la especificación: inmutable, la escribe el mundo.
> Una regla es una implementación no confiable que dice qué hace una acción o un
> contacto. Entra en uso solo si explica toda la evidencia de su ranura; cuando
> la evidencia la contradice se degrada y se repara con un candidato verificado.

| Objeto Vivo (malla) | IA de Objetos Vivos |
|---|---|
| spec escrita por una persona | evidencia escrita por el juego (`Transicion`) |
| implementación no confiable | regla (`Inerte`, `Mueve`, `Bloquea`, …) |
| `ensures` / `invariant` violado | la regla predijo mal una transición real |
| reparación: enumerador → LLM | enumerador de reglas desde la transición que contradijo |
| oráculo + reverificación | la candidata se juzga contra **toda** la evidencia de su ranura |
| HEALTHY / DEGRADED / QUARANTINED | idéntico, con el mismo umbral de 3 reparaciones fallidas |
| identidad estable | la ranura (`efecto de ACTION1`, `contacto con color 9`) |
| diario | el diario de transiciones y de veredictos, exportable a JSON |

## 2. Decisiones de arquitectura

- **Paquete nuevo `vivos/`, en proceso.** Kaggle evalúa sin internet y en un
  solo contenedor; los cuatro procesos de la malla agregan latencia y no aportan
  nada ahí. La malla queda como está (sus tests siguen verdes) y la IA aplica el
  mismo contrato dentro de un proceso.
- **Sin LLM en el camino del juego en 0.6.** Qwen 7B en CPU tarda segundos por
  decisión y el puntaje cuenta acciones, no tiempo, pero el plazo de evaluación
  sí existe. El nivel LLM queda como fase posterior, con el mismo papel que en la
  malla: proponer candidatos que el verificador acepta o rechaza.
- **Sin aprendizaje por gradiente.** Todo lo aprendido es una regla con nombre,
  inspeccionable y reproducible desde el diario.
- **Dependencias:** `numpy` para la IA; `arc-agi` / `arcengine` solo para jugar
  (`requirements-arc.txt`). La IA recibe cuadros como arreglos y no importa el
  kit oficial.

## 3. Percepción (`vivos/percepcion.py`)

- Se usa el **último** cuadro de cada respuesta (una acción puede devolver
  varios; el último es el estado).
- **Letterbox**: filas y columnas exteriores de un solo color, centradas y sin
  ese color adentro. **Fondo** = color más frecuente dentro del área de juego.
  Puede salir mal (en un laberinto chico las paredes ganan): por eso ninguna
  decisión depende del fondo, solo lo usa como primera suposición de "piso";
  lo que se puede pisar lo dicen las reglas de contacto.
- **Objetos** = componentes 4-conexos de un color distinto del fondo. Un marco
  (letterbox, muro perimetral) nunca es meta.
- **Relojes** (barras de pasos, contadores, timers): un color es reloj si cambió
  en al menos el 80 % de las transiciones (mínimo 4) y su firma más común
  (desplazamiento o variación de cantidad) cubre el 60 % de sus cambios en al
  menos dos contextos distintos (otra acción, o la misma acción con y sin otros
  cambios). El avatar no califica porque su desplazamiento depende de la
  acción. De dos candidatos que se pisan (la barra y el piso que descubre al
  gastarse) el reloj es el más raro. Se enmascara solo la caja del reloj
  (donde cambió, más las partes del mismo color pegadas a esa zona); lo que
  esté al lado en la misma fila o columna sigue visible. Un reloj detectado no se olvida en el nivel y se
  hereda al siguiente.
- **Efecto de una transición**: primero movimientos (misma forma y color,
  desplazada; los colores raros se mueven, los abundantes son lo que se
  descubre), después recoloreos (mismas celdas, otro color), y lo que quede sin
  explicar como *desaparece*, *aparece* u *otro*. Sin cambios fuera del ruido =
  *nada*. Los desplazamientos se guardan en unidades del tamaño del objeto, así
  una regla sobrevive a un nivel con otra escala de cámara.

## 4. La colonia de reglas (`vivos/reglas.py`)

Cada regla ocupa una **ranura**; una ranura tiene a lo sumo una regla vigente y
su LKG.

| Ranura | Reglas posibles | Confirmaciones para usarla |
|---|---|---|
| `accion:<A>` (acciones simples) | `inerte` · `mueve:<color>:<fx>:<fy>` (si ese color se mueve, se mueve exactamente así) · `cambia` | 3 · 1 · 1 |
| `contacto:<color>` (el avatar intenta entrar) | `bloquea` · `atraviesa` · `activa` (cambia otra cosa o sube de nivel) · `fatal` | 2 · 2 · 1 · 1 |
| `click:<color>` (ACTION6 sobre un objeto) | `inerte` · `recolorea:<b>` · `quita` · `otro` · `fatal` | 2 · 1 · 1 · 1 · 1 |

**Confirmaciones = estados distintos.** Repetir la misma acción en el mismo
estado no suma evidencia: un avatar trabado en una esquina no convierte una
flecha en `inerte`.

**Juicio.** Cada regla juzga una transición: confirma (+1), contradice (−1) o no
aplica (0). `mueve` no se contradice con un intento bloqueado. El desenlace de
la partida (GAME_OVER, subir de nivel) se le atribuye al contacto, nunca a la
acción. Un GAME_OVER durante un contacto (`muerte`) confirma `fatal` pero no
refuta lo que ya se sabía de ese color: la partida puede terminar por otra
causa, como una barra de energía, justo al pisar el piso.

**Ciclo de vida.** `healthy` → contradicción → `degraded` → el enumerador
propone candidatas desde la transición que contradijo; cada una se juzga contra
**toda** la evidencia de la ranura y se instala la que no tenga contradicciones
(empate: más confirmaciones, después la más específica, después la más corta)
→ `healthy`, con la anterior como LKG. Mientras está degradada, cada evidencia
nueva es otro intento. Tres intentos fallidos seguidos → `quarantined`: la
ranura no se usa para decidir y su color se trata como intransitable.

**Reinterpretación.** La evidencia (cuadros, acción, desenlace) es inmutable;
su lectura no. Cuando la percepción aprende un reloj nuevo, las transiciones del
nivel se releen con la máscara nueva y todas las ranuras se reconstruyen desde
el diario. Lo mismo pasa con los contactos cuando cambia el modelo del avatar.

**Conjeturas.** Si tres de ACTION1-4 mueven al mismo color en direcciones
opuestas dos a dos y la cuarta no tiene una regla usable, se conjetura la
dirección espejo. Una conjetura no entra en la colonia: solo deja planificar.
Usarla es un experimento; si falla con el frente libre queda refutada.

**Las reglas persisten entre niveles**: qué tecla mueve, qué color frena, qué
color mata y qué color hace ganar se usan desde la primera acción del nivel
siguiente. Esa es la transferencia de habilidades.

## 5. Decisión (`vivos/agente.py`)

En orden, en cada turno:

1. **Plan en curso**, mientras cada paso salga como se predijo.
2. **Meta dirigida (con avatar conocido).** Avatar = color de las reglas `mueve`.
   Se planifica en imaginación, sin gastar acciones, con BFS sobre las
   posiciones del avatar y los desplazamientos de las reglas (y conjeturas).
   Son transitables el fondo, la máscara, el propio avatar y los colores
   `atraviesa`; no lo son los colores `bloquea`, `fatal` o en cuarentena. Metas,
   por prioridad: colores que ya hicieron subir de nivel, colores sin probar,
   colores `activa`; empate por largo del camino. Un objeto no se toca dos veces
   desde el mismo estado ni más de tres veces por nivel.
3. **Exploración por grafo.** Nodo = cuadro sin ruido; arista = acción → nodo
   siguiente. Se prueba primero la acción no intentada más prometedora según la
   colonia (descarta `fatal`; deja para el final `inerte` y los choques contra
   `bloquea`; adelanta clicks sobre colores con efecto conocido o sin probar);
   si no queda ninguna, se navega por aristas conocidas al estado más cercano
   con acciones pendientes.
4. Si todo está agotado, una acción no descartada al azar (semilla fija).

**Clicks**: un click por objeto (la celda más central), hasta 64 por estado;
nunca los 4096 puntos, nunca el marco ni el ruido.

**GAME_OVER**: se manda RESET. La acción que terminó la partida se marca en el
grafo; se evita recién cuando termina la partida dos veces desde el mismo
estado (una vez puede ser culpa de otra cosa). **Subida de nivel**: se guarda la
habilidad (las acciones del nivel), se reinician el grafo y la percepción
(heredando los relojes) y la colonia sigue.

## 6. Juego y medición

- `jugar_arc.py --juego ls20` juega con el kit oficial (`arc_agi.Arcade`; el modo
  sale de `OPERATION_MODE`: `normal`, `online`, `offline`, `competition`).
  Exporta la colonia de cada juego a `vivos-data/<juego>.json` y, si el kit
  trae la base humana (`baseline_actions`), estima el puntaje (`vivos/puntaje.py`:
  `min(1.15, humanas/acciones)²` por nivel, promedio ponderado por índice).
- `jugar_arc.py --local todos` juega el banco local (`vivos/juegos.py`, hecho
  con ARCEngine, sin red): `laberinto` (reloj que avanza, 2 niveles con escalas
  distintas), `interruptores` (solo clicks, con señuelos), `llave` (tocar la
  llave abre la puerta), `trampa` (lava que mata) y `energia` (barra que se
  gasta y mata).
- `bench_arc.py` compara con un agente aleatorio y verifica la aceptación.
- `--segundos` pone un tope de tiempo por juego (estado `TIEMPO`): con plazo de
  evaluación es mejor cortar un juego que perder la corrida entera.
- `vivos/adaptador_agents.py` arma la clase para `arcprize/ARC-AGI-3-Agents`.

Aceptación de 0.6.0: en el banco local la IA gana todos los niveles de cada
juego dentro de 300 acciones, con menos acciones que el aleatorio. Corre en CI
y en `probar.sh`.

Resultado (entorno de desarrollo, 20 semillas, tope 600 acciones):

| Juego | Niveles | IA: acciones (mediana / peor) | Aleatorio: niveles | Aleatorio: acciones (mediana) |
|---|---|---|---|---|
| laberinto | 2/2 siempre | 55 / 69 (camino mínimo ≈ 44) | 0–2 | 600 |
| interruptores | 2/2 siempre | 7 / 8 (mínimo 6) | 2 | 114 |
| llave | 1/1 siempre | 40 / 46 | 0–1 | 405 |
| trampa | 1/1 siempre | 22 / 31 | 0 | 600 |
| energia | 1/1 siempre | 57 / 57 | 0 | 600 |

Entre 5 y 18 ms por decisión en CPU. El nivel 2 del laberinto, más largo y con
otra escala, se resuelve casi por el camino mínimo porque las reglas del nivel 1
(teclas, paredes, salida, reloj) ya están en la colonia.

**Lo que esto no demuestra.** El banco es propio y chico; los juegos de la
competencia son más variados (objetos empujables, reglas que dependen de otro
objeto, metas que no son "tocar algo"). Falta medir sobre los 25 juegos
públicos de ARC-AGI-3 con el kit oficial, que en la mini PC tiene red: es el
paso siguiente (`jugar_arc.py --todos`).

## 7. Fuera de alcance en 0.6.x

- Proponer reglas con un LLM (fase siguiente, mismo verificador).
- Reglas condicionales compuestas (efecto que depende de otro objeto) y
  objetos empujables; se agregan como nuevos tipos de regla con su ranura.
- Un reloj que recorre el área de juego se enmascara en toda la caja que
  barrió: tapa lo que haya debajo de su recorrido.
- Entrenar redes. Cualquier componente aprendido tiene que poder escribirse
  como regla con nombre.
