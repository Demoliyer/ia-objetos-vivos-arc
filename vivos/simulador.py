"""Verificador y planificador para simuladores escritos por el modelo (injerto en el Duck harness).

El LLM escribe su hipótesis de cómo funciona el juego como una función de Python
``step(grid, action) -> grid``. Este módulo:

- ``check_model(step)``: la contrasta con TODAS las transiciones observadas del nivel actual
  y devuelve dónde se equivoca (sin gastar acciones del juego);
- ``plan(step, goal)``: busca en anchura sobre ese simulador el camino más corto a un estado
  que cumpla ``goal`` (sin gastar acciones del juego).

Es la idea de la IA de Objetos Vivos aplicada al LLM: una regla vale lo que vale su
verificación contra lo observado, y la búsqueda solo se apoya en reglas verificadas.

Restricciones del entorno de destino (sandbox del harness): Python puro, sin numpy ni
imports fuera de la lista blanca, tiempo de CPU limitado por llamada (30 s). Por eso todo
tiene presupuesto de tiempo y de nodos, y nada lanza excepciones por un ``step`` defectuoso:
se informa como desacierto.

Las grillas son listas de listas: letras de color (vista `.ascii` del harness) o enteros. Las acciones son las etiquetas
del harness: "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "UNDO", "MOUSE(row=R, col=C)".
"""

from __future__ import annotations

import re
import math
import time

_MOUSE = re.compile(r"MOUSE\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)")
_BORDE = 4  # mismo recorte que usa el harness para `gameplay_changed`
_MAX_EJEMPLOS = 6
# El sandbox corre cada snippet en un proceso nuevo y lo mata a los 30 s: el presupuesto es
# compartido entre todas las llamadas del snippet, contado desde que se cargó este módulo.
_T0 = time.monotonic()
_PRESUPUESTO_PROCESO = 24.0
_MAX_NODOS = 20000


class _Reloj:
    """Plazo de una llamada: el menor entre su `time_limit` y lo que le queda al proceso.
    Corta antes de un `step` si no entra uno más tan lento como el más lento visto."""

    def __init__(self, time_limit):
        restante = _PRESUPUESTO_PROCESO - (time.monotonic() - _T0)
        self.limite = max(0.0, min(float(time_limit), restante))
        self.t0 = time.monotonic()
        self.peor = 0.0

    def queda(self):
        return time.monotonic() - self.t0 + self.peor <= self.limite

    def medir(self, inicio):
        self.peor = max(self.peor, time.monotonic() - inicio)


def parse_action(action):
    """'MOUSE(row=3, col=7)' -> ('MOUSE', 3, 7); 'UP' -> ('UP', None, None)."""
    texto = str(action or "").strip()
    m = _MOUSE.fullmatch(texto)
    if m:
        return "MOUSE", int(m.group(1)), int(m.group(2))
    return texto.upper(), None, None


def mouse(row, col):
    """Etiqueta de click para usar en `plan(..., actions=[...])` o en `action(...)`."""
    return f"MOUSE(row={int(row)}, col={int(col)})"


def _grilla(frame_o_grilla):
    """Grilla como lista de listas. De un frame del harness se usa `.ascii` (letras de color,
    la misma vista que tiene el modelo: la grilla numérica no se le expone)."""
    ascii_ = getattr(frame_o_grilla, "ascii", None)
    if isinstance(ascii_, str) and ascii_:
        return [list(linea) for linea in ascii_.split("\n")]
    g = getattr(frame_o_grilla, "_grid", frame_o_grilla)
    return [list(fila) for fila in g]


def _clave(g):
    return tuple(tuple(fila) for fila in g)


def _diferencias(esperada, predicha, borde):
    """Celdas distintas, separadas en interior y franja de borde. None si las formas difieren."""
    if not isinstance(predicha, (list, tuple)) or len(predicha) != len(esperada):
        return None
    filas = len(esperada)
    interior, en_borde = [], []
    for r in range(filas):
        fe, fp = esperada[r], predicha[r]
        if isinstance(fp, str):
            fp = list(fp)
        if not isinstance(fp, (list, tuple)) or len(fp) != len(fe):
            return None
        cols = len(fe)
        for c in range(cols):
            if fe[c] != fp[c]:
                celda = (r, c, fe[c], fp[c])
                if borde <= r < filas - borde and borde <= c < cols - borde:
                    interior.append(celda)
                else:
                    en_borde.append(celda)
    return interior, en_borde


def _salida(copia, predicha):
    """Normaliza lo que devuelve `step(copia, action)`: grilla completa; solo los cambios como
    dict {(fila, col): color}; o None, que es la copia tal como la dejó `step` (sin cambios, o
    modificada en el lugar)."""
    if predicha is None:
        return copia
    if isinstance(predicha, dict):
        g = [fila[:] for fila in copia]
        for (r, c), v in predicha.items():
            r, c = int(r), int(c)
            if r < 0 or c < 0:  # en Python -1 sería la última fila: acá es un error del step
                raise IndexError(f"negative cell ({r}, {c})")
            g[r][c] = v
        return g
    return predicha


def _familia(action):
    """Familia de la acción para el informe: las flechas juntas, MOUSE junto, el resto por nombre."""
    nombre, _, _ = parse_action(action)
    return "arrows" if nombre in ("UP", "DOWN", "LEFT", "RIGHT") else nombre


def _faltan_inventa(interior, antes):
    """Separa las celdas erradas en cambios reales que el step no predijo (r, c, antes, real) y
    cambios que el step inventó donde el tablero quedó igual (r, c, real, predicho)."""
    faltan, inventa = [], []
    for r, c, real, pred in interior:
        try:
            previo = antes[r][c]
        except Exception:
            continue
        if real != previo and pred == previo:
            faltan.append((r, c, previo, real))
        elif pred != previo and real == previo:
            inventa.append((r, c, real, pred))
    return faltan, inventa


def _transiciones_del_nivel(transitions, current_frame=None, include_all_levels=False):
    """Transiciones utilizables: con cuadro previo, sin RESET, sin muerte ni cambio de nivel,
    y (por defecto) del nivel actual."""
    nivel = getattr(current_frame, "level", None)
    salida = []
    for i, t in enumerate(transitions or []):
        antes, despues = getattr(t, "before_frame", None), getattr(t, "after_frame", None)
        if antes is None or despues is None:
            continue
        nombre, _, _ = parse_action(getattr(t, "action", ""))
        if nombre in ("RESET", "UNDO"):  # dependen de la historia, no solo de la grilla
            continue
        res = getattr(t, "result", {}) or {}
        if res.get("automatic") or res.get("game_over") or res.get("level_completed") \
                or res.get("run_complete") or res.get("done"):
            continue
        if getattr(antes, "level", None) != getattr(despues, "level", None):
            continue
        if not include_all_levels and nivel is not None and getattr(antes, "level", None) != nivel:
            continue
        salida.append((i, t))
    return salida


def check_model(step, transitions, current_frame=None, *, include_all_levels=False,
                border=_BORDE, max_report=3, time_limit=8.0):
    """Contrasta `step(grid, action) -> grid` con las transiciones observadas.

    Devuelve un dict con:
      checked, exact (predicción idéntica), interior_ok (idéntica fuera de la franja de
      `border` celdas, donde suelen estar barras y contadores), y `mismatches`: las primeras
      transiciones donde falla, con la acción, el índice en `transitions` y celdas de ejemplo
      (r, c, real, predicho). `all_interior_ok` es True si el simulador explica todo el
      interior de lo observado: recién ahí conviene planificar con él.
    """
    reloj = _Reloj(time_limit)
    casos = _transiciones_del_nivel(transitions, current_frame, include_all_levels)
    informe = {"checked": 0, "exact": 0, "interior_ok": 0, "mismatches": [], "errors": 0,
               "available": len(casos), "by_family": {}}
    familias = informe["by_family"]
    for i, t in casos:
        if not reloj.queda():
            informe["note"] = f"time limit reached after {informe['checked']} transitions"
            break
        antes, real = _grilla(t.before_frame), _grilla(t.after_frame)
        informe["checked"] += 1
        fam = familias.setdefault(_familia(t.action), [0, 0])
        fam[1] += 1
        inicio_step = time.monotonic()
        try:
            copia = [fila[:] for fila in antes]
            predicha = _salida(copia, step(copia, t.action))
        except Exception as exc:  # un simulador roto es un desacierto, no un crash
            reloj.medir(inicio_step)
            informe["errors"] += 1
            if len(informe["mismatches"]) < max_report:
                informe["mismatches"].append({"index": i, "action": t.action,
                                              "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        reloj.medir(inicio_step)
        dif = _diferencias(real, predicha, border)
        if dif is None:
            if len(informe["mismatches"]) < max_report:
                informe["mismatches"].append({"index": i, "action": t.action,
                                              "error": "step returned a grid of the wrong shape/type"})
            continue
        interior, en_borde = dif
        if not interior:
            informe["interior_ok"] += 1
            fam[0] += 1
            if not en_borde:
                informe["exact"] += 1
                continue
        if interior and len(informe["mismatches"]) < max_report:
            faltan, inventa = _faltan_inventa(interior, antes)
            informe["mismatches"].append({
                "index": i, "action": t.action,
                "wrong_interior_cells": len(interior), "wrong_border_cells": len(en_borde),
                "examples_r_c_actual_predicted": interior[:_MAX_EJEMPLOS],
                "missed_changes": len(faltan), "invented_changes": len(inventa),
                "missed_example": faltan[0] if faltan else None,
                "invented_example": inventa[0] if inventa else None,
            })
    n = informe["checked"]
    informe["all_interior_ok"] = n > 0 and informe["interior_ok"] == n
    return informe


def _desplazar(g, borde, dr=1, dc=1):
    """Copia de la grilla con el interior corrido (dr, dc) celdas; lo que entra se rellena con
    el color más común del interior. El borde (barras) queda igual."""
    filas, cols = len(g), len(g[0])
    cuenta = {}
    for r in range(borde, filas - borde):
        for c in range(borde, cols - borde):
            cuenta[g[r][c]] = cuenta.get(g[r][c], 0) + 1
    fondo = max(cuenta, key=cuenta.get) if cuenta else None
    h = [fila[:] for fila in g]
    for r in range(borde, filas - borde):
        for c in range(borde, cols - borde):
            rr, cc = r - dr, c - dc
            h[r][c] = g[rr][cc] if borde <= rr < filas - borde and borde <= cc < cols - borde else fondo
    return h


def _accion_desplazada(action, dr=1, dc=1):
    nombre, r, c = parse_action(action)
    return mouse(r + dr, c + dc) if nombre == "MOUSE" else action


def shift_test(step, transitions, current_frame=None, *, include_all_levels=True, border=_BORDE,
               max_casos=2, time_limit=1.0):
    """¿`step` depende de coordenadas fijas? Toma transiciones con cambios lejos del borde, corre
    el tablero (y el click) una celda en diagonal y exige la misma predicción, corrida.
    Devuelve True (general), False (coordenadas fijas) o None (no hay casos o no alcanzó el tiempo).
    Idea de la revisión de Argon (2026-10-17)."""
    reloj = _Reloj(time_limit)
    probados = 0
    for _, t in _transiciones_del_nivel(transitions, current_frame, include_all_levels):
        if probados >= max_casos or not reloj.queda():
            break
        antes, despues = _grilla(t.before_frame), _grilla(t.after_frame)
        dif = _diferencias(despues, antes, border + 2)
        if dif is None or not dif[0]:
            continue
        filas, cols = len(antes), len(antes[0])
        # Los cambios en la franja exterior (HUD, barra de pasos) no cuentan; solo descartan el
        # caso los del anillo entre `border` y `border + 2`, que el corrimiento empujaría afuera.
        if any(border <= r < filas - border and border <= c < cols - border for r, c, _, _ in dif[1]):
            continue
        esperado = _desplazar(despues, border)
        inicio_step = time.monotonic()
        try:
            copia = _desplazar(antes, border)
            predicho = _salida(copia, step(copia, _accion_desplazada(t.action)))
        except Exception:
            return False
        finally:
            reloj.medir(inicio_step)
        probados += 1
        try:
            for r in range(border + 1, filas - border - 1):
                fp = predicho[r] if r < len(predicho) else None
                if fp is None or list(fp[border + 1:cols - border - 1]) != esperado[r][border + 1:cols - border - 1]:
                    return False
        except Exception:  # predicción con forma rara: no es general
            return False
    return True if probados else None


def auto_check(step, transitions, current_frame=None, *, time_limit=2.0):
    """Una línea para el modelo: cuánto de lo observado en el nivel explica su `step` retenido.
    Vacío si no hay transiciones o si algo falla (nunca lanza)."""
    try:
        # Un step que lee el tablero global en vez de su parámetro `grid` no sirve para simular
        # ni para otro nivel (mecanismo A de la revisión de Argon).
        globales = {"current_frame", "latest_frame", "previous_frame", "last_action_frame",
                    "history", "transitions", "last_transition"}
        leidas = sorted(globales & set(getattr(getattr(step, "__code__", None), "co_names", ()) or ()))
        aviso = (f" Warning: step reads the live global {leidas[0]}; it must compute everything from "
                 "its `grid` argument, or it cannot simulate other states." if leidas else "")
        texto = _auto_check_texto(step, transitions, current_frame, time_limit)
        if aviso:
            return (texto or "[auto-check of your retained step]") + aviso
        return texto
    except Exception:
        return ""


def _auto_check_texto(step, transitions, current_frame, time_limit):
    try:
        r = check_model(step, transitions, current_frame, max_report=1, time_limit=time_limit)
        n = r["checked"]
        if r.get("available", 0) < 2:  # nivel recién empezado: ¿sirve el step de los anteriores?
            previo = check_model(step, transitions, current_frame, include_all_levels=True,
                                 max_report=1, time_limit=time_limit)
            m = previo["checked"]
            if m >= 3 and previo["interior_ok"] == m and "note" not in previo:
                general = shift_test(step, transitions, current_frame, time_limit=time_limit / 2)
                if general is False:
                    return ("[auto-check of your retained step] new level: your step reproduces the "
                            "earlier levels but fails when the board is shifted by one cell. If the game "
                            "really uses absolute positions, ignore this; if it depends on where objects "
                            "are, it likely has fixed coordinates: locate objects by color/shape inside "
                            "`grid` before planning on this level.")
                if general:
                    return (f"[auto-check of your retained step] new level: your step reproduces all {m} "
                            "recorded transitions of the earlier levels and is position-independent. If this "
                            "level works the same way, skip re-exploring: plan_with_model(step, goal, actions) "
                            "right away.")
        if not n:
            return ""
        total = r.get("available", n)
        cortado = "note" in r
        texto = (f"[auto-check of your retained step] it reproduces {r['interior_ok']}/{n} recorded "
                 "transitions of this level (interior)"
                 + (f"; only {n} of {total} could be checked in the time budget." if cortado else "."))
        if r["mismatches"]:
            m = r["mismatches"][0]
            if "error" in m:
                texto += f" First miss: {m['action']} -> {m['error'][:120]}."
            else:
                ej = ", ".join(f"({a},{b}) actual {x!s} predicted {y!s}"
                               for a, b, x, y in m["examples_r_c_actual_predicted"][:2])
                texto += (f" First miss: {m['action']} (transition {m['index']}), "
                          f"{m['wrong_interior_cells']} cells wrong, e.g. {ej}.")
                partes = []
                if m.get("missed_example"):
                    a, b, x, y = m["missed_example"]
                    partes.append(f"{m['missed_changes']} real changes you did not predict, e.g. "
                                  f"({a},{b}) went {x!s}->{y!s}")
                if m.get("invented_example"):
                    a, b, x, y = m["invented_example"]
                    partes.append(f"{m['invented_changes']} changes you invented, e.g. ({a},{b}) "
                                  f"stayed {x!s} but you predicted {y!s}")
                if partes:
                    texto += " " + "; ".join(partes) + "."
        elif n >= 3 and not cortado:
            texto += (" It explains everything seen so far: plan_with_model(step, goal, actions) "
                      "can search for the shortest plan without spending actions.")
        fams = r.get("by_family") or {}
        if len(fams) > 1 and r["interior_ok"] < n:
            texto += " By action: " + ", ".join(f"{k} {ok}/{tot}" for k, (ok, tot) in sorted(fams.items())) + "."
        return texto
    except Exception:
        return ""


def plan(step, goal, start, actions, *, max_depth=60, max_nodes=4000, time_limit=8.0,
         border=_BORDE, cost=False):
    """Camino más corto, dentro del simulador `step`, desde `start` hasta un estado que
    cumpla `goal`.

    goal: función grid -> bool, o una grilla objetivo (se compara el interior).
    actions: etiquetas a probar en cada estado (p. ej. ["UP", "DOWN", "LEFT", "RIGHT"] o
    clicks hechos con `mouse(r, c)`).
    cost=True: `goal(grid)` devuelve un número (0 o menos = meta) y la búsqueda es mejor primero.
    Sin cost=True, el valor de `goal` se lee como verdad (1/0 sirven como True/False).
    Devuelve {"plan": [...], "length": n, "explored": k} o {"plan": None, "reason": ...}.
    Los estados se comparan por su interior (fuera de la franja `border`), para que una barra
    de pasos que se achica no impida reconocer estados repetidos.
    """
    reloj = _Reloj(time_limit)
    max_nodes = min(int(max_nodes), _MAX_NODOS)
    if start is None:
        return {"plan": None, "reason": "no start frame"}
    try:
        inicio = _grilla(start)
    except Exception:
        return {"plan": None, "reason": "start board is not a valid grid"}
    acciones = [str(a) for a in (actions or [])]
    if not acciones:
        return {"plan": None, "reason": "no actions given"}

    def interior(g):
        filas = len(g)
        return tuple(tuple(f[border:len(f) - border]) for f in g[border:filas - border])

    if callable(goal):
        es_meta = goal
    else:
        try:
            objetivo = interior(_grilla(goal))
            hash(objetivo)
        except Exception:
            return {"plan": None, "reason": "goal board is not a valid grid"}
        es_meta = lambda g: interior(g) == objetivo  # noqa: E731

    def cumple(g):
        try:
            return bool(es_meta(g))
        except Exception:
            return False

    if cost:
        if not callable(goal):
            return {"plan": None, "reason": "cost=True needs goal to be a function grid -> number"}
        costo_inicio = _numero(es_meta, inicio)
        if costo_inicio is None:
            return {"plan": None, "reason": "cost=True: goal(start) did not return a finite number"}
        return _plan_por_costo(step, es_meta, inicio, acciones, interior, reloj, max_depth,
                               max_nodes, costo_inicio)
    try:
        valor_inicio = es_meta(inicio)
    except Exception:
        valor_inicio = False
    if (isinstance(valor_inicio, (int, float)) and not isinstance(valor_inicio, bool)
            and valor_inicio not in (0, 1)):
        return {"plan": None, "reason": f"goal returned {valor_inicio!r}, not True/False; if it is a "
                "distance to the goal, call plan_with_model(step, goal, actions, cost=True)"}
    if cumple(inicio):
        return {"plan": [], "length": 0, "explored": 0}
    try:
        vistos = {interior(inicio)}
    except Exception:
        return {"plan": None, "reason": "start board is not a valid grid"}
    frontera = [(inicio, [])]
    explorados = errores = 0
    for prof in range(max_depth):
        siguiente = []
        for g, camino in frontera:
            for a in acciones:
                if not reloj.queda():
                    return {"plan": None, "reason": "time limit", "explored": explorados,
                            "depth_reached": prof, "step_errors": errores}
                inicio_step = time.monotonic()
                try:
                    copia = [f[:] for f in g]
                    h = _salida(copia, step(copia, a))
                    h = [list(f) for f in h]
                    k = interior(h)
                    hash(k)
                except Exception:  # step roto o celdas no hasheables: error del simulador
                    errores += 1
                    continue
                finally:
                    reloj.medir(inicio_step)
                if k in vistos:
                    continue
                vistos.add(k)
                explorados += 1
                nuevo = camino + [a]
                if cumple(h):
                    return {"plan": nuevo, "length": len(nuevo), "explored": explorados,
                            "step_errors": errores}
                if explorados >= max_nodes:
                    return {"plan": None, "reason": "node limit", "explored": explorados,
                            "depth_reached": prof + 1, "step_errors": errores}
                siguiente.append((h, nuevo))
        if not siguiente:
            return {"plan": None, "reason": "goal unreachable in the simulator",
                    "explored": explorados, "step_errors": errores}
        frontera = siguiente
    return {"plan": None, "reason": "depth limit", "explored": explorados, "step_errors": errores}


def _numero(f, g):
    """f(g) como float finito; None si falla, no es número o es NaN/infinito."""
    try:
        v = f(g)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _plan_por_costo(step, costo, inicio, acciones, interior, reloj, max_depth, max_nodes,
                    costo_inicio):
    """Mejor primero sobre `costo(grid) -> número` (0 o menos = meta; menor es mejor).
    Si no llega a 0 devuelve igual el camino al estado de menor costo encontrado, para que
    el modelo pueda avanzar aunque la meta quede lejos (sugerencia de la revisión de Argon)."""
    import heapq

    def valor(g):
        return _numero(costo, g)

    if costo_inicio <= 0:
        return {"plan": [], "length": 0, "explored": 0, "cost": costo_inicio}
    try:
        # profundidad mínima conocida por estado: mejor-primero no llega primero por el camino
        # más corto, así que un estado se reabre si aparece una ruta más corta (revisión de Astra)
        vistos = {interior(inicio): 0}
    except Exception:
        return {"plan": None, "reason": "start board is not a valid grid"}
    frontera = [(costo_inicio, 0, 0, inicio, [])]
    mejor = (costo_inicio, [])
    explorados = errores = orden = 0

    def parcial(motivo):
        if not mejor[1]:  # ningún estado alcanzable baja el costo: mínimo local, decirlo
            motivo += "; no reachable state lowers the cost (local minimum): explore or change the cost"
        return {"plan": None, "reason": motivo, "explored": explorados, "step_errors": errores,
                "start_cost": costo_inicio, "best_cost": mejor[0], "best_plan": mejor[1],
                "best_length": len(mejor[1])}

    while frontera:
        _, _, prof, g, camino = heapq.heappop(frontera)
        if prof >= max_depth:
            continue
        for a in acciones:
            if not reloj.queda():
                return parcial("time limit")
            inicio_step = time.monotonic()
            try:
                copia = [f[:] for f in g]
                h = _salida(copia, step(copia, a))
                h = [list(f) for f in h]
                k = interior(h)
                hash(k)
            except Exception:
                errores += 1
                continue
            finally:
                reloj.medir(inicio_step)
            if k in vistos and vistos[k] <= prof + 1:
                continue
            nuevo_estado = k not in vistos
            vistos[k] = prof + 1
            if nuevo_estado:
                explorados += 1
            c = valor(h)
            if c is None:
                errores += 1
                continue
            nuevo = camino + [a]
            if c < mejor[0] or (c == mejor[0] and len(nuevo) < len(mejor[1])):
                mejor = (c, nuevo)
            if c <= 0:
                return {"plan": nuevo, "length": len(nuevo), "explored": explorados,
                        "step_errors": errores, "cost": c}
            if explorados >= max_nodes:
                return parcial("node limit")
            orden += 1
            heapq.heappush(frontera, (c, orden, prof + 1, h, nuevo))
    return parcial("cost 0 unreachable in the simulator")


# --- Modelo de estado: observe(grid) -> estado, advance(estado, action) -> estado ---------------
# Idea de la revisión del 18-oct: el modelo ya resume el tablero en un estado abstracto
# (posición del jugador, fichas, contadores) y razona sobre eso. Verificar ese estado campo por
# campo es la población de objetos vivos: cada campo vive o cae según lo observado.

def _congelar(x, prof=0):
    """Estado -> valor hasheable y comparable (dict/list/set -> tuplas ordenadas)."""
    if prof > 20:
        raise ValueError("state nested too deep")
    if isinstance(x, dict):
        return ("<d>",) + tuple(sorted(((_congelar(k, prof + 1), _congelar(v, prof + 1))
                                        for k, v in x.items()), key=repr))
    if isinstance(x, (set, frozenset)):
        return ("<s>",) + tuple(sorted((_congelar(v, prof + 1) for v in x), key=repr))
    if isinstance(x, (list, tuple)):
        return tuple(_congelar(v, prof + 1) for v in x)
    hash(x)
    return x


def _copia_estado(s):
    import copy
    return copy.deepcopy(s)


def _corto(v, n=60):
    t = repr(v)
    return t if len(t) <= n else t[:n - 3] + "..."


def check_state(observe, advance, transitions, current_frame=None, *, include_all_levels=False,
                border=_BORDE, time_limit=8.0):
    """Contrasta `advance(observe(antes), action)` con `observe(despues)` en cada transición.

    Si el estado es un dict, cada clave es un campo y se verifica por separado (`fields`:
    {campo: [aciertos, total]} y `field_misses` con el primer error de cada campo). Además cuenta
    los puntos ciegos (`blind`): transiciones donde el interior del tablero cambió pero
    `observe` devolvió el mismo estado, o sea que el estado no registra algo que pasó.
    """
    reloj = _Reloj(time_limit)
    casos = _transiciones_del_nivel(transitions, current_frame, include_all_levels)
    informe = {"checked": 0, "state_ok": 0, "available": len(casos), "fields": {},
               "field_misses": {}, "blind": 0, "blind_example": None, "errors": 0,
               "first_error": None, "changed": 0, "ambient_cells": 0,
               "changed_raw": 0, "blind_raw": 0, "blind_raw_example": None}
    ambientales = _celdas_ambientales(casos, border)
    informe["ambient_cells"] = len(ambientales)
    for i, t in casos:
        if not reloj.queda():
            informe["note"] = f"time limit reached after {informe['checked']} transitions"
            break
        inicio = time.monotonic()
        try:
            antes, despues = _grilla(t.before_frame), _grilla(t.after_frame)
            s0 = observe([f[:] for f in antes])
            c0 = _congelar(s0)
            s1 = observe([f[:] for f in despues])
            c1 = _congelar(s1)
            pred = advance(_copia_estado(s0), t.action)
            cp = _congelar(pred)
        except Exception as exc:
            reloj.medir(inicio)
            informe["checked"] += 1
            informe["errors"] += 1
            if informe["first_error"] is None:
                informe["first_error"] = {"index": i, "action": t.action,
                                          "error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        reloj.medir(inicio)
        informe["checked"] += 1
        if cp == c1:
            informe["state_ok"] += 1
        if isinstance(s1, dict) and isinstance(pred, dict):
            for k in set(s1) | set(pred):
                try:
                    clave = k if isinstance(k, str) else f"{k!r}:{type(k).__name__}"
                    bien = k in s1 and k in pred and _congelar(s1[k]) == _congelar(pred[k])
                except Exception:
                    clave, bien = repr(k), False
                campo = informe["fields"].setdefault(clave, [0, 0])
                campo[1] += 1
                if bien:
                    campo[0] += 1
                elif clave not in informe["field_misses"]:
                    informe["field_misses"][clave] = {
                        "index": i, "action": t.action,
                        "before": _corto(s0.get(k, "<missing>") if isinstance(s0, dict) else None),
                        "actual": _corto(s1.get(k, "<missing>")),
                        "predicted": _corto(pred.get(k, "<missing>"))}
        elif cp != c1 and "state" not in informe["field_misses"]:
            informe["field_misses"]["state"] = {"index": i, "action": t.action,
                                                "before": _corto(s0), "actual": _corto(s1),
                                                "predicted": _corto(pred)}
        dif = _diferencias(despues, antes, border)
        crudos = dif[0] if dif else []
        reales = [d for d in crudos if (d[0], d[1]) not in ambientales]
        if reales:
            informe["changed"] += 1
        # Sin filtrar: si el interior cambió y observe no, se cuenta aunque las celdas sean
        # "ambientales". El filtro no puede habilitar "explica todo" con un estado que nunca
        # cambió (revisión de Astra, 20-oct: observe() -> {} con un jugador que se mueve).
        if crudos:
            informe["changed_raw"] += 1
            if c0 == c1:
                informe["blind_raw"] += 1
                if informe["blind_raw_example"] is None:
                    r, c, nuevo, viejo = crudos[0]
                    informe["blind_raw_example"] = {"index": i, "action": t.action, "cell": (r, c),
                                                    "was": viejo, "became": nuevo,
                                                    "cells_changed": len(crudos)}
        if c0 == c1:
            if reales:
                dif = (reales, dif[1])
                informe["blind"] += 1
                if informe["blind_example"] is None:
                    r, c, nuevo, viejo = dif[0][0]
                    informe["blind_example"] = {"index": i, "action": t.action, "cell": (r, c),
                                                "was": viejo, "became": nuevo,
                                                "cells_changed": len(dif[0])}
    n = informe["checked"]
    # estado constante: el tablero cambió y observe nunca vio nada (p. ej. devuelve {})
    informe["constant"] = informe["changed_raw"] > 0 and informe["blind_raw"] == informe["changed_raw"]
    if informe["constant"] and informe["blind_example"] is None:
        informe["blind_example"] = informe["blind_raw_example"]
    informe["all_ok"] = n > 0 and informe["state_ok"] == n and not informe["constant"]
    return informe


def _celdas_ambientales(casos, border, muestra=12):
    """Celdas del interior que cambian en casi todas las transiciones recientes, con acciones
    distintas: relojes o animaciones, no mecánicas (sugerencia de Argon, 18-oct)."""
    try:
        ultimos = casos[-muestra:]
        acciones = {parse_action(t.action)[0] for _, t in ultimos}
        if len(ultimos) < 3 or len(acciones) < 2:
            return set()
        cuenta = {}
        for _, t in ultimos:
            dif = _diferencias(_grilla(t.after_frame), _grilla(t.before_frame), border)
            for r, c, _, _ in (dif[0] if dif else []):
                cuenta[(r, c)] = cuenta.get((r, c), 0) + 1
        umbral = 0.8 * len(ultimos)
        return {celda for celda, k in cuenta.items() if k >= umbral}
    except Exception:
        return set()


def _coordenada(v, filas, cols, border):
    return (isinstance(v, tuple) and len(v) == 2 and all(type(x) is int for x in v)
            and border <= v[0] < filas - border and border <= v[1] < cols - border)


def state_shift_test(observe, grid, *, border=_BORDE):
    """Campos de `observe` con coordenadas fijas: corre el interior una celda en diagonal y
    devuelve los campos (r, c) o colecciones de (r, c) que no se movieron o desaparecieron.
    Una llamada extra a observe. Idea de Argon (18-oct)."""
    try:
        g = _grilla(grid)
        filas, cols = len(g), len(g[0])
        s0 = observe([f[:] for f in g])
        if not isinstance(s0, dict) or not s0:
            return []
        s1 = observe(_desplazar(g, border))
        if not isinstance(s1, dict):
            return ["<state>"]
        def corrida(rc):
            r, c = rc[0] + 1, rc[1] + 1
            return (r, c) if r < filas - border and c < cols - border else None

        sospechosos = []
        for k, v0 in s0.items():
            v1 = s1.get(k)
            nombre = k if isinstance(k, str) else repr(k)
            if _coordenada(v0, filas, cols, border):
                esperado = corrida(v0)
                if esperado is not None and v1 != esperado:
                    sospechosos.append(nombre)
            elif isinstance(v0, (list, tuple, set, frozenset)) and v0 and all(
                    _coordenada(e, filas, cols, border) for e in v0):
                esperado = {corrida(e) for e in v0} - {None}
                try:
                    obtenido = set(v1) if isinstance(v1, (list, tuple, set, frozenset)) else None
                except Exception:
                    obtenido = None
                if esperado and obtenido != esperado:
                    sospechosos.append(nombre)
        return sospechosos
    except Exception:
        return []


def plan_state(observe, advance, goal, start, actions, *, max_depth=80, max_nodes=20000,
               time_limit=8.0, cost=False):
    """Camino más corto en el espacio de estados de `advance`, desde `observe(start)` hasta un
    estado con `goal(state)` verdadero. Mucho más barato que copiar grillas."""
    reloj = _Reloj(time_limit)
    max_nodes = min(int(max_nodes), 200000)
    acciones = [str(a) for a in (actions or [])]
    if not acciones:
        return {"plan": None, "reason": "no actions given"}
    if start is None:
        return {"plan": None, "reason": "no start frame"}
    try:
        s0 = observe(_grilla(start))
        k0 = _congelar(s0)
    except Exception as exc:
        return {"plan": None, "reason": f"observe failed on the current board: {type(exc).__name__}: {exc}"[:200]}

    errores_meta = [0]
    raro = [None]

    def _distancia(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (0, 1)

    def cumple(s):
        # None -> False; 1/0 -> verdad; otro número es una distancia (error de uso sin
        # cost=True); el resto (tuplas, listas, strings) por su verdad de Python.
        try:
            v = goal(s)
        except Exception:
            errores_meta[0] += 1
            return False
        if v is None:
            return False
        if _distancia(v):
            if raro[0] is None:
                raro[0] = _corto(v)
            return False
        return bool(v)

    try:
        v0 = goal(s0)
    except Exception as exc:
        return {"plan": None, "reason": f"goal failed on the current state: {type(exc).__name__}: {exc}"[:200]}
    if cost:
        return _plan_estado_por_costo(advance, goal, s0, k0, acciones, reloj, max_depth, max_nodes)
    if _distancia(v0):
        return {"plan": None, "reason": f"goal returned {_corto(v0)}, not True/False; if it is a "
                "distance to the goal, call plan_with_state(observe, advance, goal, actions, cost=True)"}
    if cumple(s0):
        return {"plan": [], "length": 0, "explored": 0}
    vistos = {k0}
    frontera = [(s0, [])]
    explorados = errores = 0
    for prof in range(max_depth):
        siguiente = []
        for s, camino in frontera:
            for a in acciones:
                if not reloj.queda():
                    return {"plan": None, "reason": "time limit", "explored": explorados,
                            "depth_reached": prof, "advance_errors": errores, "goal_errors": errores_meta[0]}
                inicio = time.monotonic()
                try:
                    h = advance(_copia_estado(s), a)
                    k = _congelar(h)
                except Exception:
                    errores += 1
                    continue
                finally:
                    reloj.medir(inicio)
                if k in vistos:
                    continue
                vistos.add(k)
                explorados += 1
                nuevo = camino + [a]
                if cumple(h):
                    return {"plan": nuevo, "length": len(nuevo), "explored": explorados,
                            "advance_errors": errores, "goal_errors": errores_meta[0]}
                if raro[0] is not None:
                    return {"plan": None, "reason": f"goal returned {raro[0]}, not True/False; for a "
                            "distance use cost=True", "explored": explorados}
                if explorados >= max_nodes:
                    return {"plan": None, "reason": "node limit", "explored": explorados,
                            "depth_reached": prof + 1, "advance_errors": errores, "goal_errors": errores_meta[0]}
                siguiente.append((h, nuevo))
        if not siguiente:
            return {"plan": None, "reason": "goal unreachable with this advance",
                    "explored": explorados, "advance_errors": errores, "goal_errors": errores_meta[0]}
        frontera = siguiente
    return {"plan": None, "reason": "depth limit", "explored": explorados, "advance_errors": errores, "goal_errors": errores_meta[0]}


def _plan_estado_por_costo(advance, costo, s0, k0, acciones, reloj, max_depth, max_nodes):
    """Mejor primero sobre `costo(state) -> número` (0 o menos = meta). Si no llega, devuelve el
    camino al estado de menor costo (best_plan), como plan_with_model con cost=True."""
    import heapq
    c0 = _numero(costo, s0)
    if c0 is None:
        return {"plan": None, "reason": "cost=True: goal(state) did not return a finite number"}
    if c0 <= 0:
        return {"plan": [], "length": 0, "explored": 0, "cost": c0}
    vistos = {k0: 0}  # profundidad mínima por estado (ver _plan_por_costo)
    frontera = [(c0, 0, 0, s0, [])]
    mejor = (c0, [])
    explorados = errores = orden = 0

    def parcial(motivo):
        return {"plan": None, "reason": motivo, "explored": explorados, "advance_errors": errores,
                "start_cost": c0, "best_cost": mejor[0], "best_plan": mejor[1],
                "best_length": len(mejor[1])}

    while frontera:
        _, _, prof, s, camino = heapq.heappop(frontera)
        if prof >= max_depth:
            continue
        for a in acciones:
            if not reloj.queda():
                return parcial("time limit")
            inicio = time.monotonic()
            try:
                h = advance(_copia_estado(s), a)
                k = _congelar(h)
            except Exception:
                errores += 1
                continue
            finally:
                reloj.medir(inicio)
            if k in vistos and vistos[k] <= prof + 1:
                continue
            nuevo_estado = k not in vistos
            vistos[k] = prof + 1
            if nuevo_estado:
                explorados += 1
            c = _numero(costo, h)
            if c is None:
                errores += 1
                continue
            nuevo = camino + [a]
            if c < mejor[0] or (c == mejor[0] and len(nuevo) < len(mejor[1])):
                mejor = (c, nuevo)
            if c <= 0:
                return {"plan": nuevo, "length": len(nuevo), "explored": explorados,
                        "advance_errors": errores, "cost": c}
            if explorados >= max_nodes:
                return parcial("node limit")
            orden += 1
            heapq.heappush(frontera, (c, orden, prof + 1, h, nuevo))
    return parcial("cost 0 unreachable with this advance")


def _vacio(v):
    return v is None or (isinstance(v, (list, tuple, set, frozenset, dict, str)) and len(v) == 0)


def _nivel_nuevo_estado(observe, advance, transitions, current_frame, time_limit, cab):
    """Al empezar un nivel: ¿el modelo de los niveles anteriores sirve y observe lee este tablero?"""
    previo = check_state(observe, advance, transitions, current_frame, include_all_levels=True,
                         time_limit=time_limit)
    m = previo["checked"]
    if m < 3 or not previo["all_ok"] or "note" in previo or current_frame is None:
        return ""
    base = f"{cab} new level: your advance() predicted all {m} recorded transitions of the earlier levels"
    try:
        s_nuevo = observe(_grilla(current_frame))
    except Exception as exc:
        return (f"{base}, but observe() crashed on this new board ({type(exc).__name__}: "
                f"{str(exc)[:80]}). Fix only observe() to read the new layout; keep advance().")
    try:
        casos = _transiciones_del_nivel(transitions, None, True)
        s_viejo = observe(_grilla(casos[-1][1].after_frame)) if casos else None
    except Exception:
        s_viejo = None
    if isinstance(s_viejo, dict) and isinstance(s_nuevo, dict):
        perdidos = [k if isinstance(k, str) else repr(k) for k, v in s_viejo.items()
                    if not _vacio(v) and _vacio(s_nuevo.get(k))]
        if perdidos:
            return (f"{base}, but on this new board observe() returns nothing for "
                    f"{', '.join(perdidos[:3])}. Update observe() for the new colors/shapes; keep advance().")
    fijos = state_shift_test(observe, current_frame)
    if fijos:
        return (f"{base}, but field {', '.join(fijos[:3])} did not move when the board was shifted by "
                "one cell: it looks like fixed coordinates. Find it by color/shape before planning here.")
    return (f"{base} and observe() reads this board. If this level works the same way, plan with "
            "plan_with_state(observe, advance, goal, actions) instead of re-exploring.")


def auto_check_state(observe, advance, transitions, current_frame=None, *, time_limit=2.0):
    """Una línea para el modelo sobre su observe/advance retenidos. Vacío si no hay
    transiciones o si algo falla (nunca lanza)."""
    try:
        r = check_state(observe, advance, transitions, current_frame, time_limit=time_limit)
        n = r["checked"]
        cab = "[auto-check of your observe/advance model]"
        if not n:
            if r.get("available", 0) == 0:
                return _nivel_nuevo_estado(observe, advance, transitions, current_frame,
                                           time_limit, cab)
            return ""
        total = r.get("available", n)
        corte = f" (only {n} of {total} checked in the time budget)" if "note" in r else ""
        if r["constant"]:
            b = r["blind_example"]
            return (f"{cab} observe() returned the same state before and after every board change "
                    f"in this level{corte}: it captures nothing yet. E.g. transition {b['index']} "
                    f"({b['action']}): {b['cells_changed']} cells changed, ({b['cell'][0]},{b['cell'][1]}) "
                    f"{b['was']!r}->{b['became']!r}. Make observe() find the objects that move.")
        texto = f"{cab} it predicts the next state in {r['state_ok']}/{n} recorded transitions of this level{corte}."
        if r["first_error"]:
            e = r["first_error"]
            texto += f" Error at transition {e['index']} ({e['action']}): {e['error'][:120]}."
        campos = r["fields"]
        if campos:
            malos = sorted((k for k, (ok, tot) in campos.items() if ok < tot),
                           key=lambda k: campos[k][0] - campos[k][1])
            buenos = [k for k, (ok, tot) in campos.items() if ok == tot]
            if buenos:
                texto += " Correct fields: " + ", ".join(
                    f"{k} {campos[k][0]}/{campos[k][1]}" for k in sorted(buenos)[:6]) + "."
            for k in malos[:2]:
                fm = r["field_misses"].get(k, {})
                texto += (f" Field {k} {campos[k][0]}/{campos[k][1]}: first wrong at transition "
                          f"{fm.get('index')} ({fm.get('action')}), was {fm.get('before')}, became "
                          f"{fm.get('actual')}, you predicted {fm.get('predicted')}.")
        elif "state" in r["field_misses"]:
            fm = r["field_misses"]["state"]
            texto += (f" First wrong at transition {fm['index']} ({fm['action']}): actual "
                      f"{fm['actual']}, predicted {fm['predicted']}. Return a dict of named fields "
                      "to see which part is wrong.")
        if r["blind"]:
            b = r["blind_example"]
            texto += (f" Blind spot: in {r['blind']} transitions the board changed but observe() "
                      f"returned the same state, e.g. transition {b['index']} ({b['action']}): "
                      f"{b['cells_changed']} cells changed, ({b['cell'][0]},{b['cell'][1]}) "
                      f"{b['was']!r}->{b['became']!r}. If that change matters, your state is missing it.")
        if r["all_ok"] and not r["blind"] and n >= 3 and "note" not in r:
            fijos = state_shift_test(observe, current_frame) if current_frame is not None else []
            if fijos:
                texto += (f" Warning: field {', '.join(fijos[:3])} did not move when the board was "
                          "shifted by one cell: fixed coordinates will break on the next level; find "
                          "it by color/shape.")
            texto += (" It explains everything seen so far: plan_with_state(observe, advance, goal, "
                      "actions) searches over states without spending actions (advance() never sees "
                      "the board, so walls or obstacles must be in the state).")
        return texto
    except Exception:
        return ""


def nombres_que_se_pierden(funciones, codigo):
    """Nombres que las funciones del modelo leen y que el snippet asignó al nivel superior
    (DIRS = ..., P = ...). El harness retiene solo las `def`: esas constantes no existen en el
    snippet siguiente y la función explota con NameError (medido en v3: BARS, P, rows).
    Sugerencia de Argon (18-oct)."""
    try:
        import ast
        asignados = set()
        for nodo in ast.parse(str(codigo or "")).body:
            objetivos = []
            if isinstance(nodo, ast.Assign):
                objetivos = nodo.targets
            elif isinstance(nodo, (ast.AnnAssign, ast.AugAssign)):
                objetivos = [nodo.target]
            for o in objetivos:
                for n in ast.walk(o):
                    if isinstance(n, ast.Name):
                        asignados.add(n.id)
        if not asignados:
            return []

        import dis

        def nombres(code, prof=0):
            # solo lecturas de variables globales: co_names también trae atributos (s.x)
            vistos = set()
            try:
                for ins in dis.get_instructions(code):
                    if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                        vistos.add(ins.argval)
            except Exception:
                vistos = set(getattr(code, "co_names", ()) or ())
            if prof < 5:
                for c in getattr(code, "co_consts", ()) or ():
                    if hasattr(c, "co_names"):
                        vistos |= nombres(c, prof + 1)
            return vistos

        perdidos = []
        for f in funciones:
            for n in sorted(nombres(getattr(f, "__code__", None)) & asignados):
                if n not in perdidos:
                    perdidos.append(n)
        return perdidos
    except Exception:
        return []

