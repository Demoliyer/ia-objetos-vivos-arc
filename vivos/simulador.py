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
               "available": len(casos)}
    for i, t in casos:
        if not reloj.queda():
            informe["note"] = f"time limit reached after {informe['checked']} transitions"
            break
        antes, real = _grilla(t.before_frame), _grilla(t.after_frame)
        informe["checked"] += 1
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
            if not en_borde:
                informe["exact"] += 1
                continue
        if interior and len(informe["mismatches"]) < max_report:
            informe["mismatches"].append({
                "index": i, "action": t.action,
                "wrong_interior_cells": len(interior), "wrong_border_cells": len(en_borde),
                "examples_r_c_actual_predicted": interior[:_MAX_EJEMPLOS],
            })
    n = informe["checked"]
    informe["all_interior_ok"] = n > 0 and informe["interior_ok"] == n
    return informe


def auto_check(step, transitions, current_frame=None, *, time_limit=2.0):
    """Una línea para el modelo: cuánto de lo observado en el nivel explica su `step` retenido.
    Vacío si no hay transiciones o si algo falla (nunca lanza)."""
    try:
        r = check_model(step, transitions, current_frame, max_report=1, time_limit=time_limit)
        n = r["checked"]
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
        elif n >= 3 and not cortado:
            texto += (" It explains everything seen so far: plan_with_model(step, goal, actions) "
                      "can search for the shortest plan without spending actions.")
        return texto
    except Exception:
        return ""


def plan(step, goal, start, actions, *, max_depth=60, max_nodes=4000, time_limit=8.0,
         border=_BORDE):
    """Camino más corto, dentro del simulador `step`, desde `start` hasta un estado que
    cumpla `goal`.

    goal: función grid -> bool, o una grilla objetivo (se compara el interior).
    actions: etiquetas a probar en cada estado (p. ej. ["UP", "DOWN", "LEFT", "RIGHT"] o
    clicks hechos con `mouse(r, c)`).
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
