# SPDX-License-Identifier: MIT-0
"""Predictor del motor de Objetos Vivos en Python puro (sin numpy), para el sandbox del modelo.

El aprendizaje corre del lado del harness con ``vivos/mundo.py`` (numpy) y llega acá ya
serializado por ``vivos/motor_llm.py``. Este módulo solo aplica esas reglas a un tablero:
es una copia fiel de ``ModeloMundo.predecir`` (memoria exacta, sprite rígido, parche
relativo) sobre listas de caracteres, verificada contra la versión numpy en
``tests/test_motor.py``.

Grillas: lista de filas, cada fila una lista (o cadena) de caracteres de color, igual que
``current_frame.ascii`` partido en líneas. Internamente los colores son ``ord(caracter)``.
Coordenadas como en ``mundo.py``: (x, y) = (columna, fila).
"""

import hashlib
import re

_MOUSE = re.compile(r"^\s*MOUSE\s*\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)\s*$", re.I)


def accion_motor(action):
    """'MOUSE(row=R, col=C)' -> 'MOUSE@C,R' (el formato de mundo.py); el resto en mayúsculas."""
    s = str(action).strip()
    m = _MOUSE.match(s)
    if m:
        return f"MOUSE@{int(m.group(2))},{int(m.group(1))}"
    return s.upper()


def a_enteros(grid):
    return [[ord(c) for c in fila] for fila in grid]


def a_letras(g):
    return [[chr(v) for v in fila] for fila in g]


def interior(g):
    alto, ancho = len(g), len(g[0])
    y0, y1, x0, x1 = 0, alto, 0, ancho
    borde = g[0][0]
    while y1 - y0 > 1 and all(g[y0][x] == borde for x in range(x0, x1)):
        y0 += 1
    while y1 - y0 > 1 and all(g[y1 - 1][x] == borde for x in range(x0, x1)):
        y1 -= 1
    while x1 - x0 > 1 and all(g[y][x0] == borde for y in range(y0, y1)):
        x0 += 1
    while x1 - x0 > 1 and all(g[y][x1 - 1] == borde for y in range(y0, y1)):
        x1 -= 1
    centrado = abs(y0 - (alto - y1)) <= 1 and abs(x0 - (ancho - x1)) <= 1
    if not centrado or any(g[y][x] == borde for y in range(y0, y1) for x in range(x0, x1)):
        return 0, alto, 0, ancho
    return y0, y1, x0, x1


def fondo(g):
    y0, y1, x0, x1 = interior(g)
    cuenta = {}
    for y in range(y0, y1):
        for x in range(x0, x1):
            v = g[y][x]
            cuenta[v] = cuenta.get(v, 0) + 1
    # np.unique ordena los valores: ante empate gana el menor
    return min(cuenta, key=lambda v: (-cuenta[v], v))


def huella(g, mascara):
    """Igual criterio que mundo.huella: celdas enmascaradas no cuentan."""
    h = hashlib.sha1()
    for y, fila in enumerate(g):
        h.update(bytes((0 if (x, y) in mascara else (v & 0xFF)) for x, v in enumerate(fila)))
        h.update(bytes((1 if (x, y) in mascara else 0) for x in range(len(fila))))
    return h.hexdigest()


def objeto_en(g, mascara, x, y, bg):
    """(color, ancho-1, alto-1, x0, y0) del componente 4-conexo que contiene (x, y), o None."""
    alto, ancho = len(g), len(g[0])
    if not (0 <= x < ancho and 0 <= y < alto) or (x, y) in mascara or g[y][x] == bg:
        return None
    color = g[y][x]
    visto = {(x, y)}
    pila = [(x, y)]
    while pila:
        cx, cy = pila.pop()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (0 <= nx < ancho and 0 <= ny < alto and (nx, ny) not in visto
                    and (nx, ny) not in mascara and g[ny][nx] == color):
                visto.add((nx, ny))
                pila.append((nx, ny))
    xs = [c[0] for c in visto]
    ys = [c[1] for c in visto]
    return color, max(xs) - min(xs), max(ys) - min(ys), min(xs), min(ys)


def disparador(accion, g, mascara, bg):
    if "@" not in accion:
        return accion, None
    base = accion.split("@")[0]
    x, y = (int(v) for v in accion.split("@")[1].split(","))
    u = objeto_en(g, mascara, x, y, bg)
    if u is None:
        return f"{base}@fondo", (x, y)
    c, w, h, x0, y0 = u
    return f"{base}@{c},{w},{h}", (x0, y0)


def _ubicar(sp, g):
    dibujo = sp["dibujo"]
    if not dibujo:
        return None
    alto, ancho = len(g), len(g[0])
    rx, ry, c0 = dibujo[0]
    mejores = []
    for y in range(alto):
        fila = g[y]
        for x in range(ancho):
            if fila[x] != c0:
                continue
            ox, oy = x - rx, y - ry
            ok = True
            for dx, dy, c in dibujo:
                X, Y = ox + dx, oy + dy
                if not (0 <= X < ancho and 0 <= Y < alto) or g[Y][X] != c:
                    ok = False
                    break
            if ok:
                mejores.append((ox, oy))
    if not mejores:
        return None
    if sp["ultima"] is None:
        return mejores[0]
    lx, ly = sp["ultima"]
    return min(mejores, key=lambda p: abs(p[0] - lx) + abs(p[1] - ly))


def _bloqueado(sp, g, esq, bg):
    alto, ancho = len(g), len(g[0])
    ox, oy = esq
    px, py = sp["paso"]
    propias = {(ox + dx, oy + dy) for dx, dy, _ in sp["dibujo"]}
    for dx, dy, _ in sp["dibujo"]:
        X, Y = ox + dx + px, oy + dy + py
        if not (0 <= X < ancho and 0 <= Y < alto):
            return True
        if (X, Y) in propias:
            continue
        c = g[Y][X]
        if c != bg and sp["choca"].get(c, 0) > 0 and sp["pasa"].get(c, 0) == 0:
            return True
    return False


def _mover(M, g, sp, esq, bg):
    out = [list(f) for f in g]
    alto, ancho = len(g), len(g[0])
    ox, oy = esq
    px, py = sp["paso"]
    for dx, dy, _ in sp["dibujo"]:
        x, y = ox + dx, oy + dy
        out[y][x] = M["suelo"].get((x, y), bg)
    for dx, dy, c in sp["dibujo"]:
        X, Y = ox + dx + px, oy + dy + py
        if 0 <= X < ancho and 0 <= Y < alto:
            out[Y][X] = c
    return out


def _aplica(p, g, ancla):
    alto, ancho = len(g), len(g[0])
    ax, ay = ancla
    for dx, dy, antes, _ in p["celdas"]:
        X, Y = ax + dx, ay + dy
        if not (0 <= X < ancho and 0 <= Y < alto) or g[Y][X] != antes:
            return False
    return True


def _aplicar(p, g, ancla):
    out = [list(f) for f in g]
    ax, ay = ancla
    for dx, dy, _, despues in p["celdas"]:
        out[ay + dy][ax + dx] = despues
    return out


def preparar(datos):
    """Del JSON que manda el harness a estructuras de consulta (claves enteras, tuplas, sets)."""
    sprites = {}
    for k, sp in (datos.get("sprites") or {}).items():
        sprites[k] = {
            "dibujo": [tuple(int(v) for v in d) for d in sp["dibujo"]],
            "paso": tuple(int(v) for v in sp["paso"]),
            "choca": {int(c): int(n) for c, n in sp["choca"].items()},
            "pasa": {int(c): int(n) for c, n in sp["pasa"].items()},
            "ultima": tuple(int(v) for v in sp["ultima"]) if sp.get("ultima") is not None else None,
            "confiable": bool(sp["confiable"]),
        }
    parches = {}
    for k, lista in (datos.get("parches") or {}).items():
        ps = [{"celdas": [tuple(int(v) for v in c) for c in p["celdas"]], "puntaje": int(p["puntaje"]),
               "confiable": bool(p["confiable"])} for p in lista]
        parches[k] = sorted(ps, key=lambda p: -p["puntaje"])  # mismo orden estable que mundo.predecir
    return {
        "mascara": {(int(x), int(y)) for x, y in datos.get("mascara") or []},
        "sprites": sprites,
        "parches": parches,
        "suelo": {(int(x), int(y)): int(c) for x, y, c in datos.get("suelo") or []},
        "memoria": {k: [tuple(int(v) for v in c) for c in d] for k, d in (datos.get("memoria") or {}).items()},
    }


def predecir(M, g, accion):
    """ModeloMundo.predecir sobre enteros. `accion` en formato de mundo.py."""
    mascara = M["mascara"]
    visto = M["memoria"].get(huella(g, mascara) + "|" + accion)
    if visto is not None:
        out = [list(f) for f in g]
        for x, y, v in visto:
            out[y][x] = v
        return out
    bg = fondo(g)
    clave, ancla = disparador(accion, g, mascara, bg)
    salida = g
    sp = M["sprites"].get(accion) if ancla is None else None
    if sp is not None and sp["confiable"]:
        esq = _ubicar(sp, g)
        if esq is not None and not _bloqueado(sp, g, esq, bg):
            salida = _mover(M, g, sp, esq, bg)
    a = ancla if ancla is not None else (0, 0)
    for p in M["parches"].get(clave, []):
        if p["confiable"] and _aplica(p, salida, a):
            return _aplicar(p, salida, a)
    return [list(f) for f in salida]


def engine_step(M, grid, action):
    """Para el modelo: tablero de letras + acción del harness -> tablero de letras predicho."""
    g = a_enteros(grid)
    return a_letras(predecir(M, g, accion_motor(action)))
