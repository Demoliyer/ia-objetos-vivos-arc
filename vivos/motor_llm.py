# SPDX-License-Identifier: MIT-0
"""Motor de Objetos Vivos junto al LLM (candidata M): aprende solo, sin gastar tokens ni
acciones, las reglas del nivel actual a partir de lo que ya jugó el modelo, y si su regla
predice lo observado la entrega serializada al sandbox, donde ``motor_puro.engine_step`` la
aplica.

Aprendizaje igual al medido en ``experimentos/reglas/medir_motor.py`` (20-oct-2026): en línea,
predice cada transición antes de verla y después la aprende; regla completa = desde la 4.ª
transición con cambio, al menos 3 medidas y >= 90 % de aciertos exactos fuera de la máscara
(franja de 4 celdas + banda de relojes).

Corre del lado del harness (numpy disponible). Nunca lanza: ante cualquier problema devuelve
None y el modelo juega como F.
"""

from __future__ import annotations

import numpy as np

try:  # en el paquete
    from .motor_puro import accion_motor, huella as huella_pura
    from .mundo import ModeloMundo, huella
    from .percepcion import DetectorDeRuido
except ImportError:  # dentro del harness: los módulos se cargan en el mismo espacio de nombres
    pass

FRANJA = 4
CALENTAMIENTO = 3
MINIMO = 3
UMBRAL = 0.9
MAX_MEMORIA = 400  # transiciones guardadas en la memoria exacta que se manda al sandbox
MAX_TRANS = 150  # últimas transiciones del nivel que se aprenden en cada snippet (~3 ms cada una)


def banda_de_relojes(ruido) -> np.ndarray:
    """La máscara del detector más la banda entera de cada reloj (copia de evaluar_mundo.py)."""
    m = ruido.mascara.copy()
    for c in ruido.relojes:
        z = ruido._zona(c)
        if not z.any():
            continue
        ys, xs = np.nonzero(z)
        if xs.max() - xs.min() >= ys.max() - ys.min():
            m[ys.min(): ys.max() + 1, :] = True
        else:
            m[:, xs.min(): xs.max() + 1] = True
    return m


def _grilla(frame):
    if not isinstance(frame, dict):
        return None
    asc = frame.get("ascii")
    if not isinstance(asc, str) or not asc:
        return None
    filas = [f for f in asc.split("\n") if f]
    if not filas or any(len(f) != len(filas[0]) for f in filas):
        return None
    return np.array([[ord(c) for c in f] for f in filas], dtype=np.int16)


_FIN = ("automatic", "game_over", "level_completed", "run_complete", "done")


def transiciones(history, nivel, maximo=None):
    """(antes, accion_motor, despues) del nivel `nivel`, con el mismo filtro que check_model.
    Recorre la historia desde el final y convierte tableros solo hasta juntar `maximo` transiciones
    válidas (auditoría de Astra, 27-oct: antes se convertía toda la historia y después se recortaba)."""
    h = history if isinstance(history, list) else list(history or [])
    out = []
    for i in range(len(h) - 1, 0, -1):
        if maximo is not None and len(out) >= maximo:
            break
        e, previo = h[i], h[i - 1]
        if not isinstance(e, dict) or not isinstance(previo, dict):
            continue
        accion = str(e.get("action") or "").strip()
        if not accion:
            continue
        nombre = accion.split("(")[0].strip().upper()
        if nombre in ("RESET", "UNDO"):
            continue
        res = e.get("result") or {}
        if isinstance(res, dict) and any(res.get(k) for k in _FIN):
            continue
        a, d = previo.get("frame"), e.get("frame")
        if not isinstance(a, dict) or not isinstance(d, dict):
            continue
        if a.get("level") != d.get("level") or a.get("level") != nivel:
            continue
        ga, gd = _grilla(a), _grilla(d)
        if ga is None or gd is None or ga.shape != gd.shape:
            continue
        out.append((ga, accion_motor(accion), gd))
    out.reverse()
    return out


def aprender(trans):
    """Aprendizaje en línea. Devuelve (modelo, ruido, aciertos_tardios, medidas_tardias)."""
    if not trans:
        return None
    alto, ancho = trans[0][0].shape
    borde = np.ones((alto, ancho), dtype=bool)
    borde[FRANJA:-FRANJA, FRANJA:-FRANJA] = False
    ruido, modelo = DetectorDeRuido(), ModeloMundo()
    aciertos = []
    for antes, accion, despues in trans:
        mascara = banda_de_relojes(ruido) | borde
        fuera = ~mascara
        # Solo cuentan los pares (estado, acción) no vistos: un acierto de la memoria exacta no
        # prueba una mecánica (auditoría de Astra, 27-oct; medido: ft09 36 -> 36 niveles, tu93 4 -> 0).
        if not np.array_equal(antes[fuera], despues[fuera]) and (huella(antes, mascara), accion) not in modelo.memoria:
            pred = modelo.predecir(antes, accion, mascara)
            aciertos.append(bool(np.array_equal(pred[fuera], despues[fuera])))
        ruido.observar(antes, despues, accion)
        modelo.observar(antes, accion, despues, banda_de_relojes(ruido) | borde)
    tarde = aciertos[CALENTAMIENTO:]
    return modelo, banda_de_relojes(ruido) | borde, sum(tarde), len(tarde)


def serializar(modelo, mascara, trans):
    """Reglas a JSON para motor_puro.preparar. La memoria exacta se rearma con la máscara final."""
    ys, xs = np.nonzero(mascara)
    celdas_masc = {(int(x), int(y)) for x, y in zip(xs, ys)}
    sprites = {}
    for k, sp in modelo.sprites.items():
        sprites[k] = {
            "dibujo": [[int(dx), int(dy), int(c)] for (dx, dy), c in sp.dibujo.items()],
            "paso": [int(sp.paso[0]), int(sp.paso[1])],
            "choca": {str(int(c)): int(n) for c, n in sp.choca.items()},
            "pasa": {str(int(c)): int(n) for c, n in sp.pasa.items()},
            "ultima": [int(sp.ultima[0]), int(sp.ultima[1])] if sp.ultima is not None else None,
            "confiable": bool(sp.confiable),
        }
    parches = {}
    for k, lista in modelo.parches.items():
        parches[k] = [{"celdas": [[int(dx), int(dy), int(a), int(d)] for (dx, dy), (a, d) in p.celdas.items()],
                       "puntaje": int(p.aciertos - p.fallas), "confiable": bool(p.confiable)} for p in lista]
    memoria = {}
    for antes, accion, despues in trans[-MAX_MEMORIA:]:
        dif = np.nonzero(antes != despues)
        clave = huella_pura(antes.tolist(), celdas_masc) + "|" + accion
        memoria[clave] = [[int(x), int(y), int(despues[y, x])] for y, x in zip(*dif)]
    return {
        "mascara": sorted([x, y] for x, y in celdas_masc),
        "sprites": sprites,
        "parches": parches,
        "suelo": [[int(x), int(y), int(c)] for (x, y), c in modelo.suelo.items()],
        "memoria": memoria,
    }


CONTADORES = {}


def contar(clave, n=1):
    CONTADORES[clave] = CONTADORES.get(clave, 0) + n


def motor_para(history, nivel):
    """Reglas serializadas si el motor tiene una regla completa del nivel; si no, None.
    Devuelve también (aciertos, medidas) para el aviso."""
    try:
        trans = transiciones(history, nivel, MAX_TRANS)
        contar("llamadas")
        r = aprender(trans)
        if r is None:
            return None, (0, 0)
        modelo, mascara, ok, n = r
        if n < MINIMO or ok < UMBRAL * n:
            return None, (ok, n)
        try:
            datos = serializar(modelo, mascara, trans)
        except Exception:
            contar("serialize_failed")
            return None, (ok, n)
        contar("motor_offered")
        return datos, (ok, n)
    except Exception:
        contar("learn_failed")
        return None, (0, 0)
