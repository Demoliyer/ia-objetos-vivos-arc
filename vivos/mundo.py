"""Modelo del mundo simulable: predice el cuadro siguiente a partir de reglas vivas.

Medido sobre trayectorias reales, un lenguaje "por objeto de un color" no alcanza: el
58 % de las transiciones con cambio son fusiones, separaciones o cambios de forma de
componentes de un color (una pieza que toca algo de su color se "funde"). Por eso las
reglas no dependen de la segmentación:

1. Sprite rígido (flechas y acciones que desplazan algo): el dibujo multicolor que
   se corre, su desplazamiento por acción, con qué colores choca (aprendido de las
   veces que la acción no hizo nada) y qué piso queda al destaparlo (memoria de piso).
2. Parche relativo (clicks y acciones con efecto local): las celdas que cambiaron,
   relativas al objeto clickeado, con su color de antes y de después. Se aplica en
   otro click sobre el mismo tipo de objeto si las celdas de "antes" coinciden: eso
   captura interruptores, apariciones y cambios de forma, y la condición sale sola.
3. Memoria exacta: si ese estado con esa acción ya se vio, lo mismo que entonces.

Cada regla es legible y se verifica contra lo observado; una regla que predice mal
pierde confianza y deja lugar a la siguiente.

El módulo no decide: predice. Se mide aparte con ``evaluar_mundo.py`` antes de que
el agente planifique con él.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .percepcion import Objeto, _traslacion_rigida, fondo, segmentar

# ---------------------------------------------------------------- utilidades


def tipo_de(o: Objeto) -> tuple[int, int, int]:
    return (o.color, o.x1 - o.x0, o.y1 - o.y0)


def objeto_en(grid: np.ndarray, mascara: np.ndarray | None, x: int, y: int) -> Objeto | None:
    for o in segmentar(grid, mascara):
        if o.x0 <= x <= o.x1 and o.y0 <= y <= o.y1 and (x, y) in o.celdas:
            return o
    return None


def huella(grid: np.ndarray, mascara: np.ndarray | None) -> bytes:
    g = grid.astype(np.int8, copy=True)
    if mascara is not None and mascara.any():
        g[mascara] = -1
    return g.tobytes()


def disparador(accion: str, grid: np.ndarray, mascara: np.ndarray | None) -> tuple[str, tuple[int, int] | None]:
    """Clave del disparador y el ancla (esquina del objeto clickeado) si es un click."""
    if "@" not in accion:
        return accion, None
    base = accion.split("@")[0]
    x, y = (int(v) for v in accion.split("@")[1].split(","))
    u = objeto_en(grid, mascara, x, y)
    if u is None:
        return f"{base}@fondo", (x, y)
    c, w, h = tipo_de(u)
    return f"{base}@{c},{w},{h}", (u.x0, u.y0)


# ---------------------------------------------------------------- sprite rígido


@dataclass
class Sprite:
    """Lo que una acción desplaza: dibujo relativo (dx, dy) -> color y paso."""

    dibujo: dict[tuple[int, int], int]
    paso: tuple[int, int]
    confirmaciones: int = 1
    fallas: int = 0
    aciertos: int = 0
    choca: Counter = field(default_factory=Counter)  # color en el destino cuando no se movió
    pasa: Counter = field(default_factory=Counter)  # color en el destino cuando sí se movió
    ultima: tuple[int, int] | None = None  # esquina donde se lo vio por última vez

    @property
    def confiable(self) -> bool:
        return self.fallas <= self.aciertos

    def ubicar(self, grid: np.ndarray) -> tuple[int, int] | None:
        """Esquina donde está el dibujo (coincidencia exacta), la más cercana a la última."""
        if not self.dibujo:
            return None
        alto, ancho = grid.shape
        (rx, ry), c0 = next(iter(self.dibujo.items()))
        ys, xs = np.nonzero(grid == c0)
        mejores = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            ox, oy = x - rx, y - ry
            ok = True
            for (dx, dy), c in self.dibujo.items():
                X, Y = ox + dx, oy + dy
                if not (0 <= X < ancho and 0 <= Y < alto) or grid[Y, X] != c:
                    ok = False
                    break
            if ok:
                mejores.append((ox, oy))
        if not mejores:
            return None
        if self.ultima is None:
            return mejores[0]
        lx, ly = self.ultima
        return min(mejores, key=lambda p: abs(p[0] - lx) + abs(p[1] - ly))

    def bloqueado(self, grid: np.ndarray, esquina: tuple[int, int], bg: int) -> bool:
        """El destino pisa un color que solo se vio frenando, nunca dejando pasar."""
        alto, ancho = grid.shape
        ox, oy = esquina
        px, py = self.paso
        propias = {(ox + dx, oy + dy) for dx, dy in self.dibujo}
        for dx, dy in self.dibujo:
            X, Y = ox + dx + px, oy + dy + py
            if not (0 <= X < ancho and 0 <= Y < alto):
                return True
            if (X, Y) in propias:
                continue
            c = int(grid[Y, X])
            if c != bg and self.choca[c] > 0 and self.pasa[c] == 0:
                return True
        return False


# ---------------------------------------------------------------- parche relativo


@dataclass
class Parche:
    """Celdas que cambian, relativas al ancla: (dx, dy) -> (color antes, color después)."""

    celdas: dict[tuple[int, int], tuple[int, int]]
    confirmaciones: int = 1
    fallas: int = 0
    aciertos: int = 0

    @property
    def confiable(self) -> bool:
        return self.fallas <= self.aciertos

    def aplica(self, grid: np.ndarray, ancla: tuple[int, int]) -> bool:
        alto, ancho = grid.shape
        ax, ay = ancla
        for (dx, dy), (antes, _) in self.celdas.items():
            X, Y = ax + dx, ay + dy
            if not (0 <= X < ancho and 0 <= Y < alto) or grid[Y, X] != antes:
                return False
        return True

    def aplicar(self, grid: np.ndarray, ancla: tuple[int, int]) -> np.ndarray:
        out = grid.copy()
        ax, ay = ancla
        for (dx, dy), (_, despues) in self.celdas.items():
            out[ay + dy, ax + dx] = despues
        return out


MAX_PARCHES = 12  # por disparador


# ---------------------------------------------------------------- modelo


class ModeloMundo:
    def __init__(self) -> None:
        self.memoria: dict[tuple[bytes, str], np.ndarray] = {}
        self.sprites: dict[str, Sprite] = {}
        self.parches: dict[str, list[Parche]] = {}
        self.nada: Counter = Counter()  # disparador -> veces que no hizo nada
        self.suelo: dict[tuple[int, int], int] = {}  # celda -> color visto al destaparla

    # -------------------------------------------------------- aprender

    def observar(self, antes: np.ndarray, accion: str, despues: np.ndarray, mascara: np.ndarray | None) -> None:
        fuera = np.ones(antes.shape, dtype=bool) if mascara is None else ~mascara
        cambio = (antes != despues) & fuera
        clave, ancla = disparador(accion, antes, mascara)
        bg = fondo(antes)
        self._verificar(antes, accion, despues, fuera, clave, ancla, bg)
        self.memoria[(huella(antes, mascara), accion)] = despues
        sp = self.sprites.get(accion) if ancla is None else None
        if not cambio.any():
            self.nada[clave] += 1
            if sp is not None:
                esq = sp.ubicar(antes)
                if esq is not None:
                    self._anotar_destino(sp, antes, esq, bg, sp.choca)
            return
        rigida = _traslacion_rigida(antes, despues, cambio, bg) if ancla is None else None
        if rigida is not None:
            (dx, dy), celdas, explica = rigida
            xs = [x for x, _ in celdas]
            ys = [y for _, y in celdas]
            ox, oy = min(xs), min(ys)
            dibujo = {(x - ox, y - oy): int(antes[y, x]) for x, y in celdas}
            if sp is None or sp.paso != (dx, dy) or sp.dibujo != dibujo:
                if sp is not None and sp.paso == (dx, dy):
                    # mismo paso, el dibujo cambió (un indicador que gira): se actualiza
                    sp.dibujo = dibujo
                    sp.confirmaciones += 1
                else:
                    sp = Sprite(dibujo, (dx, dy))
                    self.sprites[accion] = sp
            else:
                sp.confirmaciones += 1
            self._anotar_destino(sp, antes, (ox, oy), bg, sp.pasa)
            sp.ultima = (ox + dx, oy + dy)
            # piso destapado: lo que queda donde estaba el sprite y ya no está
            nuevas = {(x + dx, y + dy) for x, y in celdas}
            for x, y in celdas:
                if (x, y) not in nuevas:
                    self.suelo[(x, y)] = int(despues[y, x])
            resto = cambio & ~explica
            if not resto.any():
                return
        else:
            resto = cambio
        # Parche relativo al ancla (o absoluto si no hay ancla).
        ax, ay = ancla if ancla is not None else (0, 0)
        ys, xs = np.nonzero(resto)
        celdas = {(int(x) - ax, int(y) - ay): (int(antes[y, x]), int(despues[y, x])) for x, y in zip(xs, ys)}
        lista = self.parches.setdefault(clave, [])
        for p in lista:
            if p.celdas == celdas:
                p.confirmaciones += 1
                return
        lista.append(Parche(celdas))
        if len(lista) > MAX_PARCHES:
            lista.sort(key=lambda p: (p.aciertos - p.fallas, p.confirmaciones), reverse=True)
            del lista[MAX_PARCHES:]

    def _verificar(self, antes, accion, despues, fuera, clave, ancla, bg) -> None:
        """Cada regla que habría actuado compara su predicción con lo que pasó: suma un
        acierto o una falla. Así una regla que predice mal pierde la palabra."""
        sp = self.sprites.get(accion) if ancla is None else None
        if sp is not None:
            esq = sp.ubicar(antes)
            if esq is not None and not sp.bloqueado(antes, esq, bg):
                pred = self._mover(antes, sp, esq, bg)
                if np.array_equal(pred[fuera], despues[fuera]):
                    sp.aciertos += 1
                else:
                    sp.fallas += 1
        ax, ay = ancla if ancla is not None else (0, 0)
        for p in self.parches.get(clave, []):
            if p.aplica(antes, (ax, ay)):
                ok = all(despues[ay + dy, ax + dx] == d for (dx, dy), (_, d) in p.celdas.items())
                if ok:
                    p.aciertos += 1
                else:
                    p.fallas += 1

    @staticmethod
    def _anotar_destino(sp: Sprite, grid: np.ndarray, esquina: tuple[int, int], bg: int, cuenta: Counter) -> None:
        alto, ancho = grid.shape
        ox, oy = esquina
        px, py = sp.paso
        propias = {(ox + dx, oy + dy) for dx, dy in sp.dibujo}
        for dx, dy in sp.dibujo:
            X, Y = ox + dx + px, oy + dy + py
            if 0 <= X < ancho and 0 <= Y < alto and (X, Y) not in propias:
                c = int(grid[Y, X])
                if c != bg:
                    cuenta[c] += 1

    # -------------------------------------------------------- predecir

    def predecir(self, antes: np.ndarray, accion: str, mascara: np.ndarray | None) -> np.ndarray:
        visto = self.memoria.get((huella(antes, mascara), accion))
        if visto is not None:
            return visto
        clave, ancla = disparador(accion, antes, mascara)
        bg = fondo(antes)
        salida = antes
        sp = self.sprites.get(accion) if ancla is None else None
        if sp is not None and sp.confiable:
            esq = sp.ubicar(antes)
            if esq is not None and not sp.bloqueado(antes, esq, bg):
                salida = self._mover(antes, sp, esq, bg)
        ax, ay = ancla if ancla is not None else (0, 0)
        for p in sorted(self.parches.get(clave, []), key=lambda p: -(p.aciertos - p.fallas)):
            if p.confiable and p.aplica(salida, (ax, ay)):
                return p.aplicar(salida, (ax, ay))
        return salida

    def _mover(self, grid: np.ndarray, sp: Sprite, esq: tuple[int, int], bg: int) -> np.ndarray:
        out = grid.copy()
        alto, ancho = grid.shape
        ox, oy = esq
        px, py = sp.paso
        for dx, dy in sp.dibujo:
            x, y = ox + dx, oy + dy
            out[y, x] = self.suelo.get((x, y), bg)
        for (dx, dy), c in sp.dibujo.items():
            X, Y = ox + dx + px, oy + dy + py
            if 0 <= X < ancho and 0 <= Y < alto:
                out[Y, X] = c
        return out
