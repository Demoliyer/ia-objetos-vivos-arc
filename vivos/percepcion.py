"""Percepción: cuadros 64x64 → objetos, ruido y efectos de una transición.

Todo es determinista y sin aprendizaje: lo que se aprende vive en la colonia de
reglas (``reglas.py``). Ver ``docs/IA_OBJETOS_VIVOS.md`` §3.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

LADO = 64


@dataclass(frozen=True)
class Objeto:
    """Componente 4-conexo de un color. Coordenadas (x, y) de pantalla."""

    color: int
    celdas: frozenset[tuple[int, int]]
    x0: int
    y0: int
    x1: int  # inclusivo
    y1: int
    marco: bool = False

    @property
    def tamanio(self) -> int:
        return len(self.celdas)

    @property
    def forma(self) -> frozenset[tuple[int, int]]:
        return frozenset((x - self.x0, y - self.y0) for x, y in self.celdas)

    @property
    def es_marco(self) -> bool:
        """Letterbox o muro perimetral: toca los cuatro bordes o vive fuera del área de juego."""
        return self.marco or (self.x0 == 0 and self.y0 == 0 and self.x1 == LADO - 1 and self.y1 == LADO - 1)

    def celda_interior(self) -> tuple[int, int]:
        """Una celda del objeto lo más cercana posible a su centro (para clicks)."""
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        return min(self.celdas, key=lambda c: ((c[0] - cx) ** 2 + (c[1] - cy) ** 2, c[1], c[0]))


def cuadro(frames) -> np.ndarray:
    """El último cuadro de una respuesta del juego como ``int16`` (y, x)."""
    if frames is None or len(frames) == 0:
        raise ValueError("respuesta sin cuadros")
    arr = np.asarray(frames[-1], dtype=np.int16)
    if arr.shape != (LADO, LADO):
        raise ValueError(f"cuadro de forma {arr.shape}, se esperaba {(LADO, LADO)}")
    return arr


def interior(grid: np.ndarray) -> tuple[int, int, int, int]:
    """Rectángulo (y0, y1, x0, x1) sin el letterbox: filas y columnas exteriores
    de un único color, el mismo en los cuatro lados, centradas, y que no
    aparece adentro."""
    alto, ancho = grid.shape
    y0, y1, x0, x1 = 0, alto, 0, ancho
    borde = int(grid[0, 0])
    while y1 - y0 > 1 and (grid[y0, x0:x1] == borde).all():
        y0 += 1
    while y1 - y0 > 1 and (grid[y1 - 1, x0:x1] == borde).all():
        y1 -= 1
    while x1 - x0 > 1 and (grid[y0:y1, x0] == borde).all():
        x0 += 1
    while x1 - x0 > 1 and (grid[y0:y1, x1 - 1] == borde).all():
        x1 -= 1
    # Si el color del borde también está adentro, no es letterbox: es el fondo del
    # juego que llega hasta el borde.
    # El letterbox del motor está centrado; un margen de fondo alrededor de un
    # objeto, no.
    centrado = abs(y0 - (alto - y1)) <= 1 and abs(x0 - (ancho - x1)) <= 1
    if not centrado or (grid[y0:y1, x0:x1] == borde).any():
        return 0, alto, 0, ancho
    return y0, y1, x0, x1


def fondo(grid: np.ndarray) -> int:
    """Color más frecuente dentro del área de juego (sin letterbox)."""
    y0, y1, x0, x1 = interior(grid)
    valores, cuentas = np.unique(grid[y0:y1, x0:x1], return_counts=True)
    return int(valores[np.argmax(cuentas)])


_CACHE_SEG: "OrderedDict[bytes, tuple[Objeto, ...]]" = OrderedDict()
_CACHE_SEG_MAX = 8192


def segmentar(grid: np.ndarray, mascara: np.ndarray | None = None, *, incluir_fondo: bool = False) -> list[Objeto]:
    """Componentes 4-conexos por color. Las celdas enmascaradas no forman objetos.

    Es pura: los estados se repiten mucho (sobre todo al releer la evidencia cuando
    cambia la máscara), así que se recuerda por contenido."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(grid).tobytes())
    h.update(str(grid.shape).encode())
    h.update(b"F" if incluir_fondo else b"-")
    if mascara is not None:
        h.update(np.ascontiguousarray(mascara).tobytes())
    k = h.digest()
    hit = _CACHE_SEG.get(k)
    if hit is not None:
        _CACHE_SEG.move_to_end(k)
        return list(hit)
    objetos = _segmentar(grid, mascara, incluir_fondo)
    _CACHE_SEG[k] = tuple(objetos)
    if len(_CACHE_SEG) > _CACHE_SEG_MAX:
        _CACHE_SEG.popitem(last=False)
    return objetos


def _segmentar(grid: np.ndarray, mascara: np.ndarray | None, incluir_fondo: bool) -> list[Objeto]:
    bg = fondo(grid)
    iy0, iy1, ix0, ix1 = interior(grid)
    alto, ancho = grid.shape
    visto = np.zeros_like(grid, dtype=bool)
    if mascara is not None:
        visto |= mascara
    if not incluir_fondo:
        visto |= grid == bg
    objetos: list[Objeto] = []
    ys, xs = np.nonzero(~visto)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visto[y0, x0]:
            continue
        color = int(grid[y0, x0])
        pila = [(x0, y0)]
        visto[y0, x0] = True
        celdas = []
        while pila:
            x, y = pila.pop()
            celdas.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < ancho and 0 <= ny < alto and not visto[ny, nx] and grid[ny, nx] == color:
                    visto[ny, nx] = True
                    pila.append((nx, ny))
        cx = [c[0] for c in celdas]
        cy = [c[1] for c in celdas]
        afuera = all(not (iy0 <= y < iy1 and ix0 <= x < ix1) for x, y in celdas)
        objetos.append(Objeto(color, frozenset(celdas), min(cx), min(cy), max(cx), max(cy), afuera))
    objetos.sort(key=lambda o: (o.y0, o.x0, o.color))
    return objetos


def clave(grid: np.ndarray, mascara: np.ndarray | None) -> str:
    """Huella del estado visible, sin las celdas de ruido."""
    g = grid.astype(np.int8, copy=True)
    if mascara is not None and mascara.any():
        g[mascara] = -1
    return hashlib.blake2b(g.tobytes(), digest_size=12).hexdigest()


GROSOR_RELOJ = 6  # celdas: más grueso que esto en las dos dimensiones no es una barra


class DetectorDeRuido:
    """Relojes y contadores: colores que cambian en casi todas las transiciones
    y siempre de la misma manera, hagas lo que hagas.

    Un color es reloj si cambió en al menos el 80 % de las transiciones (mínimo
    4) y su firma más común (desplazamiento o variación de cantidad) cubre el
    60 % de sus cambios y apareció en al menos dos contextos distintos (otra
    acción, o la misma acción con y sin otros cambios). El avatar no califica: su
    desplazamiento depende de la acción. La máscara es la
    caja de cada reloj, no la fila o columna entera.
    """

    UMBRAL = 0.8
    MINIMO = 4
    COBERTURA = 0.6

    def __init__(self) -> None:
        self.reiniciar()

    def reiniciar(self) -> None:
        self.transiciones = 0
        self._cambios: dict[int, int] = {}
        self._firmas: dict[int, dict[tuple, set[str]]] = {}
        self._cuenta_firmas: dict[int, dict[tuple, int]] = {}
        # Zona de cada color por firma: un reloj crece solo donde cambia como reloj.
        # Si el mismo color también es parte del juego (piezas que se mueven), esos
        # cambios caen en otra firma y no agrandan la máscara.
        self._zona_firma: dict[int, dict[tuple, np.ndarray]] = {}
        self._zona_heredada: dict[int, np.ndarray] = {}
        self.relojes: list[int] = []
        self.heredados: list[int] = []
        self.mascara = np.zeros((LADO, LADO), dtype=bool)

    def heredar(self, relojes: list[int], grid: np.ndarray, zonas: dict[int, np.ndarray] | None = None) -> None:
        """Nivel nuevo: los colores que eran relojes se enmascaran desde el primer
        cuadro, sin esperar a redescubrirlos.

        Con ``zonas`` (dónde vivía cada reloj), solo se hereda el color cerca de ese
        lugar: el color de una barra puede ser el piso del nivel siguiente, y tapar
        todo el piso deja al agente ciego (ls20, nivel 2: 1980 celdas tapadas)."""
        for c in relojes:
            zona = grid == c
            if zonas is not None and c in zonas and zonas[c].any():
                zona &= _dilatar(_dilatar(_caja(zonas[c])))
            if not zona.any():
                continue
            self._zona_heredada[c] = zona.copy()
            self.mascara |= _caja(zona)
        self.heredados = [c for c in relojes if c in self._zona_heredada]
        self.relojes = list(self.heredados)

    def _zona(self, c: int) -> np.ndarray:
        """Donde vive el reloj ``c``: lo heredado más lo que cambió con su firma dominante."""
        zona = np.zeros((LADO, LADO), dtype=bool)
        if c in self._zona_heredada:
            zona |= self._zona_heredada[c]
        cf = self._cuenta_firmas.get(c)
        if cf:
            f = max(cf.items(), key=lambda kv: kv[1])[0]
            zona |= self._zona_firma[c][f]
        return zona

    @staticmethod
    def _firma(a: np.ndarray, d: np.ndarray) -> tuple | None:
        na, nd = int(a.sum()), int(d.sum())
        if na == nd and na > 0:
            ya, xa = np.nonzero(a)
            yd, xd = np.nonzero(d)
            f = ("mueve", int(yd.min() - ya.min()), int(xd.min() - xa.min()))
            return None if f == ("mueve", 0, 0) else f
        return None if na == nd else ("cuenta", nd - na)

    def observar(self, antes: np.ndarray, despues: np.ndarray, accion: str = "") -> bool:
        """Registra una transición. Devuelve True si la máscara cambió."""
        self.transiciones += 1
        cambio = antes != despues
        if cambio.any():
            bg = fondo(antes)
            colores = set(np.unique(antes[cambio]).tolist()) | set(np.unique(despues[cambio]).tolist())
            otros_cambian = {c: False for c in colores}
            for c in colores:
                otros_cambian[c] = any(o != c and o != bg for o in colores)
            for c in colores:
                if c == bg:
                    continue
                # Contexto: la acción y si cambió algo más. Un reloj cambia igual
                # cuando la acción no hizo nada que cuando sí.
                contexto = f"{accion}|{otros_cambian[c]}"
                a, d = antes == c, despues == c
                self._cambios[c] = self._cambios.get(c, 0) + 1
                f = self._firma(a, d)
                if f is not None:
                    self._firmas.setdefault(c, {}).setdefault(f, set()).add(contexto)
                    cf = self._cuenta_firmas.setdefault(c, {})
                    cf[f] = cf.get(f, 0) + 1
                    zona = self._zona_firma.setdefault(c, {}).setdefault(f, np.zeros((LADO, LADO), dtype=bool))
                    celdas = cambio & (a | d)
                    # Más celdas de las que explica la firma (la barra creció 1 pero además
                    # se movieron piezas del mismo color): solo cuenta lo pegado a la zona.
                    limite = abs(f[1]) + 2 if f[0] == "cuenta" else 2 * int(a.sum()) + 2
                    if int(celdas.sum()) > limite:
                        celdas &= _dilatar(zona)
                    zona |= celdas
        if self.transiciones < self.MINIMO:
            return False
        # Un reloj detectado no se olvida dentro del nivel: un RESET que rellena la
        # barra o una racha sin cambios no lo vuelven parte del juego.
        relojes = list(dict.fromkeys(self.heredados + self.relojes))
        candidatos = []
        for c, n in self._cambios.items():
            if c in relojes or n < self.UMBRAL * self.transiciones or c not in self._cuenta_firmas:
                continue
            f, veces = max(self._cuenta_firmas[c].items(), key=lambda kv: kv[1])
            if veces >= self.COBERTURA * n and len(self._firmas[c][f]) >= 2:
                candidatos.append(c)
        # El color que aparece donde el reloj se borra (el piso bajo una barra que
        # se gasta) cambia igual de seguido, pero es consecuencia: de dos
        # candidatos que se pisan, el reloj es el más raro.
        abundancia = {int(c): int(k) for c, k in zip(*np.unique(despues, return_counts=True))}
        for c in sorted(candidatos, key=lambda c: (abundancia.get(c, 0), c)):
            zona = self._zona(c)
            if any((zona & self._zona(r)).sum() >= 0.5 * max(1, self._zona(r).sum()) for r in relojes):
                continue
            relojes.append(c)
        nueva = np.zeros((LADO, LADO), dtype=bool)
        for c in relojes:
            # Solo la caja del reloj: lo que haya al lado, en la misma fila o
            # columna, sigue visible. Incluye las partes del reloj que todavía no
            # cambiaron (el resto de una barra llena) si tocan la zona ya vista.
            zona = self._zona(c)
            nueva |= _caja_de_reloj(zona, despues == c)
        self.relojes = sorted(relojes)
        if np.array_equal(nueva, self.mascara):
            return False
        self.mascara = nueva
        return True


def _dilatar(zona: np.ndarray) -> np.ndarray:
    """La zona más sus 8 vecinos."""
    out = zona.copy()
    if not zona.any():
        return out
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out |= np.roll(np.roll(zona, dy, axis=0), dx, axis=1) & _sin_vuelta(zona.shape, dy, dx)
    return out


def _sin_vuelta(forma: tuple, dy: int, dx: int) -> np.ndarray:
    """Máscara que anula lo que np.roll trae del borde opuesto."""
    m = np.ones(forma, dtype=bool)
    if dy > 0:
        m[:dy, :] = False
    elif dy < 0:
        m[dy:, :] = False
    if dx > 0:
        m[:, :dx] = False
    elif dx < 0:
        m[:, dx:] = False
    return m


RIGIDA_COBERTURA = 0.85  # fracción del cambio que una traslación tiene que explicar


def _traslacion_rigida(antes: np.ndarray, despues: np.ndarray, cambio: np.ndarray, bg: int):
    """El desplazamiento que mejor explica el cambio como un sprite corrido sobre un
    fondo quieto: ((dx, dy), celdas del sprite antes, celdas explicadas) o None.

    Candidatos: para cada color, lo que perdió contra lo que ganó (esquina de sus
    cajas). Un sprite puede cambiar algo por dentro al moverse (un indicador de
    dirección), por eso alcanza con explicar RIGIDA_COBERTURA del cambio."""
    if int(cambio.sum()) < 2:
        return None
    alto, ancho = antes.shape
    candidatos = set()
    for c in np.unique(np.concatenate([antes[cambio], despues[cambio]])).tolist():
        if c == bg:
            continue
        pierde = cambio & (antes == c) & (despues != c)
        gana = cambio & (despues == c) & (antes != c)
        if pierde.any() and gana.any():
            yp, xp = np.nonzero(pierde)
            yg, xg = np.nonzero(gana)
            candidatos.add((int(xg.min() - xp.min()), int(yg.min() - yp.min())))
            candidatos.add((int(xg.max() - xp.max()), int(yg.max() - yp.max())))
    mejor = None
    for dx, dy in candidatos:
        if (dx, dy) == (0, 0) or abs(dx) >= ancho or abs(dy) >= alto:
            continue
        # corrido[y, x] = antes[y - dy, x - dx]
        corrido = np.full_like(antes, -1)
        ys0, ys1 = max(0, dy), min(alto, alto + dy)
        xs0, xs1 = max(0, dx), min(ancho, ancho + dx)
        corrido[ys0:ys1, xs0:xs1] = antes[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
        llega = cambio & (despues == corrido) & (corrido != bg) & (corrido >= 0)
        if int(llega.sum()) < 2:
            continue
        # Un piso uniforme coincide consigo mismo al correrlo: un color es parte del
        # sprite solo si al menos la mitad de sus celdas de origen cambiaron.
        yl, xl = np.nonzero(llega)
        origen_cambia = cambio[yl - dy, xl - dx]
        colores_llega = corrido[yl, xl]
        quedan = np.zeros(len(yl), dtype=bool)
        for c in np.unique(colores_llega).tolist():
            sel = colores_llega == c
            if origen_cambia[sel].mean() >= 0.5:
                quedan |= sel
        yl, xl = yl[quedan], xl[quedan]
        if len(yl) < 2:
            continue
        llega = np.zeros_like(cambio)
        llega[yl, xl] = True
        sprite = np.zeros_like(cambio)
        sprite[yl - dy, xl - dx] = True
        # Lo que cambia dentro de la caja del sprite (origen o destino) es el sprite: puede
        # cambiar algo por dentro al moverse (tu93: el indicador de dirección gira).
        explica = cambio & (_caja(sprite) | _caja(llega))
        # Cobertura local: el cambio alrededor del sprite (un contador lejano no cuenta).
        zona = _dilatar(_dilatar(_caja(sprite | llega)))
        local = int((cambio & zona).sum())
        cubre = int((explica & zona).sum()) / max(1, local)
        if cubre < RIGIDA_COBERTURA:
            continue
        # Una traslación y la "des-oclusión" del fondo se espejan y explican casi lo
        # mismo. Lo raro se mueve; lo abundante es fondo: gana el sprite cuyo color
        # principal es menos abundante en el cuadro.
        valores, cuentas = np.unique(antes[sprite], return_counts=True)
        principal = int(valores[np.argmax(cuentas)])
        abundancia = int((antes == principal).sum())
        clave_orden = (abundancia, -cubre)
        if mejor is None or clave_orden < mejor[0]:
            mejor = (clave_orden, (dx, dy), sprite, explica)
    if mejor is None:
        return None
    _, d, sprite, explica = mejor
    ys, xs = np.nonzero(sprite)
    return d, list(zip(xs.tolist(), ys.tolist())), explica


def _caja_de_reloj(zona: np.ndarray, color: np.ndarray) -> np.ndarray:
    """La caja del reloj, extendida a la parte de la barra que todavía no cambió.

    Un reloj de HUD es delgado: si la extensión lo vuelve grueso en las dos
    dimensiones, se pegó a algo del mismo color (un piso) y se descarta; queda solo
    la zona vista (tn36, nivel 2: la barra pegada al piso tapaba 1891 celdas)."""
    base = _caja(zona)
    ext = _caja(zona | _pegado(color, zona))
    if not ext.any():
        return base
    ys, xs = np.nonzero(ext)
    alto, ancho = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if min(alto, ancho) > GROSOR_RELOJ:
        return base
    return ext


def _caja(zona: np.ndarray) -> np.ndarray:
    """Rectángulo mínimo que contiene la zona."""
    out = np.zeros_like(zona, dtype=bool)
    if zona.any():
        ys, xs = np.nonzero(zona)
        out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return out


def _pegado(celdas: np.ndarray, semilla: np.ndarray) -> np.ndarray:
    """Las celdas de ``celdas`` conectadas (4-vecinos) con la caja de ``semilla``."""
    caja = _caja(semilla)
    out = celdas & caja
    if not out.any():
        return out
    while True:
        crece = out.copy()
        crece[1:] |= out[:-1]
        crece[:-1] |= out[1:]
        crece[:, 1:] |= out[:, :-1]
        crece[:, :-1] |= out[:, 1:]
        crece &= celdas
        if np.array_equal(crece, out):
            return out
        out = crece


@dataclass
class Efecto:
    """Qué cambió entre dos cuadros, por objetos, fuera del ruido."""

    mueve: list[tuple[int, int, int]] = field(default_factory=list)  # (color, dx, dy) en píxeles
    tam_mueve: list[tuple[int, int]] = field(default_factory=list)  # (ancho, alto) de cada objeto movido
    recolorea: list[tuple[int, int]] = field(default_factory=list)  # (de, a)
    desaparece: list[int] = field(default_factory=list)
    aparece: list[int] = field(default_factory=list)
    otro: bool = False
    celdas: int = 0  # celdas cambiadas fuera del ruido

    @property
    def nada(self) -> bool:
        return self.celdas == 0

    def resumen(self) -> str:
        if self.nada:
            return "nada"
        partes = [f"mueve {c} ({dx},{dy})" for c, dx, dy in self.mueve]
        partes += [f"recolorea {a}->{b}" for a, b in self.recolorea]
        partes += [f"desaparece {c}" for c in self.desaparece]
        partes += [f"aparece {c}" for c in self.aparece]
        if self.otro:
            partes.append("otro")
        return ", ".join(partes)


def efecto(antes: np.ndarray, despues: np.ndarray, mascara: np.ndarray | None = None) -> Efecto:
    cambio = antes != despues
    if mascara is not None:
        cambio &= ~mascara
    out = Efecto(celdas=int(cambio.sum()))
    if out.nada:
        return out
    # Cambio total de escena (subida de nivel, pantalla nueva): no se interpreta.
    if out.celdas > 0.6 * LADO * LADO:
        out.otro = True
        return out
    bg_a, bg_d = fondo(antes), fondo(despues)
    tocados_a = [o for o in segmentar(antes, mascara) if o.color != bg_a and _toca(o, cambio)]
    tocados_d = [o for o in segmentar(despues, mascara) if o.color != bg_d and _toca(o, cambio)]
    usados_a: set[int] = set()
    usados_d: set[int] = set()
    explicado = np.zeros_like(cambio)

    def explicar(*objs: Objeto) -> None:
        for o in objs:
            for x, y in o.celdas:
                explicado[y, x] = True

    # Traslación rígida: un sprite (de uno o varios colores) que se corre sobre un fondo.
    # Se explica antes que nada, para que lo que destapa no parezca moverse al revés
    # (tu93: el bloque que pasa sobre un nodo hacía creer que el nodo era el avatar).
    rigida = _traslacion_rigida(antes, despues, cambio, bg_a)
    if rigida is not None:
        (dx, dy), sprite, explica = rigida
        colores = {}
        for x, y in sprite:
            colores.setdefault(int(antes[y, x]), []).append((x, y))
        for c, celdas in sorted(colores.items(), key=lambda kv: -len(kv[1])):
            xs = [x for x, _ in celdas]
            ys = [y for _, y in celdas]
            out.mueve.append((c, dx, dy))
            out.tam_mueve.append((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
        explicado |= explica

    # Los colores raros se mueven; los abundantes (piso, fondo mal estimado) son lo que se descubre.
    abundancia = {int(c): int(n) for c, n in zip(*np.unique(antes, return_counts=True))}
    orden = sorted(range(len(tocados_a)), key=lambda i: (abundancia.get(tocados_a[i].color, 0), i))

    def ya_explicado(o: Objeto) -> bool:
        return all(explicado[y, x] for x, y in o.celdas)

    # Movimiento: misma forma y color, desplazada. Ante varias, la más cercana.
    for i in orden:
        a = tocados_a[i]
        if ya_explicado(a):
            usados_a.add(i)
            continue
        opciones = [
            (abs(d.x0 - a.x0) + abs(d.y0 - a.y0), j)
            for j, d in enumerate(tocados_d)
            if j not in usados_d and a.color == d.color and a.forma == d.forma and a.celdas != d.celdas
        ]
        if opciones:
            _, j = min(opciones)
            d = tocados_d[j]
            out.mueve.append((a.color, d.x0 - a.x0, d.y0 - a.y0))
            out.tam_mueve.append((a.x1 - a.x0 + 1, a.y1 - a.y0 + 1))
            usados_a.add(i)
            usados_d.add(j)
            explicar(a, d)
    for j, d in enumerate(tocados_d):
        if j not in usados_d and ya_explicado(d):
            usados_d.add(j)
    # Recoloreo: mismas celdas, otro color.
    for i, a in enumerate(tocados_a):
        if i in usados_a:
            continue
        for j, d in enumerate(tocados_d):
            if j in usados_d or a.color == d.color or a.celdas != d.celdas:
                continue
            out.recolorea.append((a.color, d.color))
            usados_a.add(i)
            usados_d.add(j)
            explicar(a, d)
            break
    # Lo que queda sin explicar: objetos que aparecen, desaparecen o cambian de forma.
    resto = cambio & ~explicado
    if not resto.any():
        return out
    sobran_a = [a for i, a in enumerate(tocados_a) if i not in usados_a and _toca(a, resto)]
    sobran_d = [d for j, d in enumerate(tocados_d) if j not in usados_d and _toca(d, resto)]
    colores_d = {d.color for d in sobran_d}
    colores_a = {a.color for a in sobran_a}
    for a in sobran_a:
        if a.color not in colores_d:
            if a.color not in out.desaparece:
                out.desaparece.append(a.color)
        else:
            out.otro = True
    for d in sobran_d:
        if d.color not in colores_a and d.color not in out.aparece:
            out.aparece.append(d.color)
    if not (sobran_a or sobran_d):
        # Solo cambió el fondo en celdas que nadie explica (p. ej. fondo mal estimado).
        fuera = resto.copy()
        for o in tocados_a + tocados_d:
            for x, y in o.celdas:
                fuera[y, x] = False
        out.otro = bool(fuera.any()) and not out.mueve and not out.recolorea
    return out


def tamanio_de(grid: np.ndarray, color: int, mascara: np.ndarray | None = None) -> tuple[int, int] | None:
    """(ancho, alto) del componente más grande de ``color``: la unidad de movimiento del avatar."""
    objs = [o for o in segmentar(grid, mascara, incluir_fondo=True) if o.color == color]
    if not objs:
        return None
    o = max(objs, key=lambda o: (o.tamanio, -o.y0, -o.x0))
    return o.x1 - o.x0 + 1, o.y1 - o.y0 + 1


def _toca(o: Objeto, cambio: np.ndarray) -> bool:
    return any(cambio[y, x] for x, y in o.celdas)
