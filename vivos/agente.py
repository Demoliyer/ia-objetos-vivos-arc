"""El agente: decide con la colonia de reglas y aprende de cada transición.

Ver ``docs/IA_OBJETOS_VIVOS.md`` §5. No conoce el kit oficial: recibe
observaciones con ``frame``, ``state``, ``levels_completed`` y
``available_actions`` y devuelve ``Decision`` con el nombre de la acción.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grafo import Grafo, Nodo
from .percepcion import DetectorDeRuido, Objeto, clave, cuadro, efecto, fondo, segmentar
from .reglas import ACTIVA, ATRAVIESA, BLOQUEA, CLICK_INERTE, FATAL, INERTE, Colonia, Transicion

MAX_CLICKS = 64
MARGEN_PRESUPUESTO = 2
MAX_DESVIO_NOVEDAD = 8  # pasos que se camina para ir a un estado novedoso  # pasos que hacen falta al llegar para aprovechar el estado
LIMITE_BFS = 6000
MAX_TOQUES = 3  # veces que se va a tocar el mismo objeto en un nivel


@dataclass
class Decision:
    accion: str  # "ACTION1" ... "ACTION7" o "RESET"
    x: int | None = None
    y: int | None = None
    motivo: str = ""
    tipo: str = "explorar"  # explorar | mover | tocar | grafo
    esperado: Any = None  # nodo destino (grafo) o (color, dx, dy) (mover)
    meta: tuple | None = None  # objeto a tocar

    @property
    def clave(self) -> str:
        return self.accion if self.x is None else f"{self.accion}@{self.x},{self.y}"

    @property
    def datos(self) -> dict | None:
        return None if self.x is None else {"x": self.x, "y": self.y}

    @staticmethod
    def desde_clave(clave: str, **kw) -> "Decision":
        if "@" in clave:
            accion, xy = clave.split("@")
            x, y = (int(v) for v in xy.split(","))
            return Decision(accion, x, y, **kw)
        return Decision(clave, **kw)


def _estado(obs) -> str:
    s = getattr(obs, "state", "")
    return getattr(s, "value", s) or ""


@dataclass
class Nivel:
    indice: int
    acciones: list[str] = field(default_factory=list)
    reinicios: int = 0


class AgenteVivo:
    def __init__(self, semilla: int = 0) -> None:
        self.colonia = Colonia()
        self.rng = random.Random(semilla)
        self.habilidades: dict[int, list[str]] = {}
        self.decisiones: deque[tuple[int, str, str]] = deque(maxlen=400)
        self.n = 0
        self._candidatas_cache: dict[int, list[str]] = {}
        # Colores que el avatar pisaba en la jugada ganadora de un nivel: la meta del
        # juego, para ir directo a ella en los niveles siguientes. Pasa entre niveles.
        self.metas_color: dict[int, int] = {}
        self._nuevo_nivel(0)

    # ------------------------------------------------------------------ ciclo

    def _nuevo_nivel(self, indice: int) -> None:
        self.ruido = DetectorDeRuido()
        self.grafo = Grafo()
        self.plan: deque[Decision] = deque()
        self.tocados: set[tuple[int, tuple]] = set()
        self.toques: dict[tuple, int] = {}  # objeto -> veces tocado en el nivel
        self.nivel = Nivel(indice)
        self._candidatas_cache.clear()
        # Presupuesto: pasos de la vida actual, largo de las vidas que terminaron en
        # GAME_OVER y el estado raíz (inicio del nivel) para reiniciar a propósito.
        self.paso_vida = 0
        self.vidas: list[int] = []
        self.raiz: int | None = None
        self._marcar_raiz = True
        self._plan_tras_reset: list[Decision] = []
        # Novedad (IW(1)): átomos (color, posición, forma) ya vistos en el nivel y si
        # cada estado trajo alguno nuevo al descubrirse.
        self.atomos: set[tuple] = set()
        self.novedad: dict[int, bool] = {}

    def reiniciado(self) -> None:
        """El juego volvió al inicio del nivel (RESET tras GAME_OVER)."""
        self.plan.clear()
        self.nivel.reinicios += 1
        self.paso_vida = 0
        self._marcar_raiz = True
        self._plan_tras_reset = []

    def decidir(self, obs) -> Decision:
        grid = cuadro(obs.frame)
        mascara = self.ruido.mascara
        nodo = self.grafo.nodo(grid, mascara)
        self._registrar_novedad(nodo, grid, mascara)
        if self._marcar_raiz:
            self.raiz = nodo.id
            self._marcar_raiz = False
        disponibles = list(getattr(obs, "available_actions", None) or [1, 2, 3, 4, 5, 6, 7])
        d = self._elegir(grid, mascara, nodo, disponibles)
        if d.accion == "RESET" and self.paso_vida == 0:
            # Con 0 acciones desde el inicio del nivel el motor reinicia el JUEGO entero
            # (vuelve al nivel 1). Nunca.
            d = Decision.desde_clave(self._cualquiera(nodo, disponibles), motivo="no reiniciar en el paso 0")
        self.decisiones.append((self.n, d.clave, d.motivo))
        return d

    def aprender(self, obs_antes, d: Decision, obs_despues) -> Transicion | None:
        antes = cuadro(obs_antes.frame)
        despues = cuadro(obs_despues.frame) if len(getattr(obs_despues, "frame", []) or []) else antes
        sube = obs_despues.levels_completed > obs_antes.levels_completed
        game_over = _estado(obs_despues) == "GAME_OVER"
        self.n += 1
        self.nivel.acciones.append(d.clave)
        if d.accion == "RESET":
            # Reinicio a propósito: vuelve a la raíz y sigue el camino más corto guardado.
            self.nivel.reinicios += 1
            self.paso_vida = 0
            self.plan.clear()
            self.plan.extend(self._plan_tras_reset)
            self._plan_tras_reset = []
            self.raiz = self.grafo.nodo(despues, self.ruido.mascara).id
            return None
        self.paso_vida += 1
        mascara = self.ruido.mascara
        if not sube:
            if self.ruido.observar(antes, despues, d.clave):
                mascara = self.ruido.mascara
                self.grafo.recalcular(mascara)
                self._candidatas_cache.clear()
                self.colonia.reinterpretar(self.nivel.indice, mascara)
        nodo_a = self.grafo.nodo(antes, mascara)
        ef = efecto(antes, despues, mascara)
        huella = clave(antes, mascara)
        if sube:
            ef.otro = True
        t = Transicion(
            n=self.n,
            accion=d.accion,
            click=(d.x, d.y) if d.x is not None else None,
            antes=antes,
            despues=despues,
            mascara=mascara.copy(),
            efecto=ef,
            game_over=game_over,
            sube_nivel=sube,
            clave_antes=huella,
            clave_despues="",
            nivel=self.nivel.indice,
        )
        self.colonia.registrar(t)
        if d.meta is not None:
            self.tocados.add((nodo_a.id, d.meta))
            self.toques[d.meta] = self.toques.get(d.meta, 0) + 1
        if sube:
            self._aprender_meta(antes, mascara, d)
            self.habilidades[self.nivel.indice] = list(self.nivel.acciones)
            relojes = list(self.ruido.relojes)
            zonas = {c: self.ruido._zona(c) for c in relojes}
            self._nuevo_nivel(obs_despues.levels_completed)
            self.ruido.heredar(relojes, despues, zonas)
            return t
        if game_over:
            self.grafo.arista(nodo_a, d.clave, None, fatal=True)
            self.plan.clear()
            self.vidas.append(self.paso_vida)
            return t
        nodo_d = self.grafo.nodo(despues, mascara)
        self._registrar_novedad(nodo_d, despues, mascara)
        t.clave_despues = clave(despues, mascara)
        self.grafo.arista(nodo_a, d.clave, nodo_d)
        if not self._salio_como_se_esperaba(d, ef, nodo_d):
            self.plan.clear()
            if d.tipo == "mover" and self.colonia.es_conjetura(d.accion):
                frente, _ = self.colonia.frente(antes, mascara, d.accion)
                regla = self.colonia.regla(f"contacto:{frente}") if frente is not None else None
                if regla == ATRAVIESA or frente is None or frente == fondo(antes):
                    self.colonia.refutadas.add(d.accion)
        return t

    @staticmethod
    def _salio_como_se_esperaba(d: Decision, ef, nodo_d: Nodo) -> bool:
        if d.tipo == "mover":
            return tuple(d.esperado) in ef.mueve
        if d.tipo == "grafo":
            return nodo_d.id == d.esperado
        return True

    # ------------------------------------------------------------------ elección

    def _elegir(self, grid: np.ndarray, mascara: np.ndarray, nodo: Nodo, disponibles: list[int]) -> Decision:
        if self.plan:
            d = self.plan.popleft()
            if not nodo.fatal(d.clave):
                return d
            self.plan.clear()

        plan = self._plan_a_meta(grid, mascara, nodo)
        if plan:
            self.plan.extend(plan[1:])
            return plan[0]

        candidatas = self._candidatas(nodo, disponibles)
        prio = {k: self._prioridad(k, nodo.cuadro, mascara, nodo) for k in candidatas}
        pendientes = [k for k in candidatas if not self.grafo.probada(nodo, k) and prio[k] > 0]
        if pendientes and not self.novedad.get(nodo.id, True):
            # Estado sin nada nuevo: si cerca hay uno novedoso con acciones por probar,
            # primero ese (IW(1) como prioridad, sin descartar nada).
            novedoso = lambda n: self.novedad.get(n.id, True) and self._tiene_pendientes(n, disponibles, mascara, 1)  # noqa: E731
            camino = self.grafo.camino(nodo, novedoso)
            if camino and len(camino) <= MAX_DESVIO_NOVEDAD and not self._sin_presupuesto(len(camino)):
                return self._seguir_camino(nodo, camino, "ir a un estado novedoso")
        if pendientes:
            k = self._mejor(pendientes, prio)
            return Decision.desde_clave(k, motivo=f"probar (prioridad {prio[k]})")

        for umbral in (1, 0):
            meta = lambda n, u=umbral: self._tiene_pendientes(n, disponibles, mascara, u)  # noqa: E731
            camino = self.grafo.camino(nodo, meta)
            reinicio = self._reiniciar_conviene(nodo, camino, meta)
            if reinicio is not None:
                return reinicio
            if camino:
                return self._seguir_camino(nodo, camino, "ir a un estado con acciones sin probar")
            if umbral == 1:
                bajas = [k for k in candidatas if not self.grafo.probada(nodo, k) and prio[k] == 0]
                if bajas:
                    k = self._mejor(bajas, prio)
                    return Decision.desde_clave(k, motivo="probar lo que la colonia cree inútil")

        seguras = [k for k in candidatas if prio[k] >= 0] or candidatas
        if not seguras and self.paso_vida == 0 and candidatas:
            seguras = candidatas
        if not seguras:
            # Sin ninguna acción conocida: un click al centro si se puede clickear, si no RESET.
            if 6 in disponibles:
                alto, ancho = grid.shape
                return Decision("ACTION6", ancho // 2, alto // 2, motivo="sin candidatas: click al centro")
            return Decision("RESET", motivo="sin candidatas: reiniciar")
        # Lo que ya sabemos que deja todo igual no aporta nada: repetirlo era un bucle.
        se_mueven = [k for k in seguras if (dst := self.grafo.destino(nodo, k)) is None or dst.id != nodo.id]
        k = self.rng.choice(se_mueven or seguras)
        return Decision.desde_clave(k, motivo="todo probado: al azar")

    def _registrar_novedad(self, nodo: Nodo, grid: np.ndarray, mascara: np.ndarray) -> None:
        if nodo.id in self.novedad:
            return
        atomos = {(o.color, o.x0, o.y0, o.forma) for o in segmentar(grid, mascara) if not o.es_marco}
        nuevos = atomos - self.atomos
        self.atomos |= atomos
        self.novedad[nodo.id] = bool(nuevos)

    def _sin_presupuesto(self, pasos: int) -> bool:
        tope = self.presupuesto()
        return tope is not None and self.paso_vida + pasos + MARGEN_PRESUPUESTO > tope

    def presupuesto(self) -> int | None:
        """Pasos que da una vida, si las muertes muestran un contador regular: al menos
        dos vidas que murieron con el mismo largo (±10 %), el máximo observado. Muertes
        de largo irregular (lava, choques) no definen presupuesto."""
        if len(self.vidas) < 2:
            return None
        tope = max(self.vidas)
        cerca = [v for v in self.vidas if v >= 0.9 * tope]
        if len(cerca) < 2 or tope < 4:
            return None
        if self.paso_vida > tope:  # esta vida ya duró más: hubo recarga, no hay límite fijo
            return None
        return tope

    def _reiniciar_conviene(self, nodo: Nodo, camino: list[str] | None, meta) -> Decision | None:
        """RESET más el camino más corto desde la raíz, cuando eso es mejor que caminar:
        más corto, o el destino no entra en lo que le queda a esta vida, o desde acá ya
        no se llega a nada por probar (cambios irreversibles)."""
        if self.paso_vida == 0 or self.raiz is None or nodo.id == self.raiz:
            return None
        raiz = self.grafo.nodos.get(self.raiz)
        if raiz is None:
            return None
        desde_raiz = self.grafo.camino(raiz, meta)
        if desde_raiz is None:
            return None
        tope = self.presupuesto()
        if tope is not None and len(desde_raiz) + MARGEN_PRESUPUESTO > tope:
            return None  # tampoco llega reiniciando
        if camino is not None:
            no_llega = tope is not None and self.paso_vida + len(camino) + MARGEN_PRESUPUESTO > tope
            if not no_llega and 1 + len(desde_raiz) >= len(camino):
                return None
        self._plan_tras_reset = self._decisiones_camino(raiz, desde_raiz, "reiniciar: camino más corto desde el inicio")
        return Decision("RESET", motivo=f"reiniciar ({1 + len(desde_raiz)} pasos contra "
                                       f"{len(camino) if camino is not None else 'sin camino'})")

    def _decisiones_camino(self, nodo: Nodo, camino: list[str], motivo: str) -> list[Decision]:
        decisiones = []
        actual = nodo
        for k in camino:
            sig = self.grafo.destino(actual, k)
            decisiones.append(Decision.desde_clave(k, motivo=motivo, tipo="grafo", esperado=sig.id if sig else None))
            actual = sig
        return decisiones

    def _cualquiera(self, nodo: Nodo, disponibles: list[int]) -> str:
        cand = self._candidatas(nodo, disponibles) or ["ACTION1"]
        return self.rng.choice(cand)

    @staticmethod
    def _clicks_sonda(grid: np.ndarray, mascara: np.ndarray | None, paso: int = 8) -> list[str]:
        alto, ancho = grid.shape
        out = []
        for y in range(paso // 2, alto, paso):
            for x in range(paso // 2, ancho, paso):
                if mascara is not None and mascara.shape == grid.shape and mascara[y, x]:
                    continue
                out.append(f"ACTION6@{x},{y}")
        return out[:MAX_CLICKS]

    def _mejor(self, claves: list[str], prio: dict[str, int]) -> str:
        orden = list(claves)
        self.rng.shuffle(orden)
        return max(orden, key=lambda k: prio[k])

    def _seguir_camino(self, nodo: Nodo, camino: list[str], motivo: str) -> Decision:
        decisiones = []
        actual = nodo
        for k in camino:
            sig = self.grafo.destino(actual, k)
            decisiones.append(Decision.desde_clave(k, motivo=motivo, tipo="grafo", esperado=sig.id if sig else None))
            actual = sig
        self.plan.extend(decisiones[1:])
        return decisiones[0]

    def _tiene_pendientes(self, n: Nodo, disponibles: list[int], mascara: np.ndarray, umbral: int) -> bool:
        for k in self._candidatas(n, disponibles):
            if not self.grafo.probada(n, k) and self._prioridad(k, n.cuadro, mascara, n) >= umbral:
                return True
        return False

    def _candidatas(self, nodo: Nodo, disponibles: list[int]) -> list[str]:
        cache = self._candidatas_cache.get(nodo.id)
        if cache is not None:
            return cache
        out = [f"ACTION{i}" for i in disponibles if i not in (0, 6)]
        if 6 in disponibles:
            objetos = [o for o in segmentar(nodo.cuadro, self.ruido.mascara) if not o.es_marco]
            for o in self._ordenar_para_click(objetos)[:MAX_CLICKS]:
                x, y = o.celda_interior()
                out.append(f"ACTION6@{x},{y}")
            if not objetos:
                # Juego de solo clicks sin objetos segmentables (todo enmascarado o un único color):
                # sondear una grilla de celdas para no quedarse sin acciones.
                out.extend(self._clicks_sonda(nodo.cuadro, self.ruido.mascara))
        self._candidatas_cache[nodo.id] = out
        return out

    @staticmethod
    def _ordenar_para_click(objetos: list) -> list:
        """Primero un representante de cada tipo (color y tamaño de caja), los tipos más
        raros adelante; después el resto. Con más objetos que MAX_CLICKS, cortar por
        tamaño dejaba afuera los botones (lp85, nivel 2): lo que se repite en patrón casi
        nunca es un botón, lo distinto suele serlo."""
        def tipo(o):
            return (o.color, o.x1 - o.x0, o.y1 - o.y0)

        cuenta: dict = {}
        for o in objetos:
            cuenta[tipo(o)] = cuenta.get(tipo(o), 0) + 1
        orden = sorted(objetos, key=lambda o: (cuenta[tipo(o)], o.tamanio, o.y0, o.x0))
        primeros, resto, vistos = [], [], set()
        for o in orden:
            (resto if tipo(o) in vistos else primeros).append(o)
            vistos.add(tipo(o))
        return primeros + resto

    def _prioridad(self, k: str, grid: np.ndarray, mascara: np.ndarray, nodo: Nodo) -> int:
        """-1 descartada (fatal), 0 inútil según la colonia, 1-3 de menos a más prometedora."""
        if nodo.fatal(k):
            return -1
        col = self.colonia
        if "@" in k:
            x, y = (int(v) for v in k.split("@")[1].split(","))
            color = int(grid[y, x])
            r = col.ranuras.get(f"click:{color}")
            if r is None or not r.evidencia:
                return 3
            if r.regla == FATAL:
                return -1
            if color in col.colores_meta():
                return 4
            if r.regla == CLICK_INERTE:
                return 0 if r.usable else 1
            return 3  # efecto conocido (recolorea, quita, ...): probable avance
        frente, sale = col.frente(grid, mascara, k)
        if frente is not None:
            rc = col.ranuras.get(f"contacto:{frente}")
            if rc is not None and rc.regla == FATAL:
                return -1
            if rc is not None and rc.regla == BLOQUEA and rc.estado == "healthy":
                return 0
        if sale:
            return 0
        r = col.ranuras.get(f"accion:{k}")
        if r is None or not r.evidencia:
            return 3
        if r.regla == INERTE:
            return 0 if r.usable else 1
        return 2

    # ------------------------------------------------------------------ metas con avatar

    def _plan_a_meta(self, grid: np.ndarray, mascara: np.ndarray, nodo: Nodo) -> list[Decision] | None:
        col = self.colonia
        avatar = col.avatar()
        if avatar is None:
            return None
        movs = col.desplazamientos(grid, mascara)
        movs = {a: (dx, dy) for a, (c, dx, dy) in movs.items() if c == avatar and not nodo.fatal(a)}
        ys, xs = np.nonzero((grid == avatar) & ~mascara)
        if len(xs) == 0 or not movs:
            return None
        cuerpo = list(zip(xs.tolist(), ys.tolist()))

        metas_de: dict[tuple[int, int], int] = {}
        metas: list[tuple[int, Objeto]] = []
        prioridad_color = self._prioridad_de_contacto()
        transitable = (grid == fondo(grid)) | mascara | (grid == avatar)
        for nombre, r in col.ranuras.items():
            if not nombre.startswith("contacto:") or r.regla is None:
                continue
            color = int(nombre.split(":")[1])
            if r.regla == ATRAVIESA and r.estado == "healthy":
                transitable |= grid == color
            elif r.regla in (BLOQUEA, FATAL) or r.estado == "quarantined":
                transitable &= grid != color
        for o in segmentar(grid, mascara):
            if o.color == avatar or o.es_marco:
                continue
            p = prioridad_color(o.color)
            ident = (o.color, o.x0, o.y0, o.tamanio)
            if p is None or (nodo.id, ident) in self.tocados or self.toques.get(ident, 0) >= MAX_TOQUES:
                continue
            for c in o.celdas:
                metas_de[c] = len(metas)
            metas.append((p, o))
        if not metas:
            return None

        alto, ancho = grid.shape
        acciones = sorted(movs.items())
        previo: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        cola = deque([(0, 0)])
        vistos = {(0, 0)}
        encontradas: dict[int, tuple[int, list[str]]] = {}

        def reconstruir(pos: tuple[int, int]) -> list[str]:
            pasos = []
            while pos != (0, 0):
                pos, a = previo[pos]
                pasos.append(a)
            return pasos[::-1]

        while cola and len(vistos) < LIMITE_BFS:
            pos = cola.popleft()
            for a, (dx, dy) in acciones:
                nx, ny = pos[0] + dx, pos[1] + dy
                celdas = [(x + nx, y + ny) for x, y in cuerpo]
                if any(not (0 <= x < ancho and 0 <= y < alto) for x, y in celdas):
                    continue
                tocadas = {metas_de[c] for c in celdas if c in metas_de}
                if tocadas:
                    camino = reconstruir(pos) + [a]
                    for i in tocadas:
                        if i not in encontradas:
                            encontradas[i] = (len(camino), camino)
                    continue
                if (nx, ny) in vistos or not all(transitable[y, x] for x, y in celdas):
                    continue
                vistos.add((nx, ny))
                previo[(nx, ny)] = (pos, a)
                cola.append((nx, ny))
        if not encontradas:
            return None
        i, (largo, camino) = min(encontradas.items(), key=lambda kv: (metas[kv[0]][0], kv[1][0], kv[0]))
        p, objetivo = metas[i]
        ident = (objetivo.color, objetivo.x0, objetivo.y0, objetivo.tamanio)
        motivo = f"tocar color {objetivo.color} (prioridad {p}, {largo} pasos)"
        plan = [
            Decision(a, motivo=motivo, tipo="mover", esperado=(avatar, *movs[a])) for a in camino[:-1]
        ]
        plan.append(Decision(camino[-1], motivo=motivo, tipo="tocar", meta=ident))
        return plan

    def _aprender_meta(self, antes: np.ndarray, mascara: np.ndarray, d: Decision) -> None:
        """La jugada ganadora proyectada sobre el último cuadro del nivel (el cuadro que
        devuelve el motor ya es el nivel siguiente): qué pisaría el avatar ahí."""
        col = self.colonia
        avatar = col.avatar()
        if avatar is None:
            return
        mov = col.desplazamientos(antes, mascara).get(d.accion)
        if mov is None or mov[0] != avatar:
            return
        _, dx, dy = mov
        bg = fondo(antes)
        alto, ancho = antes.shape
        abundantes = {int(c) for c, n in zip(*np.unique(antes, return_counts=True)) if n > 0.15 * antes.size}
        atraviesa = {int(k.split(":")[1]) for k, r in col.ranuras.items()
                     if k.startswith("contacto:") and r.regla == ATRAVIESA}
        pisa: dict[int, int] = {}
        for y, x in zip(*np.nonzero(antes == avatar)):
            X, Y = int(x) + dx, int(y) + dy
            if not (0 <= X < ancho and 0 <= Y < alto) or mascara[Y, X]:
                continue
            v = int(antes[Y, X])
            if v in (bg, avatar) or v in abundantes or v in atraviesa:
                continue
            pisa[v] = pisa.get(v, 0) + 1
        if pisa:
            meta = max(pisa.items(), key=lambda kv: kv[1])[0]
            self.metas_color[meta] = self.metas_color.get(meta, 0) + 1

    def _prioridad_de_contacto(self):
        col = self.colonia
        # La meta aprendida al ganar manda sobre la atribución por contacto de la colonia,
        # que puede quedarse con una pared vecina (ls20: anotaba 4 en vez de 3).
        metas = list(self.metas_color) or col.colores_meta()

        def prioridad(color: int) -> int | None:
            if color in metas:
                return 0
            r = col.ranuras.get(f"contacto:{color}")
            if r is None or not r.evidencia:
                return 1
            if r.regla == ACTIVA and r.estado == "healthy":
                return 2
            return None

        return prioridad

    def exportar(self) -> dict:
        return {
            "acciones": self.n,
            "nivel": self.nivel.indice,
            "habilidades": {str(k): v for k, v in self.habilidades.items()},
            "colonia": self.colonia.exportar(),
            "ultimas_decisiones": list(self.decisiones)[-40:],
        }
