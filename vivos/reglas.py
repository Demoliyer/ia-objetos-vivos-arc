"""La colonia: reglas vivas verificadas contra la evidencia observada.

Ver ``docs/IA_OBJETOS_VIVOS.md`` §4. La evidencia (cada ``Transicion``) es de
confianza e inmutable; una regla es una etiqueta no confiable que predice qué
pasa en su ranura. Una regla se instala solo si no contradice ninguna evidencia
de su ranura; si una transición real la contradice, la ranura se degrada y se
repara con candidatas sacadas de esa transición.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np

from .percepcion import Efecto, clave, efecto, segmentar, tamanio_de

HEALTHY = "healthy"
DEGRADED = "degraded"
QUARANTINED = "quarantined"
MAX_REPARACIONES = 3
# Anomalías toleradas: una regla con muchas confirmaciones no cae por una
# contradicción rara (el avatar que reaparece en el inicio al perder una vida
# parece un salto). Sigue verificada si la confirman al menos ANOMALIAS_MIN
# estados distintos y las contradicciones no pasan de 1 cada ANOMALIAS_RATIO.
ANOMALIAS_MIN = 8
ANOMALIAS_RATIO = 10


def sostenible(malas: int, buenas: int) -> bool:
    return malas == 0 or (buenas >= ANOMALIAS_MIN and malas * ANOMALIAS_RATIO <= buenas)

# Etiquetas de reglas. Una regla es una etiqueta; su significado está en ``juzgar``.
INERTE = "inerte"
CAMBIA = "cambia"
FATAL = "fatal"
ACTIVA = "activa"
BLOQUEA = "bloquea"
ATRAVIESA = "atraviesa"
CLICK_INERTE = "inerte"
QUITA = "quita"
FIN = "fin"  # la partida terminó o subió de nivel: no dice nada de la acción
MUERTE = "muerte"  # GAME_OVER durante un contacto: sostiene "fatal", no contradice otra cosa
OTRO = "otro"

# Confirmaciones mínimas para que una regla sana se use al decidir.
_MINIMO = {INERTE: 3, BLOQUEA: 2, ATRAVIESA: 2, FATAL: 1, ACTIVA: 1, CAMBIA: 1, QUITA: 1, OTRO: 1}


def minimo_de(regla: str, ranura: str = "") -> int:
    if ranura.startswith("click:") and regla == CLICK_INERTE:
        return 2
    return _MINIMO.get(regla.split(":")[0], 1)


@dataclass
class Transicion:
    """Evidencia: una acción real y lo que el juego devolvió.

    Los cuadros, la acción y el desenlace son inmutables. La interpretación
    (máscara de ruido, efecto, huellas de estado) se recalcula cuando la
    percepción aprende algo nuevo, como un reloj: la evidencia no cambia, cambia
    cómo se la lee."""

    n: int
    accion: str  # "ACTION1" ... "ACTION7"
    click: tuple[int, int] | None
    antes: np.ndarray
    despues: np.ndarray
    mascara: np.ndarray  # ruido vigente al registrarla
    efecto: Efecto
    game_over: bool
    sube_nivel: bool
    clave_antes: str
    clave_despues: str
    nivel: int = 0

    @property
    def click_color(self) -> int | None:
        if self.click is None:
            return None
        x, y = self.click
        if self.mascara[y, x]:
            return None
        return int(self.antes[y, x])


@dataclass(frozen=True)
class Observacion:
    """Lo que una transición dice sobre una ranura."""

    etiqueta: str  # "inerte", "mueve:8:0:-4", "bloquea", ...
    alternativas: tuple[str, ...] = ()  # otras etiquetas que la misma transición sostiene

    def etiquetas(self) -> tuple[str, ...]:
        return (self.etiqueta, *self.alternativas)


def juzgar(regla: str, obs: Observacion) -> int:
    """+1 confirma, -1 contradice, 0 no aplica."""
    if obs.etiqueta == FIN:
        return 1 if regla in obs.alternativas else 0
    if obs.etiqueta == MUERTE:
        # La partida puede terminar por otra causa (energía, tiempo) justo al tocar
        # algo: confirma "fatal" pero no refuta lo que ya se sabía del contacto.
        return 1 if regla == FATAL else 0
    if regla.startswith("mueve:"):
        color = regla.split(":")[1]
        if regla in obs.etiquetas():
            return 1
        if obs.etiqueta in (FATAL, ACTIVA):
            return 0
        if any(e.startswith(f"mueve:{color}:") for e in obs.etiquetas()):
            return -1  # el mismo color se movió distinto
        if obs.etiqueta == INERTE:
            return 0  # intento bloqueado
        return 0
    if regla == CAMBIA:
        if obs.etiqueta == INERTE:
            return 0
        return 1
    return 1 if regla in obs.etiquetas() else -1


@dataclass
class Ranura:
    """Identidad estable de una regla: ``accion:ACTION1``, ``contacto:9``, ``click:2``."""

    nombre: str
    regla: str | None = None
    lkg: str | None = None
    estado: str = HEALTHY
    confirmaciones: int = 0
    reparaciones_fallidas: int = 0
    anomalias: int = 0  # contradicciones toleradas (ver ``sostenible``)
    evidencia: list[tuple[int, Observacion, str]] = field(default_factory=list)  # (n, obs, estado)
    historial: list[str] = field(default_factory=list)
    subidas_de_nivel: int = 0

    @property
    def usable(self) -> bool:
        return (
            self.regla is not None
            and self.estado == HEALTHY
            and self.confirmaciones >= minimo_de(self.regla, self.nombre)
        )

    def verificar(self, regla: str) -> tuple[int, int]:
        """(contradicciones, confirmaciones) de una regla contra toda la evidencia.

        Las confirmaciones cuentan estados distintos: repetir la misma acción en
        el mismo estado no agrega evidencia nueva."""
        malas = 0
        estados: set[str] = set()
        for _, obs, estado in self.evidencia:
            v = juzgar(regla, obs)
            if v < 0:
                malas += 1
            elif v > 0:
                estados.add(estado)
        return malas, len(estados)

    def agregar(self, n: int, obs: Observacion, estado: str = "") -> None:
        estado = estado or f"t{n}"
        self.evidencia.append((n, obs, estado))
        if obs.etiqueta == ACTIVA and "nivel" in obs.alternativas:
            self.subidas_de_nivel += 1
        if self.regla is None:
            self._reparar(obs, f"t{n}: primera evidencia")
            return
        v = juzgar(self.regla, obs)
        if v > 0 and not any(e == estado and juzgar(self.regla, o) > 0 for _, o, e in self.evidencia[:-1]):
            self.confirmaciones += 1
        if v < 0 and self.estado == HEALTHY:
            malas, buenas = self.verificar(self.regla)
            if sostenible(malas, buenas):
                self.anomalias += 1
                if self.anomalias <= 3:
                    self.historial.append(f"t{n}: anomalía tolerada '{obs.etiqueta}' ({malas} contra {buenas})")
                return
            self.historial.append(f"t{n}: '{self.regla}' contradicha por '{obs.etiqueta}' -> degraded")
            self.estado = DEGRADED
            self._reparar(obs, f"t{n}: reparación")
        elif self.estado == DEGRADED:
            # Degradada: cada evidencia nueva es otra oportunidad de reparar.
            self._reparar(obs, f"t{n}: reintento")
        elif self.regla == CAMBIA and len(self.evidencia) % 8 == 0 and any(e.startswith("mueve:") for e in obs.etiquetas()):
            # "cambia" nunca se contradice: sin esto, una regla que cayó temprano a lo
            # genérico no vuelve a afinarse aunque la evidencia ya sostenga un "mueve".
            self._reparar(obs, f"t{n}: refinamiento")

    def _reparar(self, obs: Observacion, motivo: str) -> None:
        candidatas = [e for e in obs.etiquetas() if e not in ("nivel", FIN)]
        if obs.etiqueta == MUERTE:
            candidatas = [FATAL]
        # La regla vigente y la última buena también compiten: una anomalía no las saca de la carrera.
        candidatas += [r for r in (self.regla, self.lkg) if r is not None]
        if obs.etiqueta == INERTE and not self.nombre.startswith("contacto:"):
            candidatas.append(INERTE)
        if any(e.startswith("mueve:") for e in candidatas) or obs.etiqueta == OTRO:
            candidatas.append(CAMBIA)
        mejores = []
        for c in dict.fromkeys(candidatas):
            malas, buenas = self.verificar(c)
            if sostenible(malas, buenas):
                # Más confirmaciones primero; después menos anomalías, la más específica
                # (mueve > cambia) y la más corta.
                # En empate gana la etiqueta principal de la observación (el objeto que
                # más se movió), no el orden alfabético (tu93: el indicador 4 le ganaba al 9).
                mejores.append((-buenas, malas, c == CAMBIA, (c != obs.etiqueta, len(c)), c, buenas))
        # "cambia" junta todo lo que no es inerte y siempre suma más; si una regla
        # específica explica al menos la mitad de lo mismo, gana la específica.
        especificas = [m for m in mejores if m[4] != CAMBIA]
        if especificas:
            tope = max(m[5] for m in especificas)
            mejores = [m for m in mejores if m[4] != CAMBIA or tope * 2 < m[5]]
        if not mejores:
            if self.regla is not None:
                self.reparaciones_fallidas += 1
                self.historial.append(f"{motivo}: sin candidata verificada ({self.reparaciones_fallidas}/{MAX_REPARACIONES})")
                if self.reparaciones_fallidas >= MAX_REPARACIONES:
                    self.estado = QUARANTINED
                    self.historial.append(f"{motivo}: quarantined")
            return
        mejores.sort()
        nueva, buenas = mejores[0][4], mejores[0][5]
        if nueva != self.regla:
            if self.regla is not None:
                self.lkg = self.regla
            self.historial.append(f"{motivo}: instala '{nueva}' ({buenas} confirmaciones)")
        self.regla = nueva
        self.confirmaciones = buenas
        self.estado = HEALTHY
        self.reparaciones_fallidas = 0

    def exportar(self) -> dict:
        return {
            "regla": self.regla,
            "estado": self.estado,
            "usable": self.usable,
            "confirmaciones": self.confirmaciones,
            "evidencia": len(self.evidencia),
            "lkg": self.lkg,
            "subidas_de_nivel": self.subidas_de_nivel,
            "historial": self.historial[-12:],
        }


class Colonia:
    """Población de reglas vivas más el diario de evidencia."""

    def __init__(self) -> None:
        self.diario: list[Transicion] = []
        self.ranuras: dict[str, Ranura] = {}
        self._avatar_firma: tuple = ()
        self._tamanios: dict[tuple[str, int], tuple[int, int] | None] = {}
        self._tam_visto: dict[int, dict[tuple[int, int], int]] = {}  # color -> tamaño al moverse -> veces
        self.refutadas: set[str] = set()

    # ------------------------------------------------------------------ evidencia

    def ranura(self, nombre: str) -> Ranura:
        if nombre not in self.ranuras:
            self.ranuras[nombre] = Ranura(nombre)
        return self.ranuras[nombre]

    def registrar(self, t: Transicion) -> None:
        self.diario.append(t)
        for (c, _, _), tam in zip(t.efecto.mueve, t.efecto.tam_mueve):
            cuenta = self._tam_visto.setdefault(c, {})
            cuenta[tam] = cuenta.get(tam, 0) + 1
        self._aplicar(t)

    def tamanio_avatar(self, grid: np.ndarray, color: int, mascara: np.ndarray) -> tuple[int, int] | None:
        """La unidad de movimiento de ``color``: el tamaño con el que más se lo vio
        moverse, si hay un objeto así en pantalla; si no, el más grande (otro nivel
        puede cambiar la escala). El más grande a secas fallaba cuando otro objeto del
        mismo color es mayor (ls20: un ícono de referencia; pasos del doble)."""
        visto = self._tam_visto.get(color)
        if visto:
            objs = [o for o in segmentar(grid, mascara, incluir_fondo=True) if o.color == color]
            presentes = {(o.x1 - o.x0 + 1, o.y1 - o.y0 + 1) for o in objs}
            for tam, _ in sorted(visto.items(), key=lambda kv: -kv[1]):
                if tam in presentes:
                    return tam
        return tamanio_de(grid, color, mascara)

    def reinterpretar(self, nivel: int, mascara: np.ndarray) -> None:
        """La percepción cambió (nuevo reloj): se relee la evidencia del nivel y
        todas las ranuras se reconstruyen desde el diario."""
        for t in self.diario:
            if t.nivel != nivel:
                continue
            t.mascara = mascara.copy()
            subio = t.sube_nivel
            t.efecto = efecto(t.antes, t.despues, mascara)
            if subio:
                t.efecto.otro = True
            t.clave_antes = clave(t.antes, mascara)
            t.clave_despues = clave(t.despues, mascara) if t.clave_despues else ""
        viejas = self.ranuras
        self.ranuras = {}
        self._avatar_firma = ()
        for t in self.diario:
            self._aplicar(t, reindexar=False)
        self._avatar_firma = self._firma_avatar()
        self._reindexar_contactos()
        for k, r in self.ranuras.items():
            if k in viejas and viejas[k].regla != r.regla:
                r.historial.insert(0, f"reinterpretada: '{viejas[k].regla}' -> '{r.regla}' (nuevo ruido)")

    def _aplicar(self, t: Transicion, reindexar: bool = True) -> None:
        if t.accion == "ACTION6":
            color = t.click_color
            if color is not None:
                self.ranura(f"click:{color}").agregar(t.n, self._obs_click(t, color), t.clave_antes)
        elif t.accion != "RESET":
            self.ranura(f"accion:{t.accion}").agregar(t.n, self._obs_accion(t), t.clave_antes)
        if not reindexar:
            return
        firma = self._firma_avatar()
        if firma != self._avatar_firma:
            self._avatar_firma = firma
            self._reindexar_contactos()
        else:
            self._contacto(t)

    @staticmethod
    def _desenlace(t: Transicion) -> str | None:
        if t.game_over:
            return FATAL
        if t.sube_nivel:
            return ACTIVA
        return None

    def _obs_accion(self, t: Transicion) -> Observacion:
        fin = self._desenlace(t)
        # El desplazamiento se guarda en unidades del tamaño del objeto: así la regla
        # sobrevive a un nivel con otra escala de cámara.
        movs = tuple(
            f"mueve:{c}:{Fraction(dx, w)}:{Fraction(dy, h)}"
            for (c, dx, dy), (w, h) in zip(t.efecto.mueve, t.efecto.tam_mueve)
        )
        if fin is not None:
            # El desenlace se le atribuye al contacto, no a la acción: no juzga reglas de acción.
            return Observacion(FIN, movs)
        if t.efecto.nada:
            return Observacion(INERTE)
        if movs:
            return Observacion(movs[0], movs[1:])
        return Observacion(OTRO)

    def _obs_click(self, t: Transicion, color: int) -> Observacion:
        fin = self._desenlace(t)
        if fin is not None:
            return Observacion(fin, ("nivel",) if t.sube_nivel else ())
        e = t.efecto
        if e.nada:
            return Observacion(CLICK_INERTE)
        for a, b in e.recolorea:
            if a == color:
                return Observacion(f"recolorea:{b}")
        if color in e.desaparece:
            return Observacion(QUITA)
        return Observacion(OTRO)

    # ------------------------------------------------------------------ avatar y contactos

    def movimientos(self, conjeturas: bool = True) -> dict[str, tuple[int, Fraction, Fraction]]:
        """Acción -> (color, fx, fy) según las reglas ``mueve`` sanas (aunque tengan 1
        confirmación). fx, fy en unidades del tamaño del objeto.

        Con ``conjeturas``, si tres de ACTION1-4 mueven al mismo color en tres
        direcciones opuestas dos a dos y la cuarta no tiene una regla usable, se
        conjetura la dirección que falta (la del espejo). Una conjetura no es una
        regla: no entra en la colonia, solo deja planificar; la primera vez que se
        usa es un experimento, y si falla con el frente libre queda refutada."""
        out = {}
        for nombre, r in self.ranuras.items():
            if nombre.startswith("accion:") and r.regla and r.regla.startswith("mueve:") and r.estado == HEALTHY:
                _, c, fx, fy = r.regla.split(":")
                out[nombre.split(":", 1)[1]] = (int(c), Fraction(fx), Fraction(fy))
        if conjeturas:
            for a, mov in self._conjeturas(out).items():
                out[a] = mov
        return out

    def _conjeturas(self, reales: dict) -> dict[str, tuple[int, Fraction, Fraction]]:
        flechas = [f"ACTION{i}" for i in range(1, 5)]
        conocidas = {a: m for a, m in reales.items() if a in flechas}
        faltan = [a for a in flechas if a not in conocidas and a not in self.refutadas]
        if len(conocidas) != 3 or len(faltan) != 1 or len({m[0] for m in conocidas.values()}) != 1:
            return {}
        a = faltan[0]
        r = self.ranuras.get(f"accion:{a}")
        if r is not None and (r.usable or (r.regla not in (None, INERTE))):
            return {}
        vectores = {(m[1], m[2]) for m in conocidas.values()}
        for fx, fy in list(vectores):
            if (-fx, -fy) not in vectores:
                color = next(iter(conocidas.values()))[0]
                return {a: (color, -fx, -fy)}
        return {}

    def es_conjetura(self, accion: str) -> bool:
        return accion in self.movimientos() and accion not in self.movimientos(conjeturas=False)

    def desplazamientos(
        self, grid: np.ndarray, mascara: np.ndarray, conjeturas: bool = True
    ) -> dict[str, tuple[int, int, int]]:
        """Acción -> (color, dx, dy) en píxeles para este cuadro."""
        out = {}
        huella = clave(grid, mascara)
        for a, (c, fx, fy) in self.movimientos(conjeturas).items():
            if (huella, c) not in self._tamanios:
                if len(self._tamanios) > 50000:
                    self._tamanios.clear()
                self._tamanios[(huella, c)] = self.tamanio_avatar(grid, c, mascara)
            tam = self._tamanios[(huella, c)]
            if tam is None:
                continue
            w, h = tam
            dx, dy = fx * w, fy * h
            if dx.denominator == 1 and dy.denominator == 1:
                out[a] = (c, int(dx), int(dy))
        return out

    def avatar(self) -> int | None:
        colores = Counter(c for c, _, _ in self.movimientos(conjeturas=False).values())
        return colores.most_common(1)[0][0] if colores else None

    def _firma_avatar(self) -> tuple:
        return tuple(sorted(self.movimientos(conjeturas=False).items()))

    def frente(self, grid: np.ndarray, mascara: np.ndarray, accion: str, conjeturas: bool = True) -> tuple[int | None, bool]:
        """Color con el que chocaría el avatar al hacer ``accion`` y si el frente sale del cuadro."""
        mov = self.desplazamientos(grid, mascara, conjeturas=conjeturas).get(accion)
        if mov is None:
            return None, False
        color, dx, dy = mov
        ys, xs = np.nonzero((grid == color) & ~mascara)
        if len(xs) == 0:
            return None, False
        propias = set(zip(xs.tolist(), ys.tolist()))
        colores: Counter = Counter()
        for x, y in propias:
            nx, ny = x + dx, y + dy
            if (nx, ny) in propias:
                continue
            if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
                return None, True
            if mascara[ny, nx]:
                continue
            colores[int(grid[ny, nx])] += 1
        if not colores:
            return None, False
        return colores.most_common(1)[0][0], False

    def _contacto(self, t: Transicion) -> None:
        if t.accion in ("ACTION6", "RESET"):
            return
        color, _ = self.frente(t.antes, t.mascara, t.accion, conjeturas=False)
        if color is None:
            return
        avatar = self.movimientos(conjeturas=False)[t.accion][0]
        fin = self._desenlace(t)
        if fin == FATAL:
            obs = Observacion(MUERTE)
        elif fin is not None:
            obs = Observacion(fin, ("nivel",) if t.sube_nivel else ())
        else:
            se_movio = any(c == avatar for c, _, _ in t.efecto.mueve)
            otros = (
                [m for m in t.efecto.mueve if m[0] != avatar]
                or t.efecto.recolorea
                or t.efecto.desaparece
                or t.efecto.aparece
                or t.efecto.otro
            )
            if otros:
                obs = Observacion(ACTIVA)
            elif se_movio:
                obs = Observacion(ATRAVIESA)
            elif t.efecto.nada:
                obs = Observacion(BLOQUEA)
            else:
                obs = Observacion(OTRO)
        self.ranura(f"contacto:{color}").agregar(t.n, obs, t.clave_antes)

    def _reindexar_contactos(self) -> None:
        """El modelo del avatar cambió: la evidencia de contactos se recalcula desde el diario."""
        viejas = {k: v for k, v in self.ranuras.items() if k.startswith("contacto:")}
        for k in viejas:
            del self.ranuras[k]
        for t in self.diario:
            self._contacto(t)
        for k, vieja in viejas.items():
            nueva = self.ranuras.get(k)
            if nueva is not None:
                nueva.historial = ["reindexada: cambió el modelo del avatar"] + nueva.historial

    # ------------------------------------------------------------------ consultas para decidir

    def regla(self, nombre: str) -> str | None:
        r = self.ranuras.get(nombre)
        return r.regla if r is not None and r.usable else None

    def probado(self, nombre: str) -> bool:
        return nombre in self.ranuras and bool(self.ranuras[nombre].evidencia)

    def colores_meta(self) -> list[int]:
        """Colores cuyo contacto hizo subir de nivel, de más a menos veces."""
        r = [
            (ra.subidas_de_nivel, int(k.split(":")[1]))
            for k, ra in self.ranuras.items()
            if k.startswith(("contacto:", "click:")) and ra.subidas_de_nivel
        ]
        return [c for _, c in sorted(r, reverse=True)]

    def exportar(self) -> dict:
        return {
            "transiciones": len(self.diario),
            "avatar": self.avatar(),
            "ranuras": {k: v.exportar() for k, v in sorted(self.ranuras.items())},
        }
