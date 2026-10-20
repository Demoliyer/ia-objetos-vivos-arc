"""Bucle de juego sobre cualquier entorno con ``reset()`` / ``step(action, data)``.

Sirve para el kit oficial (``arc_agi`` EnvironmentWrapper) y para el banco
local (``vivos.juegos.EntornoLocal``).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .agente import AgenteVivo, Decision, _estado


@dataclass
class Resultado:
    juego: str
    niveles: int = 0
    win_levels: int = 0
    acciones: int = 0
    reinicios: int = 0
    estado: str = ""
    acciones_por_nivel: list[int] = field(default_factory=list)
    segundos: float = 0.0

    @property
    def gano(self) -> bool:
        return self.estado == "WIN"

    def linea(self) -> str:
        return (
            f"{self.juego}: {self.niveles}/{self.win_levels} niveles, {self.acciones} acciones "
            f"(por nivel {self.acciones_por_nivel}), {self.reinicios} reinicios, {self.estado}, {self.segundos:.1f} s"
        )


def _accion(nombre: str):
    from arcengine import GameAction

    return GameAction[nombre]


class AgenteAleatorio:
    """Línea de base: acciones disponibles al azar; clicks en cualquier celda."""

    def __init__(self, semilla: int = 0) -> None:
        self.rng = random.Random(semilla)

    def decidir(self, obs) -> Decision:
        disp = [a for a in (obs.available_actions or [1, 2, 3, 4, 5, 6]) if a != 0]
        a = self.rng.choice(disp)
        if a == 6:
            return Decision("ACTION6", self.rng.randrange(64), self.rng.randrange(64))
        return Decision(f"ACTION{a}")

    def aprender(self, *_):
        return None

    def reiniciado(self) -> None:
        pass


def jugar(
    entorno,
    agente=None,
    *,
    max_acciones: int = 1000,
    juego: str = "",
    alguien_mira=None,
    segundos_max: float | None = None,
) -> Resultado:
    """Juega hasta ganar, agotar ``max_acciones`` o ``segundos_max``, o que el
    entorno falle. Al agotar el tiempo el estado queda ``TIEMPO``: con plazo de
    evaluación (Kaggle) es mejor cortar un juego que perder la corrida entera.

    Cada RESET cuenta como acción (así lo cuenta la competencia).
    """
    agente = agente or AgenteVivo()
    res = Resultado(juego=juego)
    inicio = time.monotonic()
    obs = entorno.reset()
    if obs is None:
        res.estado = "ERROR"
        return res
    res.win_levels = obs.win_levels
    desde = 0
    while res.acciones < max_acciones:
        if segundos_max is not None and time.monotonic() - inicio > segundos_max:
            res.estado = "TIEMPO"
            break
        estado = _estado(obs)
        if estado == "WIN":
            break
        if estado == "GAME_OVER" or not len(getattr(obs, "frame", []) or []):
            obs = entorno.step(_accion("RESET"))
            res.acciones += 1
            res.reinicios += 1
            agente.reiniciado()
            if obs is None:
                res.estado = "ERROR"
                break
            continue
        d = agente.decidir(obs)
        nueva = entorno.step(_accion(d.accion), d.datos)
        res.acciones += 1
        if nueva is None:
            res.estado = "ERROR"
            break
        agente.aprender(obs, d, nueva)
        if nueva.levels_completed > obs.levels_completed:
            res.acciones_por_nivel.append(res.acciones - desde)
            desde = res.acciones
        if alguien_mira is not None:
            alguien_mira(res, d, nueva)
        obs = nueva
    res.niveles = obs.levels_completed if obs is not None else res.niveles
    res.win_levels = getattr(obs, "win_levels", res.win_levels) or res.win_levels
    if not res.estado:
        res.estado = _estado(obs)
    res.segundos = time.monotonic() - inicio
    return res
