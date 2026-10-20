"""Adaptador para el repositorio oficial ``arcprize/ARC-AGI-3-Agents``.

Ese repositorio (el que usan las plantillas de la competencia) define una clase
``Agent`` con dos métodos: ``is_done`` y ``choose_action``. Este módulo arma una
subclase que delega en ``AgenteVivo``. Uso, dentro de ARC-AGI-3-Agents:

    # agents/templates/objetos_vivos.py
    import sys; sys.path.insert(0, "/ruta/a/living-mesh")
    from vivos.adaptador_agents import crear_clase
    from ..agent import Agent
    ObjetosVivos = crear_clase(Agent)

y registrarla en ``agents/__init__.py`` como las demás plantillas
(``uv run main.py --agent=objetosvivos --game=ls20``).
"""

from __future__ import annotations

from typing import Any

from .agente import AgenteVivo, Decision, _estado


def crear_clase(base: type, max_acciones: int = 5000) -> type:
    from arcengine import GameAction

    class ObjetosVivos(base):  # type: ignore[misc, valid-type]
        """IA de Objetos Vivos: colonia de reglas verificadas contra lo observado."""

        MAX_ACTIONS = max_acciones

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.ia = AgenteVivo()
            self._previa: Any = None
            self._decision: Decision | None = None

        def is_done(self, frames: list, latest_frame: Any) -> bool:
            return _estado(latest_frame) == "WIN"

        def choose_action(self, frames: list, latest_frame: Any) -> Any:
            if self._previa is not None and self._decision is not None and latest_frame is not self._previa:
                if len(getattr(latest_frame, "frame", []) or []):
                    self.ia.aprender(self._previa, self._decision, latest_frame)
            self._previa = self._decision = None

            estado = _estado(latest_frame)
            if estado in ("NOT_PLAYED", "GAME_OVER", "") or not len(getattr(latest_frame, "frame", []) or []):
                if estado == "GAME_OVER":
                    self.ia.reiniciado()
                return GameAction.RESET

            d = self.ia.decidir(latest_frame)
            accion = GameAction[d.accion]
            if d.datos is not None:
                accion.set_data(d.datos)
            accion.reasoning = {"motivo": d.motivo}
            self._previa, self._decision = latest_frame, d
            return accion

    return ObjetosVivos
