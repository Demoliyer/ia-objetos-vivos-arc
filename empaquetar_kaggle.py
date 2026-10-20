"""Empaqueta la IA de Objetos Vivos en un único ``my_agent.py`` para el kit de Kaggle.

    .venv/bin/python empaquetar_kaggle.py                      # escribe kaggle/my_agent.py
    .venv/bin/python empaquetar_kaggle.py --salida ../kaggle-starter/agent/my_agent.py

El kit oficial (github.com/arcprize/ARC-AGI-3-Kaggle-Starter) pega el contenido de
``agent/my_agent.py`` en el notebook de entrega: tiene que ser un solo archivo. Este
script mete los módulos de ``vivos/`` comprimidos dentro del archivo, que al
importarse los extrae a un directorio temporal y define ``MyAgent`` sobre el
adaptador de ``vivos/adaptador_agents.py``.

Protecciones propias de la entrega (no existen en el banco local):
- tope de tiempo por juego (MAX_SEGUNDOS): no se conoce el límite total de Kaggle y
  el set oculto tiene del orden de 110 juegos;
- si el agente lanza una excepción en un paso, se responde con una acción segura
  (nunca RESET: con 0 acciones en el nivel reinicia el juego entero) y se sigue.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import zlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
MODULOS = ["__init__", "agente", "arena", "grafo", "percepcion", "reglas", "puntaje", "adaptador_agents"]
MAX_ACCIONES = 2000
MAX_SEGUNDOS = 180

PLANTILLA = '''"""IA de Objetos Vivos para ARC-AGI-3 (ARC Prize 2026).

Generado por empaquetar_kaggle.py desde Demoliyer/living-mesh, commit {commit}.
No editar a mano: editar vivos/ y volver a empaquetar.

Una colonia de reglas chicas, legibles y verificadas contra lo observado, que
aprende cómo funciona cada juego experimentando. Sin red neuronal ni LLM; corre en CPU.
"""
from __future__ import annotations

import base64
import logging
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

_FUENTES = {fuentes}
_HUELLA = "{huella}"


def _materializar() -> None:
    raiz = Path(tempfile.gettempdir()) / f"objetos_vivos_{{_HUELLA}}"
    paquete = raiz / "vivos"
    paquete.mkdir(parents=True, exist_ok=True)
    for nombre, dato in _FUENTES.items():
        destino = paquete / f"{{nombre}}.py"
        if not destino.exists():
            destino.write_bytes(zlib.decompress(base64.b64decode(dato)))
    if str(raiz) not in sys.path:
        sys.path.insert(0, str(raiz))


_materializar()

from vivos.adaptador_agents import crear_clase  # noqa: E402

_log = logging.getLogger("objetos_vivos")
_Base = crear_clase(Agent, max_acciones={max_acciones})


class MyAgent(_Base):  # type: ignore[misc, valid-type]
    """IA de Objetos Vivos con las protecciones de la entrega."""

    MAX_ACTIONS = {max_acciones}
    MAX_SEGUNDOS = {max_segundos}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._t0 = time.monotonic()
        self._fallas = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if time.monotonic() - self._t0 > self.MAX_SEGUNDOS:
            return True
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        try:
            return super().choose_action(frames, latest_frame)
        except Exception:  # un paso que rompe no tumba el juego
            self._fallas += 1
            _log.exception("objetos_vivos: falla en choose_action (%d)", self._fallas)
            self._previa = self._decision = None
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                return GameAction.RESET
            disponibles = [a for a in (getattr(latest_frame, "available_actions", None) or [1]) if a != 0]
            accion = GameAction.from_id(disponibles[self._fallas % len(disponibles)])
            if accion.is_complex():
                accion.set_data({{"x": 32, "y": 32}})
            return accion
'''


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "desconocido"


def empaquetar(salida: Path) -> Path:
    fuentes = {}
    for m in MODULOS:
        codigo = (RAIZ / "vivos" / f"{m}.py").read_bytes()
        fuentes[m] = base64.b64encode(zlib.compress(codigo, 9)).decode()
    huella = hashlib.sha256("".join(fuentes[m] for m in MODULOS).encode()).hexdigest()[:12]
    lineas = ["{"] + [f'    "{m}": "{fuentes[m]}",' for m in MODULOS] + ["}"]
    texto = PLANTILLA.format(
        commit=_commit(), fuentes="\n".join(lineas), huella=huella,
        max_acciones=MAX_ACCIONES, max_segundos=MAX_SEGUNDOS,
    )
    compile(texto, str(salida), "exec")  # que al menos sea Python válido
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    return salida


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", type=Path, default=RAIZ / "kaggle" / "my_agent.py")
    args = p.parse_args(argv)
    ruta = empaquetar(args.salida)
    print(f"{ruta}: {ruta.stat().st_size // 1024} KB, commit {_commit()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
