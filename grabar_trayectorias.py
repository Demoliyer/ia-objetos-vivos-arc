"""Graba trayectorias de los juegos públicos para evaluar modelos del mundo offline.

    .venv/bin/python grabar_trayectorias.py --pasos 600 --salida trayectorias/
    .venv/bin/python grabar_trayectorias.py --juego ls20 tn36

Cada juego queda en ``<salida>/<juego>.npz`` con:
    cuadros  uint8 [N+1, 64, 64]  cuadro antes de cada acción y el último después
    acciones str   [N]            clave de la acción ("ACTION3", "ACTION6@12,40", "RESET")
    nivel    int16 [N]            niveles completados antes de la acción
    corte    bool  [N]            True si la transición cruza nivel, GAME_OVER o RESET:
                                  no es una transición "normal" del mundo y no se evalúa

Juega el agente actual (es la distribución de estados que va a ver el modelo nuevo).
Requiere environment_files/ (modo offline).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OPERATION_MODE", "offline")
logging.disable(logging.CRITICAL)


def grabar(gid: str, pasos: int, semilla: int) -> dict:
    import arc_agi

    from vivos.agente import AgenteVivo
    from vivos.arena import _accion, _estado
    from vivos.percepcion import cuadro

    arc = arc_agi.Arcade()
    env = arc.make(gid, seed=semilla)
    if env is None:
        raise RuntimeError(f"{gid}: no se pudo crear el entorno")
    obs = env.reset()
    agente = AgenteVivo(semilla)
    cuadros = [cuadro(obs.frame).astype(np.uint8)]
    acciones: list[str] = []
    niveles: list[int] = []
    cortes: list[bool] = []
    for _ in range(pasos):
        estado = _estado(obs)
        if estado == "WIN":
            break
        if estado == "GAME_OVER" or not len(getattr(obs, "frame", []) or []):
            nueva = env.step(_accion("RESET"))
            agente.reiniciado()
            clave, corte = "RESET", True
        else:
            d = agente.decidir(obs)
            nueva = env.step(_accion(d.accion), d.datos)
            if nueva is None:
                break
            agente.aprender(obs, d, nueva)
            clave = d.clave
            corte = (d.accion == "RESET" or nueva.levels_completed != obs.levels_completed
                     or _estado(nueva) == "GAME_OVER" or not len(getattr(nueva, "frame", []) or []))
        if nueva is None:
            break
        niveles.append(int(obs.levels_completed))
        acciones.append(clave)
        cortes.append(bool(corte))
        frame = getattr(nueva, "frame", []) or []
        cuadros.append(cuadro(frame).astype(np.uint8) if len(frame) else cuadros[-1].copy())
        obs = nueva
    return {
        "cuadros": np.stack(cuadros),
        "acciones": np.array(acciones),
        "nivel": np.array(niveles, dtype=np.int16),
        "corte": np.array(cortes, dtype=bool),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--juego", nargs="+", help="ids cortos (por defecto, todos los de environment_files/)")
    p.add_argument("--pasos", type=int, default=600)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--salida", type=Path, default=Path("trayectorias"))
    args = p.parse_args(argv)
    juegos = args.juego or sorted(os.listdir("environment_files"))
    args.salida.mkdir(parents=True, exist_ok=True)
    fallas = 0
    for gid in juegos:
        try:
            datos = grabar(gid, args.pasos, args.semilla)
        except Exception as exc:  # un juego que falla no corta el resto
            print(f"{gid}: ERROR {exc}", file=sys.stderr, flush=True)
            fallas += 1
            continue
        np.savez_compressed(args.salida / f"{gid}.npz", **datos)
        print(f"{gid}: {len(datos['acciones'])} transiciones, {int(datos['corte'].sum())} cortes", flush=True)
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
