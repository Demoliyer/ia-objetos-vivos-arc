# SPDX-License-Identifier: MIT-0
"""Convierte las partidas grabadas por el harness (artifacts/*_events.jsonl[.gz]) al formato
de trayectorias que mide evaluar_mundo.py: cuadros [N+1,64,64], acciones [N], corte [N].

    python experimentos/reglas/convertir.py <living-mesh-datos/salidas> <dir_salida>

Acciones: flechas/SPACE/UNDO quedan con su nombre; MOUSE(row=r, col=c) -> "ACTION6@c,r"
(x = columna, y = fila, como espera vivos/mundo.disparador). Corte = RESET, cambio de nivel
o fin de partida. Los tableros son de juegos ARC: la salida queda local (trayectorias/ está
en .gitignore), nunca se versiona.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

MOUSE = re.compile(r"MOUSE\(row=(\d+),\s*col=(\d+)\)")


def leer(p):
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def clave(accion):
    m = MOUSE.fullmatch(accion.strip())
    if m:
        return f"ACTION6@{int(m.group(2))},{int(m.group(1))}"
    return accion.strip()


def convertir(ruta):
    cuadros, acciones, cortes = [], [], []
    nivel_prev = estado_prev = None
    with leer(ruta) as f:
        for linea in f:
            e = json.loads(linea)
            if e.get("type") not in ("initial", "action"):
                continue
            tablero = np.array(json.loads(e["board"]) if isinstance(e["board"], str) else e["board"], dtype=np.uint8)
            if tablero.shape != (64, 64):
                continue
            nivel, estado = e.get("level"), e.get("state")
            if e["type"] == "initial" or not cuadros:
                cuadros = [tablero]
                nivel_prev, estado_prev = nivel, estado
                continue
            a = clave(e.get("action_display", ""))
            acciones.append(a)
            cortes.append(a == "RESET" or nivel != nivel_prev or estado != estado_prev
                          or estado in ("GAME_OVER", "WIN"))
            cuadros.append(tablero)
            nivel_prev, estado_prev = nivel, estado
    if not acciones:
        return None
    return {"cuadros": np.stack(cuadros), "acciones": np.array(acciones), "corte": np.array(cortes, dtype=bool)}


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    base, dest = Path(argv[1]), Path(argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for ev in sorted(base.glob("*/artifacts/*_events.jsonl*")):
        corrida = ev.parts[-3]
        juego = ev.name.split("-")[0]
        d = convertir(ev)
        if d is None:
            continue
        nombre = f"{juego}__{corrida}"
        np.savez_compressed(dest / f"{nombre}.npz", **d)
        total += len(d["acciones"])
        print(f"{nombre}: {len(d['acciones'])} transiciones, {int(d['corte'].sum())} cortes")
    print(f"TOTAL {total} transiciones")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
