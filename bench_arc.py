"""Banco local: la IA de Objetos Vivos contra un agente aleatorio.

    .venv/bin/python bench_arc.py                 # 5 semillas, tope 600 acciones
    .venv/bin/python bench_arc.py --semillas 20 --json bench-arc.json

Por juego: niveles ganados y acciones usadas (mediana y peor caso). Sale con
código 1 si la IA no cumple la aceptación de docs/IA_OBJETOS_VIVOS.md §6: ganar
todos los niveles de cada juego dentro de 300 acciones y con menos acciones que
el aleatorio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from vivos.agente import AgenteVivo
from vivos.arena import AgenteAleatorio, jugar
from vivos.juegos import JUEGOS, EntornoLocal

ACEPTACION = 300


def correr(semillas: int, tope: int) -> dict:
    out = {}
    for nombre in JUEGOS:
        vivos = [jugar(EntornoLocal.crear(nombre), AgenteVivo(s), max_acciones=tope, juego=nombre) for s in range(semillas)]
        azar = [jugar(EntornoLocal.crear(nombre), AgenteAleatorio(s), max_acciones=tope, juego=nombre) for s in range(semillas)]
        out[nombre] = {
            "win_levels": vivos[0].win_levels,
            "vivos": [{"niveles": r.niveles, "acciones": r.acciones, "por_nivel": r.acciones_por_nivel, "segundos": round(r.segundos, 3)} for r in vivos],
            "aleatorio": [{"niveles": r.niveles, "acciones": r.acciones} for r in azar],
        }
    return out


def resumen(datos: dict) -> tuple[list[str], bool]:
    filas = [f"{'juego':14s} {'IA niveles':>10s} {'IA acciones (med/peor)':>24s} {'azar niveles':>13s} {'azar acciones':>14s}"]
    ok = True
    for nombre, d in datos.items():
        total = d["win_levels"]
        vn = [r["niveles"] for r in d["vivos"]]
        va = [r["acciones"] for r in d["vivos"]]
        an = [r["niveles"] for r in d["aleatorio"]]
        aa = [r["acciones"] for r in d["aleatorio"]]
        gano_todo = all(n == total for n in vn)
        dentro = max(va) <= ACEPTACION
        mejor = statistics.median(va) < statistics.median(aa)
        ok &= gano_todo and dentro and mejor
        marca = "" if gano_todo and dentro and mejor else "  <-- no cumple"
        filas.append(
            f"{nombre:14s} {min(vn)}-{max(vn)}/{total:<6d} {statistics.median(va):>14.0f} / {max(va):<7d} "
            f"{min(an)}-{max(an)}/{total:<8d} {statistics.median(aa):>13.0f}{marca}"
        )
    return filas, ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--semillas", type=int, default=5)
    p.add_argument("--tope", type=int, default=600, help="acciones máximas por juego")
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    datos = correr(args.semillas, args.tope)
    filas, ok = resumen(datos)
    print("\n".join(filas))
    if args.json:
        args.json.write_text(json.dumps(datos, indent=2), encoding="utf-8")
    print("aceptación: " + ("cumplida" if ok else "NO cumplida"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
