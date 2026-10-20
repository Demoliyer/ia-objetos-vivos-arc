# SPDX-License-Identifier: MIT-0
"""Qué tan seguido el LLM escribe una regla (step) y qué tan buena es, leído de los
auto-chequeos que el injerto le devolvió en las prácticas guardadas.

    python experimentos/reglas/medir_modelo.py <living-mesh-datos/salidas> [--json salida.json]

Por (corrida, juego, nivel jugado): si hubo algún auto-chequeo de `step`, el mejor k/n y si
llegó a explicar todo con n >= 3 ("regla completa"). Solo corridas con --vivos-auto o
posteriores (las de base no tienen auto-chequeo). Nivel = el del último "Current state:
step S, level L" visto antes del auto-chequeo.
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CORRIDAS_CON_AUTO = ("humo-auto", "banco8-auto", "banco8-v3", "medir-v3", "medir-v4", "medir-v4b",
                     "medir-v4c", "medir-v4s", "medir-v5", "ensayo-v4s")
NIVEL = re.compile(r"Current state: step \d+, level (\d+)")
AUTO = re.compile(r"\[auto-check of your retained step\] it reproduces (\d+)/(\d+)")


def leer(p):
    return gzip.open(p, "rt", errors="replace").read() if p.suffix == ".gz" else p.read_text(errors="replace")


def medir(base):
    filas = []
    for d in sorted(Path(base).iterdir()):
        if not d.is_dir() or not d.name.endswith(CORRIDAS_CON_AUTO) or not (d / "transcripts").is_dir():
            continue
        bench = json.loads((d / "benchmark.json").read_text())
        jugados = {g["game_id"][:4]: min(g["levels_completed"] + 1, g["number_of_levels"]) for g in bench["game_runs"]}
        completados = {g["game_id"][:4]: g["levels_completed"] for g in bench["game_runs"]}
        for t in sorted((d / "transcripts").iterdir()):
            juego = t.name[:4]
            texto = leer(t)
            por_nivel = defaultdict(list)
            nivel = 1
            for m in re.finditer(f"{NIVEL.pattern}|{AUTO.pattern}", texto):
                if m.group(1):
                    nivel = int(m.group(1))
                else:
                    por_nivel[nivel].append((int(m.group(2)), int(m.group(3))))
            for nv in range(1, jugados.get(juego, 1) + 1):
                ch = por_nivel.get(nv, [])
                completa = any(k == n and n >= 3 for k, n in ch)
                mejor = max((k / n for k, n in ch if n), default=None)
                filas.append({"corrida": d.name, "juego": juego, "nivel": nv, "auto_chequeos": len(ch),
                              "con_regla": bool(ch), "regla_completa": completa,
                              "mejor_fraccion": mejor, "nivel_completado": nv <= completados.get(juego, 0)})
    return filas


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    filas = medir(argv[1])
    n = len(filas)
    con = [f for f in filas if f["con_regla"]]
    comp = [f for f in filas if f["regla_completa"]]
    print(f"niveles jugados: {n}")
    print(f"  con alguna regla del modelo: {len(con)} ({100 * len(con) / n:.0f} %)")
    print(f"  con regla completa (explica todo, n>=3): {len(comp)} ({100 * len(comp) / n:.0f} % de todos; "
          f"{100 * len(comp) / max(1, len(con)):.0f} % de los que tuvieron regla)")
    por_juego = defaultdict(lambda: [0, 0, 0])
    for f in filas:
        a = por_juego[f["juego"]]
        a[0] += 1
        a[1] += f["con_regla"]
        a[2] += f["regla_completa"]
    for j, (t, c, k) in sorted(por_juego.items()):
        print(f"  {j}: niveles {t}, con regla {c} ({100 * c / t:.0f} %), completa {k} ({100 * k / t:.0f} %)")
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(filas, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
