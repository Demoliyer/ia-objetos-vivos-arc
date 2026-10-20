# SPDX-License-Identifier: MIT-0
"""Qué tan seguido el motor (vivos/mundo.py), sin el LLM y sin gastar tokens, aprende
solo una regla que predice el nivel, sobre las mismas partidas que jugó el LLM.

    python experimentos/reglas/medir_motor.py <dir_trayectorias> [--json salida.json]

Aprendizaje en línea por nivel (como jugaría el agente): antes de cada transición el
motor predice el cuadro siguiente con lo aprendido hasta ahí y después aprende la real.
Se compara el INTERIOR (sin la franja de 4 celdas del HUD, igual que el auto-chequeo del
injerto) y además se excluye la banda de relojes que detecta el motor.

Por nivel:
  acierto_cambios   aciertos exactos / transiciones donde algo cambió en el interior
  regla_completa    desde la 4.ª transición con cambio, acierta >= 90 % (mínimo 3 medidas)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluar_mundo import banda_de_relojes  # noqa: E402
from vivos.mundo import ModeloMundo  # noqa: E402
from vivos.percepcion import DetectorDeRuido  # noqa: E402

FRANJA = 4
CALENTAMIENTO = 3


def niveles(ruta):
    d = np.load(ruta, allow_pickle=False)
    cuadros, acciones, cortes = d["cuadros"].astype(np.int16), [str(a) for a in d["acciones"]], d["corte"]
    tramo = []
    for t, a in enumerate(acciones):
        if cortes[t]:
            if tramo:
                yield tramo
            tramo = []
            continue
        tramo.append((cuadros[t], a, cuadros[t + 1]))
    if tramo:
        yield tramo


def medir_nivel(tramo):
    borde = np.ones((64, 64), dtype=bool)
    borde[FRANJA:-FRANJA, FRANJA:-FRANJA] = False
    ruido, modelo = DetectorDeRuido(), ModeloMundo()
    aciertos = []
    for antes, accion, despues in tramo:
        mascara = banda_de_relojes(ruido) | borde
        fuera = ~mascara
        if not np.array_equal(antes[fuera], despues[fuera]):
            pred = modelo.predecir(antes, accion, mascara)
            aciertos.append(bool(np.array_equal(pred[fuera], despues[fuera])))
        ruido.observar(antes, despues, accion)
        modelo.observar(antes, accion, despues, banda_de_relojes(ruido) | borde)
    tarde = aciertos[CALENTAMIENTO:]
    return {"cambios": len(aciertos), "aciertos": sum(aciertos),
            "regla_completa": len(tarde) >= 3 and sum(tarde) >= 0.9 * len(tarde)}


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    filas = []
    for ruta in sorted(Path(argv[1]).glob("*.npz")):
        juego, corrida = ruta.stem.split("__")
        for i, tramo in enumerate(niveles(ruta), 1):
            r = medir_nivel(tramo)
            filas.append({"juego": juego, "corrida": corrida, "tramo": i, **r})
    filas = [f for f in filas if f["cambios"] > 0]
    n = len(filas)
    comp = sum(f["regla_completa"] for f in filas)
    ac = sum(f["aciertos"] for f in filas)
    ca = sum(f["cambios"] for f in filas)
    print(f"niveles (tramos con cambios): {n}; regla completa del motor: {comp} ({100 * comp / n:.0f} %); "
          f"acierto en transiciones con cambio: {ac}/{ca} ({100 * ac / ca:.0f} %)")
    por = defaultdict(lambda: [0, 0, 0, 0])
    for f in filas:
        a = por[f["juego"]]
        a[0] += 1
        a[1] += f["regla_completa"]
        a[2] += f["aciertos"]
        a[3] += f["cambios"]
    for j, (t, c, a, k) in sorted(por.items()):
        print(f"  {j}: niveles {t}, regla completa {c} ({100 * c / t:.0f} %), acierto {100 * a / max(1, k):.0f} %")
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(filas, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
