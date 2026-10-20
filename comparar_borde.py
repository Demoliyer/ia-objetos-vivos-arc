"""Compara el test de "¿cambió el juego?" del harness de Franzen (recorte de 4 celdas de borde)
contra la máscara de relojes/HUD aprendida por vivos.percepcion, sobre las trayectorias grabadas.

    .venv/bin/python comparar_borde.py

falso_cambio: el borde dice "cambió" pero fuera de relojes no cambió nada -> el guard de no-op no frena.
falso_noop:   el borde dice "no cambió" pero hubo cambio real cerca del borde -> el guard frena una acción útil.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluar_mundo import banda_de_relojes
from vivos.percepcion import DetectorDeRuido

BORDE = 4


def interior_cambio(a: np.ndarray, b: np.ndarray) -> bool:
    return not np.array_equal(a[BORDE:-BORDE, BORDE:-BORDE], b[BORDE:-BORDE, BORDE:-BORDE])


def main() -> None:
    tot = {"n": 0, "real": 0, "falso_cambio": 0, "falso_noop": 0}
    print(f"{'juego':6s} {'trans':>5s} {'real':>5s} {'f_cambio':>8s} {'f_noop':>6s}")
    for ruta in sorted(Path("trayectorias").glob("*.npz")):
        d = np.load(ruta, allow_pickle=False)
        cuadros = d["cuadros"].astype(np.int16)
        acciones = [str(a) for a in d["acciones"]]
        cortes = d["corte"]
        ruido = DetectorDeRuido()
        r = {"n": 0, "real": 0, "falso_cambio": 0, "falso_noop": 0}
        for t, accion in enumerate(acciones):
            a, b = cuadros[t], cuadros[t + 1]
            if cortes[t]:
                if accion != "RESET" and not np.array_equal(a, b):
                    ruido = DetectorDeRuido()
                continue
            if t >= 15:  # dar tiempo a que el detector aprenda los relojes
                fuera = ~banda_de_relojes(ruido)
                real = not np.array_equal(a[fuera], b[fuera])
                borde = interior_cambio(a, b)
                r["n"] += 1
                r["real"] += real
                r["falso_cambio"] += borde and not real
                r["falso_noop"] += real and not borde
            ruido.observar(a, b, accion)
        for k in tot:
            tot[k] += r[k]
        print(f"{ruta.stem:6s} {r['n']:5d} {r['real']:5d} {r['falso_cambio']:8d} {r['falso_noop']:6d}")
    print(f"{'TOTAL':6s} {tot['n']:5d} {tot['real']:5d} {tot['falso_cambio']:8d} {tot['falso_noop']:6d}")


if __name__ == "__main__":
    main()
