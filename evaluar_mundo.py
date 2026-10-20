"""Mide qué tan bien predice el modelo del mundo el cuadro siguiente.

    .venv/bin/python evaluar_mundo.py                 # todas las trayectorias grabadas
    .venv/bin/python evaluar_mundo.py --juego tu93 tn36

Recorre cada trayectoria en orden: a partir del paso ``--desde`` predice el cuadro
siguiente con lo aprendido hasta ese momento y después aprende la transición real
(aprendizaje en línea, como jugaría el agente). Compara fuera de la máscara de
relojes, que se estima con el mismo detector que usa el agente.

Bases de comparación:
    nada     el cuadro no cambia
    memoria  si ese estado con esa acción ya se vio, lo mismo que entonces; si no, nada
    modelo   memoria + reglas de efecto (vivos/mundo.py)

Métrica principal: aciertos exactos en las transiciones donde algo cambió y el estado
con esa acción NO se había visto antes (lo que la memoria no puede resolver).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from vivos.mundo import ModeloMundo, huella as _huella
from vivos.percepcion import DetectorDeRuido


def banda_de_relojes(ruido: DetectorDeRuido) -> np.ndarray:
    """La máscara del agente más la banda entera de cada reloj (las filas o columnas que
    ocupa, según su eje largo): la barra avanza una celda por acción fuera de lo ya
    visto, y predecir el contador no sirve para planificar."""
    m = ruido.mascara.copy()
    for c in ruido.relojes:
        z = ruido._zona(c)
        if not z.any():
            continue
        ys, xs = np.nonzero(z)
        if xs.max() - xs.min() >= ys.max() - ys.min():
            m[ys.min() : ys.max() + 1, :] = True
        else:
            m[:, xs.min() : xs.max() + 1] = True
    return m


def evaluar(ruta: Path, desde: int) -> dict:
    d = np.load(ruta, allow_pickle=False)
    cuadros = d["cuadros"].astype(np.int16)
    acciones = [str(a) for a in d["acciones"]]
    cortes = d["corte"]
    ruido = DetectorDeRuido()
    modelo = ModeloMundo()
    vistos: set[tuple[bytes, str]] = set()
    r = {"total": 0, "cambia": 0, "nuevo_cambia": 0, "nada": 0, "memoria": 0, "modelo": 0,
         "modelo_nuevo": 0, "memoria_nuevo": 0, "fuera": 0}
    for t, accion in enumerate(acciones):
        antes, despues = cuadros[t], cuadros[t + 1]
        if cortes[t]:
            if accion != "RESET" and not np.array_equal(antes, despues):
                ruido = DetectorDeRuido()  # nivel nuevo: la percepción se reinicia
            continue
        mascara = banda_de_relojes(ruido)
        if t >= desde:
            fuera = ~mascara
            igual = lambda p: bool(np.array_equal(p[fuera], despues[fuera]))  # noqa: E731
            cambio = not np.array_equal(antes[fuera], despues[fuera])
            clave = (_huella(antes, mascara), accion)
            nuevo = clave not in vistos
            r["total"] += 1
            r["cambia"] += cambio
            r["nada"] += igual(antes)
            pred_mem = modelo.memoria.get(clave, antes)
            r["memoria"] += igual(pred_mem)
            pred = modelo.predecir(antes, accion, mascara)
            ok = igual(pred)
            r["modelo"] += ok
            if cambio and nuevo:
                r["nuevo_cambia"] += 1
                r["modelo_nuevo"] += ok
                r["memoria_nuevo"] += igual(pred_mem)
        vistos.add((_huella(antes, mascara), accion))
        ruido.observar(antes, despues, accion)
        modelo.observar(antes, accion, despues, banda_de_relojes(ruido))
    r["fuera"] = sum(len(v) for v in modelo.parches.values())
    return r


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--juego", nargs="+")
    p.add_argument("--trayectorias", type=Path, default=Path("trayectorias"))
    p.add_argument("--desde", type=int, default=15)
    args = p.parse_args(argv)
    rutas = sorted(args.trayectorias.glob("*.npz"))
    if args.juego:
        rutas = [r for r in rutas if r.stem in args.juego]
    if not rutas:
        sys.exit("No hay trayectorias: corré antes grabar_trayectorias.py")
    tot: dict[str, int] = {}
    print(f"{'juego':6s} {'trans':>5s} {'cambia':>6s} {'nuevas':>6s} | {'nada':>5s} {'memo':>5s} {'modelo':>6s} |"
          f" {'nuevas: memo':>12s} {'modelo':>6s} | parches")
    for ruta in rutas:
        r = evaluar(ruta, args.desde)
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + v
        n, nn = max(1, r["total"]), max(1, r["nuevo_cambia"])
        print(f"{ruta.stem:6s} {r['total']:5d} {r['cambia']:6d} {r['nuevo_cambia']:6d} |"
              f" {r['nada'] / n:5.0%} {r['memoria'] / n:5.0%} {r['modelo'] / n:6.0%} |"
              f" {r['memoria_nuevo'] / nn:12.0%} {r['modelo_nuevo'] / nn:6.0%} | {r['fuera']}")
    n, nn = max(1, tot["total"]), max(1, tot["nuevo_cambia"])
    print(f"{'TOTAL':6s} {tot['total']:5d} {tot['cambia']:6d} {tot['nuevo_cambia']:6d} |"
          f" {tot['nada'] / n:5.0%} {tot['memoria'] / n:5.0%} {tot['modelo'] / n:6.0%} |"
          f" {tot['memoria_nuevo'] / nn:12.0%} {tot['modelo_nuevo'] / nn:6.0%} | {tot['fuera']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
