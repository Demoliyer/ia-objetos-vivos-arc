# SPDX-License-Identifier: MIT-0
"""Huella de un notebook: sha256 (12 hex) de la lista de fuentes de sus celdas.

Ignora salidas, metadatos y números de ejecución: dos notebooks con la misma huella corren
el mismo código. Sirve para congelar experimentos (el notebook de origen de Franzen y el
notebook preparado que se entrega) y para reconstruir qué se corrió en cada entrega.

    python kaggle/huella_notebook.py notebook.ipynb [huella_esperada]

Imprime la huella. Con huella esperada, sale con código 1 si no coincide.
La huella v1 no ve el tipo de celda (código pasado a raw conserva la huella): para eso está
kaggle/identidad_notebook.py (sha256_celdas_tipadas) y la guarda kaggle/guarda_identidad.py.
"""
import hashlib
import json
import sys


def _celdas(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["cells"]


def _fuente(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def huella(ruta):
    fuentes = [_fuente(c) for c in _celdas(ruta)]
    return hashlib.sha256(json.dumps(fuentes).encode()).hexdigest()[:12]


def main(argv):
    if len(argv) not in (2, 3):
        sys.exit(__doc__)
    h = huella(argv[1])
    print(h)
    if len(argv) == 3 and argv[2] != h:
        print(f"huella distinta: esperada {argv[2]}, obtenida {h}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
