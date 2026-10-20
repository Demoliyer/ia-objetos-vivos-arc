# SPDX-License-Identifier: MIT-0
"""Guarda de identidad tipada antes de subir (complementa la huella v1).

    python kaggle/guarda_identidad.py <notebook.ipynb> origen|candidatos

Calcula `sha256_celdas_tipadas` (kaggle/identidad_notebook.py) y lo imprime. Si la huella v1
del notebook figura en kaggle/identidades_candidatos.json (sección pedida), exige que el
sha tipado coincida: una celda de código pasada a raw/markdown conserva la huella v1 pero no
este valor. Si la huella v1 no figura (experimento nuevo), solo informa.
Códigos: 0 = coincide o no hay valor esperado; 1 = no coincide.
"""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import identidad_notebook  # noqa: E402


def verificar(notebook, seccion, tabla=AQUI / "identidades_candidatos.json"):
    ident = identidad_notebook.identidad(notebook)
    esperado = json.loads(Path(tabla).read_text())[seccion].get(ident["huella_v1"])
    return ident, esperado, esperado is None or esperado == ident["sha256_celdas_tipadas"]


def main(argv):
    if len(argv) != 3 or argv[2] not in ("origen", "candidatos"):
        sys.exit(__doc__)
    ident, esperado, ok = verificar(argv[1], argv[2])
    print(ident["sha256_celdas_tipadas"])
    if not ok:
        print(f"identidad tipada distinta para huella {ident['huella_v1']}: esperada {esperado}, "
              f"obtenida {ident['sha256_celdas_tipadas']} (¿cambió el tipo de alguna celda?)", file=sys.stderr)
        return 1
    if esperado is None:
        print(f"(sin valor esperado para huella {ident['huella_v1']}: solo se registra)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
