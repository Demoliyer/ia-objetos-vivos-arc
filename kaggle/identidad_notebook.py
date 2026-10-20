# SPDX-License-Identifier: MIT-0
"""Identidad adicional; conserva la huella v1 y distingue tipos de celda.

Propuesta por la revisión externa (Astra, 20-oct-2026); integrada sin cambios de cálculo.

Uso: python kaggle/identidad_notebook.py notebook.ipynb [kernel-metadata.json]
No modifica el notebook ni demuestra qué dependencias usó el rerun oculto.
"""
import hashlib
import json
import sys
from pathlib import Path


def identidad(ruta, metadata=None):
    raw = Path(ruta).read_bytes()
    nb = json.loads(raw)
    celdas = []
    for c in nb["cells"]:
        source = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        celdas.append({"cell_type": c["cell_type"], "source": source})
    v1 = hashlib.sha256(json.dumps([c["source"] for c in celdas]).encode()).hexdigest()[:12]
    payload = {"schema": 2, "cells": celdas}
    v2 = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = {"huella_v1": v1, "sha256_celdas_tipadas": v2,
              "sha256_archivo": hashlib.sha256(raw).hexdigest()}
    if metadata is not None:
        result["sha256_kernel_metadata"] = hashlib.sha256(Path(metadata).read_bytes()).hexdigest()
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    print(json.dumps(identidad(*sys.argv[1:]), indent=2))
