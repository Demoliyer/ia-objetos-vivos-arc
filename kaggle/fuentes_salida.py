# SPDX-License-Identifier: MIT-0
"""Versiones de las fuentes que efectivamente usó una corrida, leídas de su salida de Kaggle.

El notebook de Franzen monta dos datasets sin versión fija (taaf-kaggle-source-bundle-copy:
el harness; pennyroyal-v253: sglang). La huella del notebook no los cubre, así que se
registra lo que la corrida realmente cargó y se compara con la referencia antes de combinar
resultados.

    python kaggle/fuentes_salida.py <dir_salida> [<dir_salida> ...]

Fase: esto es lo que cargó el GUARDADO RÁPIDO (la corrida que se ve). La re-ejecución oficial
sobre los juegos ocultos no deja salida visible: sus fuentes son desconocidas; que coincidan
con las del guardado de la misma versión es una inferencia, no un dato.
`git_status.txt` viene del bundle: un repo marcado DIRTY tiene modificaciones (el parche de
Franzen) que el commit no identifica; se informa como "dirty" aparte.

Estados por carpeta: ok; DIFIERE (algún campo registrado no coincide); INCOMPLETO (falta el
registro de algún campo); ambos se informan juntos. Código de salida (todas las carpetas):
0 = coinciden los campos registrados, 1 = hay diferencias, 2 = faltan campos, 3 = ambos
(esquema de la revisión externa de Astra, 20-oct-2026). Comparar campos registrados no
certifica los bytes de las dependencias.
"""
import gzip
import re
import sys
from pathlib import Path

# Lo que cargaron las corridas del 17 al 20-oct-2026 (10 de 10 con registro; banco8-base no lo guardó).
REFERENCIA = {
    "ARC3-Inference": "aa69123",
    "tufa-arc-agi-framework": "fe9f7c4",
    "re-arc-3": "57e46d619d",
    "sglang": "0.5.19+gd00d88efc8d6",
    "torch": "2.13.0+cu130",
    "modelo": "intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "borrador": "albucino-qwen3-8-flash-next-drafter/transformers/default/1",
    "parche": "aplicado",
}


def _leer(p):
    if not p.exists():
        gz = p.with_name(p.name + ".gz")
        if not gz.exists():
            return ""
        p = gz
    return gzip.open(p, "rt", errors="replace").read() if p.suffix == ".gz" else p.read_text(errors="replace")


def fuentes(d):
    d = Path(d)
    git = _leer(d / "git_status.txt")
    # kaggle-salida lo guarda como <kernel>.log; recuperar_versiones como kernel.log.
    log = _leer(d / "arc-agi-3-vivos.log") or _leer(d / "kernel.log")
    f = {}
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework", "re-arc-3"):
        m = re.search(rf"^[ \t]*{re.escape(repo)}[ \t]+([0-9a-f]{{7,40}})", git, re.M)
        f[repo] = m.group(1) if m else "no disponible"
    # Los valores terminan en espacio, salto de línea real, "\n" escapado de JSON, comillas o "|".
    fin = r"[^\s\\\"'|]+"
    for clave, patron in (("sglang", r"sglang\s+(\d" + fin + ")"), ("torch", r"torch\s+(\d" + fin + ")"),
                          ("modelo", r"Target: /kaggle/input/models/dfranzen/(" + fin + ")"),
                          ("borrador", r"Draft source: /kaggle/input/models/dfranzen/(" + fin + ")")):
        m = re.search(patron, log)
        f[clave] = m.group(1).strip() if m else "no disponible"
    f["parche"] = "aplicado" if "harness patch applied" in log else "no disponible"
    return f


def repos_dirty(d):
    """Repos marcados DIRTY en git_status.txt: tienen cambios que el commit no identifica."""
    git = _leer(Path(d) / "git_status.txt")
    return [r for r in ("ARC3-Inference", "tufa-arc-agi-framework", "re-arc-3")
            if re.search(rf"^[ \t]*{re.escape(r)}[ \t]+[0-9a-f]{{7,40}}[ \t]+DIRTY\b", git, re.M | re.I)]


def comparar(f):
    faltan = [k for k in REFERENCIA if f.get(k) in (None, "", "no disponible")]
    difieren = {k: (f[k], v) for k, v in REFERENCIA.items() if k not in faltan and f[k] != v}
    return faltan, difieren


def estado(d):
    faltan, dif = comparar(fuentes(d))
    partes = []
    if dif:
        partes.append("DIFIERE " + "; ".join(f"{k}: {v} (ref {r})" for k, (v, r) in dif.items()))
    if faltan:
        partes.append("INCOMPLETO (sin registro: " + ", ".join(faltan) + ")")
    e = " | ".join(partes) if partes else "ok"
    sucios = repos_dirty(d)
    if sucios:
        e += " | dirty sin identificar: " + ", ".join(sucios)
    return e, bool(dif), bool(faltan)


def main(dirs):
    rc = 0
    for d in dirs:
        e, dif, falta = estado(d)
        print(f"{Path(d).name}: fuentes del guardado rápido={e}; evaluación oculta=no observable con el procedimiento actual")
        rc |= (1 if dif else 0) | (2 if falta else 0)
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
