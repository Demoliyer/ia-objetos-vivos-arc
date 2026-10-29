"""Prepara el notebook del hito 2 de Franzen para experimentar sin quemar cuota de GPU.

    python3 kaggle/preparar_franzen.py ENTRADA.ipynb SALIDA.ipynb [--env CLAVE=VALOR ...] [--completo]

Qué cambia (solo fuera de la re-ejecución oficial; la corrida que puntúa queda idéntica
salvo las variables --env, que se aplican en los dos modos):

- Guardado rápido (por defecto): el "Save & Run" del notebook original juega los 25 juegos
  públicos 4 veces durante 9 horas antes de escribir submission.parquet. Acá juega 2 juegos,
  una pasada, unos 15 minutos (el modelo carga en segundo plano), y escribe el archivo.
  Así se verifica que el modelo carga y el harness juega, por ~30-40 min de GPU.
- --env CLAVE=VALOR: sobreescribe variables de configuración del harness (setup_env) para
  probar una opción por entrega. Ver ARC3-Inference/CONFIGURATION.md en
  github.com/da-fr/arc-agi-3-solution.
- --vivos: injerta check_model / plan_with_model (vivos/simulador.py vía kaggle/injerto_vivos.py):
  el modelo escribe su simulador del juego, se verifica contra lo observado y se planifica en él.
- --completo: deja el guardado de 9 horas original (mide en los juegos públicos).

Crédito: notebook y harness de Daniel Franzen sobre el Duck Harness de Tufa Labs.
"""

from __future__ import annotations

import argparse
import ast
import re
import json
import sys
from pathlib import Path

MARCA_ENV = "os.environ.update({k: str(v) for k,v in setup_env.items()})"
MARCA_CONFIG = "# show config"
# Juegos públicos; en guardado rápido se juegan solo los de JUEGOS_RAPIDOS.
JUEGOS_PUBLICOS = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85",
    "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15",
    "tn36", "tr87", "tu93", "vc33", "wa30",
]
JUEGOS_RAPIDOS = ["ls20", "ft09"]
MINUTOS_RAPIDOS = 15

# Práctica (--minutos dado): esperar al servidor del modelo antes de fijar el presupuesto, para que
# los minutos de juego no dependan de lo que tarde Kaggle en cargar los pesos (24-oct: 13-18 min
# de una práctica de 45 se fueron esperando). El servidor se lanza en una celda anterior (Popen).
ESPERA_SERVIDOR = '''
    import urllib.request as _url
    _base = os.environ.get("LOCAL_ANALYZER_BASE_URL", "").strip().rstrip("/")
    _raiz = _base[:-3] if _base.endswith("/v1") else _base
    _t0 = time.time()
    while _raiz and time.time() - _t0 < 5400:
        try:
            with _url.urlopen(_raiz + "/health", timeout=5) as _r:
                if 200 <= int(getattr(_r, "status", 200)) < 300:
                    break
        except Exception:
            pass
        time.sleep(15)
    print(f"objetos_vivos: práctica, servidor listo tras {time.time() - _t0:.0f} s de espera")
'''


def bloque_rapido(juegos: list[str] | None = None, minutos: int | None = None) -> str:
    """Bloque del guardado rápido. No afecta la re-ejecución oficial (TRUE_SUBMISSION)."""
    espera = ESPERA_SERVIDOR if minutos else ""
    juegos = juegos or JUEGOS_RAPIDOS
    minutos = minutos or MINUTOS_RAPIDOS
    return f'''
# --- Objetos Vivos: guardado rápido (no afecta la re-ejecución oficial) ---
COMMIT_RAPIDO = True
if COMMIT_RAPIDO and not TRUE_SUBMISSION:{espera}
    demo_excluded_games = {[j for j in JUEGOS_PUBLICOS if j not in juegos]!r}
    bm.n_passes = 1
    bm.solver.concurrency = {len(juegos)}
    bm.solver.max_runtime_s_per_game = {minutos} * 60
    # La celda de ejecución corta en inicio + presupuesto - min(600, presupuesto/2).
    # Con presupuesto = transcurrido + {minutos} min + 600 s, corta ~{minutos} min después de esta celda.
    _transcurrido = time.time() - NOTEBOOK_START_EPOCH
    target.max_runtime_s = float(_transcurrido + {minutos} * 60 + 600)
    print(f"objetos_vivos: guardado rápido, juegos={juegos}, corte en ~{minutos} min")
'''


BLOQUE_RAPIDO = bloque_rapido()


def _texto(celda: dict) -> str:
    fuente = celda["source"]
    return "".join(fuente) if isinstance(fuente, list) else fuente


def _poner(celda: dict, texto: str) -> None:
    celda["source"] = texto.splitlines(keepends=True)


def _valor(crudo: str) -> str:
    """Las variables quedan como strings, igual que en setup_env tras os.environ.update."""
    return crudo


CELDA_GPU = '''# --- Objetos Vivos: en el guardado (no en la re-ejecución oficial) cortar si la GPU no es la esperada ---
if not TRUE_SUBMISSION:
    _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    print("objetos_vivos: GPU =", _gpu or "(ninguna)")
    if "RTX PRO 6000" not in _gpu.upper():
        raise RuntimeError(f"GPU inesperada {_gpu!r}: elegí 'GPU RTX Pro 6000' en Session options")
'''


def _celda(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.splitlines(keepends=True)}


def _celda_vivos(fuerte: bool = False, auto: bool = False) -> dict:
    """Celda que injerta check_model / plan_with_model (kaggle/injerto_vivos.py) en el harness."""
    raiz = Path(__file__).resolve().parent
    injerto = (raiz / "injerto_vivos.py").read_text(encoding="utf-8")
    simulador = (raiz.parent / "vivos" / "simulador.py").read_text(encoding="utf-8")
    texto = (injerto + "\n\n_VIVOS_SRC = " + repr(simulador) + "\n"
             f"print(aplicar(BUNDLE_DIR / 'src', _VIVOS_SRC, fuerte={fuerte!r}, auto={auto!r}))\n")
    compile(texto, "<celda_vivos>", "exec")
    return _celda(texto)


def _celda_arranque() -> dict:
    """Guarda de arranque (kaggle/guarda_arranque.py): en los dos modos, porque protege la corrida
    oficial de la carga lenta del modelo en Kaggle (oct-2026)."""
    texto = ((Path(__file__).resolve().parent / "guarda_arranque.py").read_text(encoding="utf-8")
             + "\nprint(aplicar(BUNDLE_DIR / 'src'))\n")
    compile(texto, "<celda_arranque>", "exec")
    return _celda(texto)


MARCA_CFG = "CFG = dict("


def _cambiar_cfg(nb: dict, cfg: dict[str, str]) -> None:
    """Cambia valores del dict CFG del servidor (celda de sglang). Afecta también la corrida oficial."""
    con_cfg = [c for c in nb["cells"] if c["cell_type"] == "code" and MARCA_CFG in _texto(c)]
    if len(con_cfg) != 1:
        sys.exit(f"No encuentro una única celda con {MARCA_CFG!r} (hay {len(con_cfg)})")
    texto = _texto(con_cfg[0])
    # Medido 2026-10-17: con SPEC_STEPS=4 (5 tokens de borrador) el servidor no arranca
    # ("Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4)").
    if "SPEC_STEPS" in cfg and int(cfg["SPEC_STEPS"]) + 1 > 4:
        sys.exit("cfg.SPEC_STEPS > 3: el servidor de Franzen no arranca (QSA admite <= 4 tokens de borrador)")
    for k, v in cfg.items():
        try:
            ast.literal_eval(v)
        except (ValueError, SyntaxError):
            sys.exit(f"cfg.{k}: {v!r} no es un literal de Python")
        rx = re.compile(rf"^(\s*){re.escape(k)}=([^\n]*?),[ \t]*$", re.M)
        if len(rx.findall(texto)) != 1:
            sys.exit(f"cfg.{k}: no aparece exactamente una vez en CFG")
        texto = rx.sub(lambda m: f"{m.group(1)}{k}={v},  # Objetos Vivos (antes {m.group(2)})", texto)
    _poner(con_cfg[0], texto)


def preparar(nb: dict, envs: dict[str, str], rapido: bool, vivos: bool = False, fuerte: bool = False,
             auto: bool = False,
             cfg: dict[str, str] | None = None, juegos: list[str] | None = None,
             minutos: int | None = None) -> dict:
    if juegos and (bad := [j for j in juegos if j not in JUEGOS_PUBLICOS]):
        sys.exit(f"juegos desconocidos: {bad}")
    if cfg:
        _cambiar_cfg(nb, cfg)
    celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
    con_env = [c for c in celdas if MARCA_ENV in _texto(c)]
    con_config = [c for c in celdas if MARCA_CONFIG in _texto(c)]
    if len(con_env) != 1 or len(con_config) != 1:
        sys.exit(f"No reconozco el notebook: celdas env={len(con_env)} config={len(con_config)}. "
                 "¿Cambió el notebook de Franzen?")
    if envs:
        lineas = "".join(f"setup_env[{k!r}] = {_valor(v)!r}\n" for k, v in envs.items())
        bloque = "# --- Objetos Vivos: experimento ---\n" + lineas
        texto = _texto(con_env[0]).replace(MARCA_ENV, bloque + MARCA_ENV)
        _poner(con_env[0], texto)
    if rapido:
        texto = _texto(con_config[0]).replace(MARCA_CONFIG, bloque_rapido(juegos, minutos) + "\n" + MARCA_CONFIG)
        _poner(con_config[0], texto)
    fuerte = fuerte or auto
    vivos = vivos or fuerte
    # Guarda de arranque: siempre (también en la corrida oficial), después del parche de Franzen.
    nb["cells"].insert(nb["cells"].index(con_env[0]) + 1, _celda_arranque())
    if vivos:  # justo después de aplicar el parche de Franzen, antes de importar el harness
        nb["cells"].insert(nb["cells"].index(con_env[0]) + 1, _celda_vivos(fuerte, auto))
    if rapido:  # antes de cargar nada pesado
        nb["cells"].insert(nb["cells"].index(con_env[0]) + 1, _celda(CELDA_GPU))
    # El notebook de Franzen marca isGpuEnabled=False aunque usa la RTX PRO 6000.
    kag = nb.setdefault("metadata", {}).setdefault("kaggle", {})
    kag["isInternetEnabled"] = False
    if kag.get("accelerator", "none") != "none":
        kag["isGpuEnabled"] = True
    for c in celdas:  # salidas viejas pesan y no sirven
        c["outputs"] = []
        c["execution_count"] = None
    return nb


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada", type=Path)
    p.add_argument("salida", type=Path)
    p.add_argument("--env", action="append", default=[], metavar="CLAVE=VALOR")
    p.add_argument("--completo", action="store_true", help="guardado original de 9 h")
    p.add_argument("--vivos", action="store_true",
                   help="injerta check_model / plan_with_model (vivos/simulador.py) en el harness")
    p.add_argument("--vivos-fuerte", action="store_true",
                   help="--vivos + recordatorio por turno para que el modelo use check_model")
    p.add_argument("--vivos-auto", action="store_true",
                   help="--vivos-fuerte + auto-chequeo del step después de cada snippet")
    p.add_argument("--cfg", action="append", default=[], metavar="CLAVE=VALOR",
                   help="cambia CFG del servidor (p. ej. MAXREQ=14); también en la corrida oficial")
    p.add_argument("--juegos", help="guardado rápido: juegos separados por coma")
    p.add_argument("--minutos", type=int, help="guardado rápido: minutos de juego")
    a = p.parse_args(argv)
    cfg = {}
    for par in a.cfg:
        if "=" not in par:
            sys.exit(f"--cfg espera CLAVE=VALOR, recibí {par!r}")
        k, v = par.split("=", 1)
        cfg[k.strip()] = v.strip()
    juegos = [j.strip() for j in a.juegos.split(",") if j.strip()] if a.juegos else None
    envs = {}
    for par in a.env:
        if "=" not in par:
            sys.exit(f"--env espera CLAVE=VALOR, recibí {par!r}")
        k, v = par.split("=", 1)
        envs[k.strip()] = v.strip()
    nb = preparar(json.loads(a.entrada.read_text(encoding="utf-8")), envs, rapido=not a.completo,
                   vivos=a.vivos, fuerte=a.vivos_fuerte, auto=a.vivos_auto, cfg=cfg, juegos=juegos, minutos=a.minutos)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    modo = ("completo (9 h)" if a.completo else
            f"rápido ({', '.join(juegos or JUEGOS_RAPIDOS)}, ~{a.minutos or MINUTOS_RAPIDOS} min)")
    if cfg:
        modo += f"; CFG {cfg}"
    extra = (" + vivos con recordatorio y auto-chequeo" if a.vivos_auto else
             " + vivos con recordatorio por turno" if a.vivos_fuerte else
             " + herramientas vivos" if a.vivos else "")
    print(f"{a.salida}: guardado {modo}; variables: {envs or 'ninguna (base)'}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
