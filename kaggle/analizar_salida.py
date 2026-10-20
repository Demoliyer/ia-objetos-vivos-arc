"""Resume la salida de un kernel de Kaggle (kernels output): archivos y uso de las herramientas vivos.

    python kaggle/analizar_salida.py DIRECTORIO [--max 3800]

Imprime una sola línea compacta (cabe en una anotación de GitHub Actions):
- archivos descargados y tamaño;
- cuántas veces aparece cada marca de uso (check_model, plan_with_model, ...) en total y en los
  3 archivos donde más aparece.

Solo stdlib. Contar apariciones en el texto es una aproximación: incluye el prompt (la ayuda que
describe las herramientas) además de las llamadas del modelo. Por eso también se cuentan formas
que solo aparecen en código ejecutado ("check_model(step", "plan_with_model(step") y el prompt
se estima con la frase fija de la ayuda.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Marcas a contar. Las que terminan en "(" con argumento son llamadas reales en código del
# modelo con alta probabilidad; la ayuda del prompt solo muestra la firma genérica.
MARCAS = {
    "check_model": re.compile(r"check_model\("),
    "plan_with_model": re.compile(r"plan_with_model\("),
    # La ayuda del prompt escribe las firmas entre comillas invertidas: se excluyen.
    "llamada_check(step": re.compile(r"(?<!`)check_model\(\s*step\b"),
    "llamada_plan(step": re.compile(r"(?<!`)plan_with_model\(\s*step\b"),
    "def_step": re.compile(r"def step\(\s*grid"),
    "ayuda_prompt": re.compile(r"share a ~24 s budget"),
    "step_errors": re.compile(r"step_errors"),
    "mismatch": re.compile(r"\bmismatch", re.I),
    "python_tool": re.compile(r"run_python|python_tool|\"python\""),
    "vivos_celda": re.compile(r"injertados|OMITIDO|ya aplicado"),
    "auto_check": re.compile(r"\[auto-check of your retained step\]"),
    "auto_estado": re.compile(r"\[auto-check of your observe/advance model\]"),
    "def_observe": re.compile(r"def observe\("),
    "unchanged": re.compile(r"\[auto-check\] UNCHANGED"),
    "constantes": re.compile(r"Warning: your functions use"),
}
# Se cuentan sobre el texto crudo: son textos nuestros (el freno), no del modelo.
MARCAS_CRUDAS = {
    "freno": re.compile(r"turns in a row without a single game action"),
}
# Textos que inyecta el injerto (ayuda del prompt y recordatorio por turno): se borran antes de
# contar, porque citan literalmente `def step(grid, action)` y `check_model(step)`.
_INYECTADO = [
    # Cada recordatorio es una sola línea: sin re.S, para no borrar lo que el modelo escribió entre
    # dos recordatorios (con re.S, el freno se comía todo hasta el próximo "instead of exploring").
    re.compile(r"\[World model\][^\n]*?(?:instead of exploring by hand\.|shows exactly where the "
               r"hypothesis is wrong\.|as a dict \{\(row, col\): color\}\.|would clearly save actions\.|"
               r"execute the plan\.)"),
    re.compile(r"- World-model tools.*?improves the\s+score\.", re.S),
    re.compile(r"It explains everything seen so far: plan_with_model\(step, goal, actions\) can search"),
    re.compile(r"skip re-exploring: plan_with_model\(step, goal, actions\) right away"),
]


def _leer(p: Path) -> str:
    """Texto de un archivo, también si está comprimido con gzip (repo de datos)."""
    if p.suffix == ".gz":
        import gzip
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return p.read_text(encoding="utf-8", errors="replace")


def _limpiar(x: str) -> str:
    for rx in _INYECTADO:
        x = rx.sub("", x)
    return x


TEXTO = {".html", ".json", ".jsonl", ".txt", ".log", ".md", ".csv", ".py", ".ipynb", ""}


def resumir(raiz: Path, maximo: int = 3800) -> str:
    archivos = sorted(p for p in raiz.rglob("*") if p.is_file())
    if not archivos:
        return f"sin archivos en {raiz}"
    total: Counter = Counter()
    por_archivo: dict[str, Counter] = {}
    for p in archivos:
        if p.suffix.lower() not in TEXTO or p.stat().st_size > 400 * 1024 * 1024:
            continue
        try:
            crudo = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        texto = _limpiar(crudo)
        c = Counter({k: len(rx.findall(texto)) for k, rx in MARCAS.items()})
        c.update({k: len(rx.findall(crudo)) for k, rx in MARCAS_CRUDAS.items()})
        c = +c  # descarta ceros
        if c:
            por_archivo[str(p.relative_to(raiz))] = c
            total.update(c)
    partes = [f"{len(archivos)} archivos, {sum(p.stat().st_size for p in archivos) / 1e6:.1f} MB"]
    grandes = sorted(archivos, key=lambda p: -p.stat().st_size)[:6]
    partes.append("mayores: " + ", ".join(f"{p.relative_to(raiz)}={p.stat().st_size // 1024}KB"
                                          for p in grandes))
    partes.append("total: " + (", ".join(f"{k}={v}" for k, v in total.most_common()) or "ninguna marca"))
    for marca in ("llamada_check(step", "llamada_plan(step", "check_model"):
        top = sorted(((c[marca], n) for n, c in por_archivo.items() if c[marca]), reverse=True)[:3]
        if top:
            partes.append(f"{marca} en: " + ", ".join(f"{n}={v}" for v, n in top))
    partes.append("eventos: " + _eventos(raiz))
    niveles = []
    for ev in sorted(raiz.rglob("*_events.jsonl")):
        x = ev.read_text(encoding="utf-8", errors="replace")
        nums = [int(n) for n in re.findall(r'"(?:levels_completed|level_index|level)"\s*:\s*(\d+)', x)]
        niveles.append(f"{ev.name.split('-')[0]}={max(nums) if nums else '?'}")
    if niveles:
        partes.append("nivel máx: " + ", ".join(niveles))
    partes.append("puntaje: " + _puntajes(raiz))
    partes.append("vivos: " + _detalle_vivos(raiz))
    for log in raiz.rglob("serve.log"):
        clave = re.compile(r"fired up|ready to roll|Uvicorn running|out of memory|OOM|Traceback|Error", re.I)
        hits = [l.strip()[:160] for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if clave.search(l)]
        partes.append(f"serve.log: {len(hits)} líneas clave; " + " / ".join(hits[:2] + hits[-2:]))
    for t in sorted(raiz.rglob("transcripts/*.txt"))[:2]:
        x = t.read_text(encoding="utf-8", errors="replace")
        partes.append(f"{t.name}: {len(x)} chars, {x.count(chr(10))} lineas, fin=" +
                      " ".join(x[-450:].split()))
    return " | ".join(partes)[:maximo]


def _puntajes(raiz: Path) -> str:
    """game_score que calcula el harness de Franzen (traces.py: min(1.15,h/a)^2 por nivel,
    ponderado por nivel, ×100), por juego, y la media. Primero summary.txt; si no, la clave en json."""
    por_juego_det: dict[str, str] = {}
    por_juego: dict[str, float] = {}
    import json
    for bench in raiz.rglob("benchmark.json"):  # fuente principal: final_score por juego
        try:
            d = json.loads(bench.read_text(encoding="utf-8"))
            for g in d.get("game_runs", []):
                juego = str(g["game_id"]).split("-")[0]
                por_juego[juego] = float(g.get("final_score") or 0.0)
                por_juego_det[juego] = (f"{por_juego[juego]:.2f} ({g.get('levels_completed')}/"
                                        f"{g.get('number_of_levels')} niv, {sum(g.get('actions_per_level') or [])} acc)")
        except (ValueError, KeyError, TypeError):
            pass
    for resumen in ([] if por_juego else raiz.rglob("summary.txt")):  # resumen del benchmark del harness (fórmula oficial)
        x = resumen.read_text(encoding="utf-8", errors="replace")
        for juego, valor, nv, tot, acc in re.findall(
                r"^\s+([a-z0-9]+)-[0-9a-f]+: score=([0-9.]+), levels=([0-9.]+)/(\d+), actions=(\d+)", x, re.M):
            por_juego[juego] = float(valor)
            por_juego_det[juego] = f"{valor} ({nv}/{tot} niv, {acc} acc)"
    if por_juego:
        media = sum(por_juego.values()) / len(por_juego)
        return f"media={media:.2f} sobre {len(por_juego)} juegos; " + ", ".join(
            f"{k}={v}" for k, v in sorted(por_juego_det.items()))
    for p in raiz.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".json", ".jsonl", ".html"}:
            continue
        if p.stat().st_size > 50 * 1024 * 1024:
            continue
        x = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'"game_id"\s*:\s*"([a-z0-9]+)-[^"]*"[^{}]{0,4000}?"game_score"\s*:\s*([0-9.]+)'
                             r'|"game_score"\s*:\s*([0-9.]+)', x):
            juego = m.group(1) or p.name.split("-")[0]
            valor = float(m.group(2) or m.group(3))
            por_juego[juego] = max(por_juego.get(juego, 0.0), valor)
    if not por_juego:
        return "sin game_score"
    media = sum(por_juego.values()) / len(por_juego)
    return f"media={media:.2f} sobre {len(por_juego)} juegos; " + ", ".join(
        f"{k}={v:.2f}" for k, v in sorted(por_juego.items()))


def _detalle_vivos(raiz: Path) -> str:
    """Qué devolvieron check_model / plan_with_model, leído de los transcripts (sin historial repetido)."""
    c: Counter = Counter()
    motivos: Counter = Counter()
    largos: list[int] = []
    ejemplo = ""
    for t in sorted(list(raiz.rglob("transcripts/*.txt")) + list(raiz.rglob("transcripts/*.txt.gz"))):
        crudo = _leer(t)
        x = _limpiar(crudo)
        juego = t.name.split("-")[0]
        c[f"{juego}.freno"] += crudo.count("turns in a row without a single game action")
        c[f"{juego}.def_observe"] += len(re.findall(r"def observe\(", x))
        c[f"{juego}.auto_estado"] += x.count("[auto-check of your observe/advance model]")
        c[f"{juego}.unchanged"] += x.count("[auto-check] UNCHANGED")
        c[f"{juego}.aviso_constantes"] += x.count("Warning: your functions use")
        c[f"{juego}.def_step"] += len(re.findall(r"def step\(", x))
        c[f"{juego}.check"] += len(re.findall(r"(?<!`)check_model\(\s*\w", x))
        c[f"{juego}.plan"] += len(re.findall(r"(?<!`)plan_with_model\(\s*\w", x))
        c[f"{juego}.auto"] += len(re.findall(r"\[auto-check of your retained step\]", x))
        c[f"{juego}.aviso_desplazamiento"] += x.count("fails when the board is shifted by one cell")
        c[f"{juego}.step_general_nivel_nuevo"] += x.count("is position-independent")
        c[f"{juego}.aviso_global"] += x.count("step reads the live global")
        for ok, n in re.findall(r"retained step\] it reproduces (\d+)/(\d+)", x):
            c[f"{juego}.auto_perfecto"] += int(ok == n)
        c[f"{juego}.ok_true"] += len(re.findall(r"'all_interior_ok': True", x))
        c[f"{juego}.ok_false"] += len(re.findall(r"'all_interior_ok': False", x))
        motivos.update(m[:50] for m in re.findall(r"'reason': '([^']*)'", x))
        largos += [int(n) for n in re.findall(r"'length': (\d+)", x)]
        for ch, ok in re.findall(r"'checked': (\d+), 'exact': \d+, 'interior_ok': (\d+)", x):
            c[f"{juego}.checks_impresos"] += 1
            c[f"{juego}.transiciones"] += int(ch)
            c[f"{juego}.interior_ok"] += int(ok)
            c[f"{juego}.check_perfecto"] += int(int(ch) > 0 and int(ok) == int(ch))
        if not ejemplo:
            m = re.search(r"\{'checked': \d+[^}]{0,300}", x)
            ejemplo = m.group(0) if m else ""
    return (", ".join(f"{k}={v}" for k, v in sorted(c.items()) if v) +
            f"; motivos plan: {dict(motivos.most_common(5))}; largos plan: {largos[:15]}"
            f"; ej check: {' '.join(ejemplo.split())[:300]}")


def _eventos(raiz: Path) -> str:
    """Valores frecuentes de claves tipo/herramienta en los *_events.jsonl (formato no documentado)."""
    import json
    claves = ("type", "event", "kind", "tool", "tool_name", "name", "action")
    c: Counter = Counter()
    for p in raiz.rglob("*.jsonl"):
        for linea in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                o = json.loads(linea)
            except ValueError:
                continue
            pila = [o]
            while pila:
                x = pila.pop()
                if isinstance(x, dict):
                    for k, v in x.items():
                        if k in claves and isinstance(v, str) and len(v) < 40:
                            c[f"{k}={v}"] += 1
                        elif isinstance(v, (dict, list)):
                            pila.append(v)
                elif isinstance(x, list):
                    pila.extend(x)
    return ", ".join(f"{k}:{v}" for k, v in c.most_common(40)) or "sin jsonl"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directorio", type=Path)
    p.add_argument("--max", type=int, default=3800)
    a = p.parse_args(argv)
    if not a.directorio.is_dir():
        print(f"no existe {a.directorio}")
        return 1
    print(resumir(a.directorio, a.max))
    return 0


if __name__ == "__main__":
    sys.exit(main())
