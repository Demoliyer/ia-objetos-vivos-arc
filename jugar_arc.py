"""Juega ARC-AGI-3 con la IA de Objetos Vivos.

    # banco local (juegos ARCEngine propios, sin red)
    .venv/bin/python jugar_arc.py --local todos
    .venv/bin/python jugar_arc.py --local laberinto --ver

    # juegos oficiales con el kit arc-agi (ARC_API_KEY opcional; sin clave usa una anónima)
    .venv/bin/python jugar_arc.py --juego ls20
    .venv/bin/python jugar_arc.py --todos --max-acciones 2000

El modo del kit sale de OPERATION_MODE (normal, online, offline, competition),
como en el kit oficial. La colonia aprendida en cada juego se exporta a JSON
legible en --exportar (default ./vivos-data).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from vivos.agente import AgenteVivo
from vivos.arena import Resultado, jugar
from vivos.puntaje import de_juego


def _exportar(destino: Path | None, juego: str, agente: AgenteVivo, res: Resultado, base: list[int] | None) -> None:
    if destino is None:
        return
    destino.mkdir(parents=True, exist_ok=True)
    datos = {
        "juego": juego,
        "resultado": {
            "niveles": res.niveles,
            "win_levels": res.win_levels,
            "acciones": res.acciones,
            "acciones_por_nivel": res.acciones_por_nivel,
            "reinicios": res.reinicios,
            "estado": res.estado,
            "base_humana": base,
            "puntaje_estimado": de_juego(res.acciones_por_nivel, base),
        },
        "agente": agente.exportar(),
    }
    (destino / f"{juego}.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _imprimir(res: Resultado, puntaje: float | None) -> None:
    extra = f", puntaje estimado {puntaje:.3f}" if puntaje is not None else ""
    print(res.linea() + extra, flush=True)


def _observador(ver: bool):
    if not ver:
        return None

    def mirar(res: Resultado, d, obs) -> None:
        print(f"  {res.acciones:5d} {d.clave:18s} {d.motivo}", flush=True)

    return mirar


def jugar_local(nombres: list[str], args) -> list[tuple[str, Resultado, float | None]]:
    from vivos.juegos import JUEGOS, EntornoLocal

    if nombres == ["todos"]:
        nombres = list(JUEGOS)
    salida = []
    for nombre in nombres:
        if nombre not in JUEGOS:
            sys.exit(f"juego local desconocido: {nombre} (hay: {', '.join(JUEGOS)})")
        agente = AgenteVivo(args.semilla)
        res = jugar(EntornoLocal.crear(nombre), agente, max_acciones=args.max_acciones, juego=nombre,
                    alguien_mira=_observador(args.ver), segundos_max=args.segundos)
        _exportar(args.exportar, nombre, agente, res, None)
        salida.append((nombre, res, None))
    return salida


def jugar_oficial(ids: list[str] | None, args) -> list[tuple[str, Resultado, float | None]]:
    try:
        import arc_agi
    except ImportError:
        sys.exit("Falta el kit oficial: .venv/bin/pip install -r requirements-arc.txt")
    arc = arc_agi.Arcade()
    entornos = {e.game_id: e for e in arc.get_environments()}
    if ids is None:
        ids = sorted(entornos)
    salida = []
    for gid in ids:
        # Acepta el id corto (ls20) o el completo (ls20-9607627b) para encontrar la base humana.
        info = entornos.get(gid) or next((e for k, e in entornos.items() if k.split("-")[0] == gid), None)
        env = arc.make(gid, seed=args.semilla)
        if env is None:
            print(f"{gid}: no se pudo crear el entorno", file=sys.stderr)
            continue
        agente = AgenteVivo(args.semilla)
        base = list(info.baseline_actions) if info is not None and info.baseline_actions else None
        try:
            res = jugar(env, agente, max_acciones=args.max_acciones, juego=gid, alguien_mira=_observador(args.ver),
                        segundos_max=args.segundos)
        except Exception:
            # Un juego que rompe al agente no tumba la corrida: queda como ERROR con su traza.
            traza = traceback.format_exc()
            print(f"{gid}: ERROR en el agente\n{traza}", file=sys.stderr, flush=True)
            res = Resultado(juego=gid, estado="ERROR")
            if args.exportar is not None:
                args.exportar.mkdir(parents=True, exist_ok=True)
                (args.exportar / f"{gid}.error.txt").write_text(traza, encoding="utf-8")
        try:
            _exportar(args.exportar, gid, agente, res, base)
        except Exception as exc:
            print(f"{gid}: no se pudo exportar ({exc})", file=sys.stderr, flush=True)
        puntaje = de_juego(res.acciones_por_nivel, base)
        _imprimir(res, puntaje)
        salida.append((gid, res, puntaje))
    try:
        print(arc.get_scorecard())
    except Exception as exc:  # el scorecard es informativo; no tumba la corrida
        print(f"(sin scorecard: {exc})", file=sys.stderr)
    return salida


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--local", nargs="+", metavar="JUEGO", help="juegos del banco local, o 'todos'")
    grupo.add_argument("--juego", nargs="+", metavar="ID", help="juegos oficiales por id (p. ej. ls20)")
    grupo.add_argument("--todos", action="store_true", help="todos los juegos oficiales disponibles")
    p.add_argument("--max-acciones", type=int, default=2000, help="tope de acciones por juego, RESET incluido")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--segundos", type=float, default=None, help="tope de tiempo por juego (corta con estado TIEMPO)")
    p.add_argument("--exportar", type=Path, default=Path("vivos-data"), help="carpeta para el JSON de cada juego")
    p.add_argument("--no-exportar", action="store_true")
    p.add_argument("--ver", action="store_true", help="mostrar cada decisión y su motivo")
    args = p.parse_args(argv)
    if args.no_exportar:
        args.exportar = None

    if args.local:
        resultados = jugar_local(args.local, args)
        for _, res, puntaje in resultados:
            _imprimir(res, puntaje)
    else:
        resultados = jugar_oficial(args.juego, args)  # imprime cada juego al terminarlo

    puntajes = [p for _, _, p in resultados if p is not None]
    if puntajes:
        print(f"promedio estimado: {sum(puntajes) / len(puntajes):.3f} sobre {len(puntajes)} juegos")
    return 0 if all(r.estado != "ERROR" for _, r, _ in resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
