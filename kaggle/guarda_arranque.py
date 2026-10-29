"""Guarda de arranque para el harness de Franzen (Kaggle con lectura lenta del modelo, oct-2026).

Problema (medido el 22-oct y reportado en el foro de la competencia, hilos 751598 y 751631):
la carga de pesos del modelo pasó de ~8 min a 35-63 min. El harness hace un solo RESET de
calentamiento al empezar; la pasarela de la competencia cierra el scorecard tras ~15 min sin
acciones, y los juegos se rinden tras 10 fallos seguidos del modelo. Resultado: 0,00.

Arreglo (el mismo que reportó AquaUrsa, puesto 12, con una entrega válida el 22-oct):
- el juego que toma el turno de calentamiento hace RESET al entrar y cada CADA segundos
  mientras el servidor del modelo no responde (mantiene vivo el scorecard);
- los demás juegos esperan sin jugar hasta que el servidor responde;
- tope de TOPE segundos: si el servidor nunca arranca, la corrida sigue como antes.
Con el servidor sano al entrar no cambia nada (el calentamiento original queda igual).

Se agrega al final de ARC3-Inference/inference/framework/solver.py del paquete de Franzen,
en los dos modos (guardado y corrida oficial). Si las anclas no están, no toca nada.
"""

from __future__ import annotations

from pathlib import Path

CODIGO = r'''

# --- Objetos Vivos: guarda de arranque (carga lenta del modelo en Kaggle, oct-2026) ---
def _vivos_guarda_arranque():
    import os as _os
    import threading as _hilos
    import time as _t
    import urllib.request as _url

    _orig = _HarnessGameSession._play_inner
    _estado = {"listo": False, "guardian": False}
    _lock = _hilos.Lock()
    _tope = float(_os.environ.get("VIVOS_ARRANQUE_TOPE_S", "7200"))
    _cada = float(_os.environ.get("VIVOS_ARRANQUE_RESET_S", "300"))
    _paso = float(_os.environ.get("VIVOS_ARRANQUE_SONDEO_S", "10"))

    def _sano():
        if _estado["listo"]:
            return True
        base = _os.environ.get("LOCAL_ANALYZER_BASE_URL", "").strip().rstrip("/")
        if not base:  # sin servidor local conocido: no hay nada que esperar
            _estado["listo"] = True
            return True
        raiz = base[:-3] if base.endswith("/v1") else base
        for ruta in (raiz + "/health", base + "/models"):
            try:
                with _url.urlopen(ruta, timeout=5) as r:
                    if 200 <= int(getattr(r, "status", 200)) < 300:
                        _estado["listo"] = True
                        return True
            except Exception:
                pass
        return False

    def _play_inner(self):
        try:
            if not _sano():
                guardian = False
                with _lock:
                    if not _estado["guardian"] and self.solver._claim_warmup_slot():
                        _estado["guardian"] = guardian = True
                juego = getattr(getattr(self.game, "game_run", None), "game_id", "?")
                log.warning("%s: guarda de arranque: esperando el servidor del modelo (%s)",
                            juego, "RESET cada %.0f s" % _cada if guardian else "sin jugar")
                t0, ultimo, resets = _t.monotonic(), None, 0
                while not _sano() and _t.monotonic() - t0 < _tope and not self.should_stop():
                    if guardian and (ultimo is None or _t.monotonic() - ultimo >= _cada):
                        try:
                            self._execute_auto_reset()
                            resets += 1
                        except Exception:
                            log.exception("guarda de arranque: el RESET falló")
                        ultimo = _t.monotonic()
                    _t.sleep(_paso)
                log.warning("%s: guarda de arranque: servidor %s tras %.0f s (%d RESET)", juego,
                            "listo" if _sano() else "SIN responder", _t.monotonic() - t0, resets)
        except Exception:
            log.exception("guarda de arranque: error; se sigue sin esperar")
        return _orig(self)

    _HarnessGameSession._play_inner = _play_inner


try:
    _vivos_guarda_arranque()
except Exception:  # sin la guarda, el harness sigue como el original
    pass
'''

ANCLAS = ("class _HarnessGameSession", "def _play_inner(self)", "def _claim_warmup_slot(self)",
          "def _execute_auto_reset(self)", "def should_stop(self)", "\nlog = ")


def aplicar(src: Path) -> str:
    """Agrega la guarda a solver.py dentro de `src` (BUNDLE_DIR / "src"). Idempotente."""
    ruta = Path(src) / "ARC3-Inference" / "inference" / "framework" / "solver.py"
    try:
        texto = ruta.read_text(encoding="utf-8")
    except OSError as exc:
        return f"guarda de arranque OMITIDA: no se pudo leer {ruta} ({exc})"
    if "_vivos_guarda_arranque" in texto:
        return "guarda de arranque: ya estaba aplicada"
    faltan = [a.strip() for a in ANCLAS if a not in texto]
    if faltan:
        return f"guarda de arranque OMITIDA: harness distinto (faltan {faltan})"
    nuevo = texto + CODIGO
    compile(nuevo, str(ruta), "exec")
    ruta.write_text(nuevo, encoding="utf-8")
    return "guarda de arranque aplicada"
