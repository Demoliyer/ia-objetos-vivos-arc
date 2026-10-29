"""Guarda de arranque: los juegos esperan al servidor y uno mantiene vivo el scorecard con RESET."""
import importlib.util
import http.server
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "kaggle"))

from guarda_arranque import aplicar  # noqa: E402

HARNESS = Path(os.environ.get("VIVOS_HARNESS", RAIZ.parent / "franzen" / "ARC3-Inference"))

FALSO = '''
import logging, threading
log = logging.getLogger("falso")

class _HarnessGameSession:
    def __init__(self, solver, nombre, parar=lambda: False):
        self.solver, self.nombre, self.parar = solver, nombre, parar
        self.game = type("G", (), {"game_run": type("R", (), {"game_id": nombre})()})()
        self.resets = 0
    def should_stop(self):
        return self.parar()
    def _execute_auto_reset(self):
        self.resets += 1
    def _claim_warmup_slot(self):
        pass
    def _play_inner(self):
        if self.solver._claim_warmup_slot():
            self._execute_auto_reset()
        return "jugó"

class Solver:
    def __init__(self):
        self._lock, self._restan = threading.Lock(), 1
    def _claim_warmup_slot(self):
        with self._lock:
            if self._restan <= 0:
                return False
            self._restan -= 1
            return True
'''


class _Servidor(http.server.BaseHTTPRequestHandler):
    listo_desde = 0.0

    def do_GET(self):  # noqa: N802
        ok = time.monotonic() >= _Servidor.listo_desde and self.path == "/health"
        self.send_response(200 if ok else 503)
        self.end_headers()

    def log_message(self, *a):
        pass


def _modulo(tmp, texto):
    ruta = tmp / "ARC3-Inference" / "inference" / "framework" / "solver.py"
    ruta.parent.mkdir(parents=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _importar(ruta):
    spec = importlib.util.spec_from_file_location(f"solver_{id(ruta)}", ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class GuardaArranque(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Servidor)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.env = dict(os.environ)
        os.environ["LOCAL_ANALYZER_BASE_URL"] = f"http://127.0.0.1:{self.srv.server_port}/v1"
        os.environ["VIVOS_ARRANQUE_SONDEO_S"] = "0.05"
        os.environ["VIVOS_ARRANQUE_RESET_S"] = "0.3"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        os.environ.clear()
        os.environ.update(self.env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _jugar(self, demora, n=3, tope="5"):
        os.environ["VIVOS_ARRANQUE_TOPE_S"] = tope
        _Servidor.listo_desde = time.monotonic() + demora
        ruta = _modulo(self.tmp, FALSO)
        self.assertEqual(aplicar(self.tmp), "guarda de arranque aplicada")
        m = _importar(ruta)
        solver = m.Solver()
        sesiones = [m._HarnessGameSession(solver, f"j{i}") for i in range(n)]
        salidas = [None] * n

        def correr(i):
            salidas[i] = sesiones[i]._play_inner()

        t0 = time.monotonic()
        hilos = [threading.Thread(target=correr, args=(i,)) for i in range(n)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(20)
        return sesiones, salidas, time.monotonic() - t0

    def test_servidor_lento_espera_y_un_solo_juego_mantiene_vivo_el_scorecard(self):
        sesiones, salidas, dur = self._jugar(demora=1.0)
        self.assertEqual(salidas, ["jugó"] * 3)
        self.assertGreaterEqual(dur, 0.9)  # nadie juega antes de que el servidor responda
        resets = sorted(s.resets for s in sesiones)
        self.assertEqual(resets[:2], [0, 0])  # solo el guardián hace RESET
        self.assertGreaterEqual(resets[2], 3)  # al entrar y cada 0,3 s (~1 s de espera)

    def test_servidor_sano_no_cambia_nada(self):
        sesiones, salidas, dur = self._jugar(demora=-1)
        self.assertEqual(salidas, ["jugó"] * 3)
        self.assertEqual(sorted(s.resets for s in sesiones), [0, 0, 1])  # el calentamiento original
        self.assertLess(dur, 1.0)

    def test_tope_si_el_servidor_nunca_arranca(self):
        sesiones, salidas, dur = self._jugar(demora=1e9, tope="0.6")
        self.assertEqual(salidas, ["jugó"] * 3)
        self.assertLess(dur, 3.0)

    def test_harness_distinto_no_se_toca(self):
        ruta = _modulo(self.tmp, "x = 1\n")
        self.assertIn("OMITIDA", aplicar(self.tmp))
        self.assertEqual(ruta.read_text(encoding="utf-8"), "x = 1\n")

    def test_idempotente(self):
        _modulo(self.tmp, FALSO)
        aplicar(self.tmp)
        self.assertIn("ya estaba", aplicar(self.tmp))

    @unittest.skipUnless(HARNESS.is_dir(), "falta el harness de Franzen")
    def test_anclas_en_el_harness_real(self):
        dst = self.tmp / "ARC3-Inference" / "inference" / "framework"
        dst.mkdir(parents=True)
        shutil.copy(HARNESS / "inference" / "framework" / "solver.py", dst / "solver.py")
        self.assertEqual(aplicar(self.tmp), "guarda de arranque aplicada")


if __name__ == "__main__":
    unittest.main()
