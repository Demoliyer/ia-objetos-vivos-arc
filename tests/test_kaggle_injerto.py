"""Injerto vivos en el harness de Franzen y preparación del notebook de Kaggle.

Necesitan material que no está en el repo (código y notebook de terceros); si falta, se saltean:
- harness: git clone https://github.com/da-fr/arc-agi-3-solution en ../franzen (o VIVOS_HARNESS)
- notebook: kaggle/franzen_nb/original.ipynb (descargado de Kaggle)
"""

from __future__ import annotations

import copy
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "kaggle"))

from injerto_vivos import aplicar  # noqa: E402
from preparar_franzen import CELDA_GPU, MARCA_CONFIG, MARCA_ENV, _texto, preparar  # noqa: E402

HARNESS = Path(os.environ.get("VIVOS_HARNESS", RAIZ.parent / "franzen" / "ARC3-Inference"))
NOTEBOOK = RAIZ / "kaggle" / "franzen_nb" / "original.ipynb"
SIMULADOR = (RAIZ / "vivos" / "simulador.py").read_text(encoding="utf-8")


@unittest.skipUnless(HARNESS.is_dir(), "falta el harness de Franzen")
class InjertoEnHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vivos_injerto_"))
        shutil.copytree(HARNESS, self.tmp / "ARC3-Inference")
        self.agente = self.tmp / "ARC3-Inference" / "inference" / "agent"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_injerta_compila_y_es_idempotente(self):
        self.assertIn("injertados", aplicar(self.tmp, SIMULADOR))
        self.assertIn("ya aplicado", aplicar(self.tmp, SIMULADOR))
        for nombre in ("python_tool_sandbox.py", "prompts.py"):
            py_compile.compile(str(self.agente / nombre), doraise=True)

    def test_herramientas_responden_en_el_sandbox_real(self):
        aplicar(self.tmp, SIMULADOR)
        codigo = (
            "import json, sys\n"
            "from inference.agent import python_tool_sandbox as sb, prompts\n"
            "assert 'plan_with_model' in prompts.FRAME_DIFF_HINT_ADDENDUM\n"
            "estado = {'current_frame': {'ascii': 'WW\\nWB', 'step': 1, 'level': 0, 'shape': [2, 2],"
            " 'grid': [[0, 0], [0, 5]]}, 'history': [], 'valid_actions': ['UP']}\n"
            "out = sb.run_sandboxed_python(code=\"print(check_model(lambda g, a: g)['checked'], "
            "plan_with_model(lambda g, a: g, lambda g: True, ['UP'])['length'], mouse(1, 2))\","
            " timeout_seconds=20, initial_state=estado, action_handler=lambda *a, **k: {})\n"
            "print(json.dumps(out))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(self.tmp / "ARC3-Inference")
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertFalse(out.get("error"), out)
        self.assertIn("0 0 MOUSE(row=1, col=2)", out["stdout"])

    def test_modo_fuerte_agrega_recordatorio_por_turno(self):
        self.assertIn("recordatorio", aplicar(self.tmp, SIMULADOR, fuerte=True))
        py_compile.compile(str(self.agente / "tool_agent.py"), doraise=True)
        codigo = (
            "from types import SimpleNamespace as NS\n"
            "from inference.agent.tool_agent import ToolAgent\n"
            "f = ToolAgent._retained_function_context\n"
            "print(repr(f(NS(_kept_functions={}))))\n"
            "print(repr(f(NS(_kept_functions={'step': 'def step(g, a): return g'}))))\n"
            "print(repr(f(None)))\n"
            "a = NS(_kept_functions={'step': 'def step(g, a): return g'}, _vivos_paso=7)\n"
            "for i in range(5):\n"
            "    t = f(a)\n"
            "print(repr(t))\n"
            "a._vivos_paso = 8\n"
            "print(repr(f(a)))\n"
        )
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "ARC3_PERSISTENT_FUNCTIONS")}
        env["PYTHONPATH"] = str(self.tmp / "ARC3-Inference")
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        sin, con, nada, freno, quieto = r.stdout.strip().splitlines()[-5:]
        # Freno apagado desde vivos-v4s (sospechoso de la caída de v3: 7 disparos en vc33 con un
        # step perfecto); con 5 turnos sin jugar el recordatorio sigue siendo el normal.
        self.assertNotIn("turns in a row without a single game action", freno)
        self.assertIn("Your `step` simulator is retained", freno)
        self.assertIn("Your `step` simulator is retained", quieto)  # hubo jugada: se resetea
        self.assertIn("No `step(grid, action)` simulator is retained yet", sin)
        self.assertIn("Your `step` simulator is retained", con)
        self.assertIn("No `step", nada)  # objeto sin atributos: no lanza

    def test_modo_auto_chequea_el_step_despues_del_snippet(self):
        self.assertIn("auto-chequeo", aplicar(self.tmp, SIMULADOR, auto=True))
        py_compile.compile(str(self.agente / "python_tool_sandbox.py"), doraise=True)
        vacio = [["." for _ in range(10)] for _ in range(10)]
        a, b = [f[:] for f in vacio], [f[:] for f in vacio]
        a[4][4] = "X"
        b[4][5] = "X"
        frame = lambda g: {"ascii": "\n".join("".join(f) for f in g), "step": 0, "level": 0,  # noqa: E731
                           "shape": [10, 10], "grid": [[0] * 10 for _ in range(10)]}
        estado = {"current_frame": frame(b), "valid_actions": ["RIGHT"],
                  "history": [{"action": "", "frame": frame(a)},
                              {"action": "RIGHT", "frame": frame(b), "result": {}}]}
        codigos = {
            "delta": "def step(grid, action):\n    return {(4, 4): '.', (4, 5): 'X'}\nprint('hola')",
            "propio": "def step(grid, action):\n    return grid\nprint(check_model(step)['checked'])",
            "sin_step": "print('nada')",
        }
        codigo = (
            "import json\n"
            "from inference.agent import python_tool_sandbox as sb\n"
            f"estado = {estado!r}\n"
            "for nombre, code in " + repr(codigos) + ".items():\n"
            "    out = sb.run_sandboxed_python(code=code, timeout_seconds=20, initial_state=estado,"
            " action_handler=lambda *a, **k: {})\n"
            "    print(nombre, json.dumps(out.get('stdout', '')))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(self.tmp / "ARC3-Inference")
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env,
                           timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        salidas = dict(l.split(" ", 1) for l in r.stdout.strip().splitlines()[-3:])
        self.assertIn("it reproduces 1/1 recorded transitions", salidas["delta"])
        self.assertIn("hola", salidas["delta"])
        self.assertNotIn("auto-check", salidas["propio"])  # el modelo ya llamó a check_model
        self.assertNotIn("auto-check", salidas["sin_step"])

    def test_modo_auto_aguanta_steps_hostiles(self):
        """Revisión 2026-10-17: un step con print, bucle infinito o action() no rompe el snippet."""
        aplicar(self.tmp, SIMULADOR, auto=True)
        vacio = [["." for _ in range(10)] for _ in range(10)]
        frame = {"ascii": "\n".join("".join(f) for f in vacio), "step": 0, "level": 0,
                 "shape": [10, 10], "grid": [[0] * 10 for _ in range(10)]}
        estado = {"current_frame": frame, "valid_actions": ["RIGHT"],
                  "history": [{"action": "", "frame": frame}]
                  + [{"action": "RIGHT", "frame": frame, "result": {}} for _ in range(6)]}
        casos = {
            "print_texto": ("def step(g, a):\n    print('dbg', a)\n    return g\nprint('ok')", None),
            "print_json": ("def step(g, a):\n    print(1)\n    return g\nprint('ok')", None),
            "retenido_print": ("print('ok')", ["def step(g, a):\n    print('dbg')\n    return g\n"]),
            "bucle": ("def step(g, a):\n    while True: pass\nprint('ok')", None),
            "llama_action": ("def step(g, a_):\n    action('RIGHT')\n    return g\nprint('ok')", None),
            "en_el_lugar": ("def step(g, a):\n    g[5][5] = 'Z'\nprint('ok')", None),
        }
        codigo = (
            "import json, time\n"
            "from inference.agent import python_tool_sandbox as sb\n"
            f"estado = {estado!r}\n"
            "n = [0]\n"
            "def ah(*a, **k):\n"
            "    n[0] += 1\n"
            "    return {'action_result': {'executed': True, 'gameplay_changed': True}, 'state': estado}\n"
            "for nombre, (code, kept) in " + repr(casos) + ".items():\n"
            "    n[0] = 0; t = time.monotonic()\n"
            "    out = sb.run_sandboxed_python(code=code, timeout_seconds=30, initial_state=estado,"
            " action_handler=ah, kept_functions=kept)\n"
            "    print(nombre, json.dumps({'stdout': out.get('stdout', ''), 'error': out.get('error'),"
            " 'acciones': n[0], 'seg': time.monotonic() - t}))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(self.tmp / "ARC3-Inference")
        env["ARC3_PERSISTENT_FUNCTIONS"] = "1"
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env,
                           timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        res = {l.split(" ", 1)[0]: json.loads(l.split(" ", 1)[1]) for l in r.stdout.strip().splitlines()
               if l.split(" ", 1)[0] in casos}
        self.assertEqual(set(res), set(casos), r.stdout[-2000:])
        for nombre, o in res.items():
            self.assertFalse(o["error"], (nombre, o))
            self.assertIn("ok", o["stdout"], (nombre, o))
            self.assertNotIn("dbg", o["stdout"], (nombre, o))
            self.assertEqual(o["acciones"], 0, (nombre, o))
            self.assertLess(o["seg"], 15, (nombre, o))

    def test_modo_auto_estado_y_sin_repetir(self):
        """v4: observe/advance se chequean solos; el mismo auto-chequeo dos veces se acorta."""
        aplicar(self.tmp, SIMULADOR, auto=True)
        vacio = [["." for _ in range(10)] for _ in range(10)]
        a, b = [f[:] for f in vacio], [f[:] for f in vacio]
        a[4][4] = "X"
        b[4][5] = "X"
        frame = lambda g: {"ascii": "\n".join("".join(f) for f in g), "step": 0, "level": 0,  # noqa: E731
                           "shape": [10, 10], "grid": [[0] * 10 for _ in range(10)]}
        estado = {"current_frame": frame(b), "valid_actions": ["RIGHT"],
                  "history": [{"action": "", "frame": frame(a)},
                              {"action": "RIGHT", "frame": frame(b), "result": {}}]}
        obs = ("def observe(g):\n    return {'x': [r for r in range(10) for c in range(10) if g[r][c]=='X'][0],"
               " 'y': [c for r in range(10) for c in range(10) if g[r][c]=='X'][0]}\n")
        codigos = {
            "bien": obs + "def advance(s, a):\n    return {'x': s['x'], 'y': s['y'] + 1}\nprint('hola')",
            "mal": obs + "def advance(s, a):\n    return {'x': s['x'] + 1, 'y': s['y'] + 1}\nprint('hola')",
            "propio": obs + "def advance(s, a):\n    return s\nprint(check_state(observe, advance)['checked'])",
            "plan": obs + "def advance(s, a):\n    return {'x': s['x'], 'y': s['y'] + 1}\n"
                    "print(plan_with_state(observe, advance, lambda s: s['y'] == 8, ['RIGHT'])['length'])",
            "con_step": obs + "def advance(s, a):\n    return s\n"
                        "def step(grid, action):\n    return {(4, 4): '.', (4, 5): 'X'}\nprint('hola')",
            "constante": "DIRS = {'RIGHT': 1}\n" + obs + "def advance(s, a):\n    return {'x': s['x'], 'y': s['y'] + DIRS[a]}\nprint('hola')",
            "constante_ajena": "OTRA = 5\n" + obs + "def advance(s, a):\n    return {'x': s['x'], 'y': s['y'] + 1}\nprint('hola')",
            "except_pelado": obs + "def advance(s, a):\n    while True:\n        try:\n            sum(range(10**6))\n"
                             "        except:\n            pass\nprint('hola')",
        }
        codigo = (
            "import json\n"
            "from types import SimpleNamespace as NS\n"
            "from inference.agent import python_tool_sandbox as sb\n"
            "from inference.agent import tool_agent as ta\n"
            f"estado = {estado!r}\n"
            "for nombre, code in " + repr(codigos) + ".items():\n"
            "    out = sb.run_sandboxed_python(code=code, timeout_seconds=20, initial_state=estado,"
            " action_handler=lambda *a, **k: {})\n"
            "    print(nombre, json.dumps(out.get('stdout', '')))\n"
            "ag = NS()\n"
            "r1 = ta._vivos_no_repetir(ag, {'stdout': 'x\\n[auto-check of your step] 0/3'})\n"
            "r2 = ta._vivos_no_repetir(ag, {'stdout': 'y\\n[auto-check of your step] 0/3\\n'})\n"
            "r3 = ta._vivos_no_repetir(ag, {'stdout': 'z\\n[auto-check of your step] 1/3'})\n"
            "r4 = ta._vivos_no_repetir(None, {'stdout': 'w'})\n"
            "ta._vivos_no_repetir(ag, {'stdout': 'error sin auto-chequeo'})\n"
            "r5 = ta._vivos_no_repetir(ag, {'stdout': 'z\\n[auto-check of your step] 1/3'})\n"
            "ta._vivos_no_repetir(ag, {'stdout': 'a\\n[auto-check of your step] 3/3 It explains everything'})\n"
            "r6 = ta._vivos_no_repetir(ag, {'stdout': 'b\\n[auto-check of your step] 3/3 It explains everything'})\n"
            "print('no_acorta', json.dumps(r6['stdout']))\n"
            "print('rep', json.dumps([r1['stdout'], r2['stdout'], r3['stdout'], r4['stdout'], r5['stdout']]))\n"
            "print('envuelto', ta.run_sandboxed_python.__name__, ta.ToolAgent._run_python_tool.__name__)\n"
            "vistos = []\n"
            "def falso(**k):\n"
            "    vistos.append(k['initial_state'].get('_vivos_modo'))\n"
            "    return {'stdout': ''}\n"
            "import inspect\n"
            "_cl = inspect.getclosurevars(ta.run_sandboxed_python).nonlocals\n"
            "ctx = _cl['_ctx']\n"
            "ta.run_sandboxed_python.__closure__[[c.cell_contents is _cl['_orig_sandbox'] for c in ta.run_sandboxed_python.__closure__].index(True)].cell_contents = falso\n"
            "ctx.agente = NS()\n"
            "for code in ['def observe(g):\\n    return {}', 'action(\"UP\")', 'def step(g, a):\\n    return g', 'x = 1']:\n"
            "    ta.run_sandboxed_python(code=code, initial_state={})\n"
            "print('modos', json.dumps(vistos))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(self.tmp / "ARC3-Inference")
        r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, env=env,
                           timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        salidas = dict(l.split(" ", 1) for l in r.stdout.strip().splitlines()[-12:])
        self.assertIn("your functions use DIRS", salidas["constante"])
        self.assertIn("predicts the next state in 1/1", salidas["constante"])
        self.assertNotIn("Warning", salidas["constante_ajena"])
        self.assertIn("explains everything", salidas["no_acorta"])  # lo que está bien no se acorta
        self.assertIn("it reproduces 1/1", salidas["con_step"])
        self.assertIn("hola", salidas["except_pelado"])
        self.assertNotIn("plan_with_state(observe, advance, goal", salidas["plan"])
        self.assertIn("predicts the next state in 1/1", salidas["bien"])
        self.assertIn("hola", salidas["bien"])
        self.assertIn("Field x 0/1", salidas["mal"])
        self.assertNotIn("auto-check", salidas["propio"])
        self.assertIn("3", salidas["plan"])
        rep = json.loads(salidas["rep"])
        self.assertIn("0/3", rep[0])
        self.assertIn("UNCHANGED: same report as after your previous snippet", rep[1])
        self.assertNotIn("fails", rep[1])  # neutral: el informe repetido puede ser positivo (Astra)
        self.assertNotIn("0/3", rep[1])
        self.assertIn("1/3", rep[2])
        self.assertEqual(rep[3], "w")
        self.assertIn("1/3", rep[4])  # hubo un snippet sin auto-chequeo en el medio: no se acorta
        self.assertEqual(salidas["envuelto"], "_sandbox _run_python_tool")
        self.assertEqual(json.loads(salidas["modos"]), ["state", "state", "step", "step"])

    def test_harness_ajeno_no_lanza_ni_toca_el_prompt(self):
        prompts = self.agente / "prompts.py"
        sandbox = self.agente / "python_tool_sandbox.py"
        sandbox.write_text("_SANDBOX_BOOTSTRAP = 'nada'\n", encoding="utf-8")
        antes = prompts.read_text(encoding="utf-8")
        self.assertIn("OMITIDO", aplicar(self.tmp, SIMULADOR))
        self.assertEqual(prompts.read_text(encoding="utf-8"), antes)


@unittest.skipUnless(NOTEBOOK.is_file(), "falta kaggle/franzen_nb/original.ipynb")
class PrepararNotebook(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def _indice_env(self, nb):
        return next(i for i, c in enumerate(nb["cells"]) if MARCA_ENV in _texto(c))

    def test_orden_de_celdas_con_vivos(self):
        out = preparar(copy.deepcopy(self.nb), {"ARC3_X": "1"}, rapido=True, vivos=True)
        i = self._indice_env(out)
        self.assertEqual(len(out["cells"]), len(self.nb["cells"]) + 2)
        self.assertIn("nvidia-smi", _texto(out["cells"][i + 1]))
        self.assertIn("aplicar(BUNDLE_DIR", _texto(out["cells"][i + 2]))
        self.assertIn("setup_env['ARC3_X'] = '1'", _texto(out["cells"][i]))
        self.assertTrue(out["metadata"]["kaggle"]["isGpuEnabled"])
        self.assertFalse(out["metadata"]["kaggle"]["isInternetEnabled"])

    def test_cambios_de_guardado_no_tocan_la_reejecucion_oficial(self):
        out = preparar(copy.deepcopy(self.nb), {}, rapido=True, vivos=False)
        config = next(_texto(c) for c in out["cells"] if MARCA_CONFIG in _texto(c))
        self.assertIn("if COMMIT_RAPIDO and not TRUE_SUBMISSION:", config)
        self.assertIn("if not TRUE_SUBMISSION:", CELDA_GPU)

    def test_completo_solo_cambia_variables(self):
        out = preparar(copy.deepcopy(self.nb), {"ARC3_X": "1"}, rapido=False, vivos=False)
        self.assertEqual(len(out["cells"]), len(self.nb["cells"]))
        textos = [_texto(c) for c in out["cells"]]
        self.assertFalse(any("COMMIT_RAPIDO" in t or "Objetos Vivos: en el guardado" in t for t in textos))
        originales = [_texto(c) for c in self.nb["cells"]]
        distintas = [i for i, (a, b) in enumerate(zip(originales, textos)) if a != b]
        self.assertEqual(distintas, [self._indice_env(self.nb)])

    def test_cfg_cambia_solo_las_claves_pedidas(self):
        out = preparar(copy.deepcopy(self.nb), {}, rapido=False, vivos=False,
                       cfg={"MAXREQ": "14", "MAMBA_CACHE": "84"})
        textos = [_texto(c) for c in out["cells"]]
        originales = [_texto(c) for c in self.nb["cells"]]
        distintas = [i for i, (a, b) in enumerate(zip(originales, textos)) if a != b]
        self.assertEqual(len(distintas), 1)
        antes, despues = originales[distintas[0]].splitlines(), textos[distintas[0]].splitlines()
        cambiadas = [b for a, b in zip(antes, despues) if a != b]
        self.assertEqual(len(antes), len(despues))
        self.assertEqual([c.split("=")[0].strip() for c in cambiadas], ["MAXREQ", "MAMBA_CACHE"])
        compile(re.sub(r"(?m)^(\s*)!", r"\1#", textos[distintas[0]]), "<cfg>", "exec")

    def test_vivos_fuerte_en_la_celda(self):
        out = preparar(copy.deepcopy(self.nb), {}, rapido=True, fuerte=True)
        i = self._indice_env(out)
        self.assertIn("fuerte=True", _texto(out["cells"][i + 2]))
        auto = preparar(copy.deepcopy(self.nb), {}, rapido=True, auto=True)
        self.assertIn("fuerte=True, auto=True", _texto(auto["cells"][i + 2]))
        normal = preparar(copy.deepcopy(self.nb), {}, rapido=True, vivos=True)
        self.assertIn("fuerte=False", _texto(normal["cells"][i + 2]))

    def test_cfg_invalido_corta(self):
        with self.assertRaises(SystemExit):
            preparar(copy.deepcopy(self.nb), {}, rapido=True, cfg={"NOEXISTE": "1"})
        with self.assertRaises(SystemExit):
            preparar(copy.deepcopy(self.nb), {}, rapido=True, cfg={"MAXREQ": "import os"})
        with self.assertRaises(SystemExit):  # el servidor no arranca con 5 tokens de borrador
            preparar(copy.deepcopy(self.nb), {}, rapido=True, cfg={"SPEC_STEPS": "4"})

    def test_juegos_y_minutos_del_guardado_rapido(self):
        out = preparar(copy.deepcopy(self.nb), {}, rapido=True, juegos=["ls20", "vc33", "tu93"],
                       minutos=60)
        config = next(_texto(c) for c in out["cells"] if MARCA_CONFIG in _texto(c))
        self.assertIn("bm.solver.concurrency = 3", config)
        self.assertIn("max_runtime_s_per_game = 60 * 60", config)
        self.assertNotIn("'vc33'", config.split("demo_excluded_games =")[1].splitlines()[0])
        with self.assertRaises(SystemExit):
            preparar(copy.deepcopy(self.nb), {}, rapido=True, juegos=["zz99"])


if __name__ == "__main__":
    unittest.main()
