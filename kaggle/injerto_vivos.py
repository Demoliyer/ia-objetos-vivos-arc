"""Injerta en el Duck harness (versión de Franzen) las herramientas de vivos/simulador.py.

Se ejecuta dentro del notebook de Kaggle, después de aplicar el parche de Franzen y antes de
importar el harness. preparar_franzen.py --vivos pega este archivo en una celda junto con el
código de vivos/simulador.py (_VIVOS_SRC) y llama a aplicar(BUNDLE_DIR / "src", _VIVOS_SRC).

Qué agrega (solo stdlib; el sandbox corre con `python -I -S`):
- en el sandbox del modelo: check_model(step), plan_with_model(step, goal, actions), mouse(r, c);
- en el prompt de sistema (junto a la ayuda de frame_diff): cómo usarlas.
Idempotente: si ya está aplicado no hace nada.
"""

from __future__ import annotations

from pathlib import Path

COLA_SANDBOX = r'''

# --- Objetos Vivos: check_model / plan_with_model (vivos/simulador.py) ---
_VIVOS_SRC = __VIVOS_SRC_REPR__
_VIVOS_GLOBALS = (
    '        runtime_globals["check_model"] = lambda step, **kw: _VIVOS["check_model"]('
    'step, runtime_globals.get("transitions"), runtime_globals.get("current_frame"), **kw)\n'
    '        runtime_globals["plan_with_model"] = lambda step, goal, actions, start=None, **kw: '
    '_VIVOS["plan"](step, goal, start if start is not None else runtime_globals.get("current_frame"), '
    'actions, **kw)\n'
    '        runtime_globals["mouse"] = _VIVOS["mouse"]\n'
)
_VIVOS_ANCLA_GLOBALS = '        runtime_globals["frame_diff"] = frame_diff\n'
_VIVOS_ANCLA_INICIO = "\nHOST_STDOUT = sys.stdout"
if (_SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_GLOBALS) == 1
        and _SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_INICIO) == 1):
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_INICIO,
        "\n_VIVOS = {}\nexec(compile(" + repr(_VIVOS_SRC) + ", '<vivos_simulador>', 'exec'), _VIVOS)\n"
        + _VIVOS_ANCLA_INICIO,
        1,
    )
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_GLOBALS, _VIVOS_ANCLA_GLOBALS + _VIVOS_GLOBALS, 1
    )
else:  # harness distinto al esperado: se sigue sin el injerto antes que romper la corrida
    logging.getLogger(__name__).warning("objetos_vivos: anclas no encontradas; injerto omitido")
'''

AYUDA_PROMPT = (
    "- World-model tools (they cost no game actions): write your hypothesis of the mechanics as a "
    "top-level Python function `step(grid, action)`. `grid` is the board as a list of rows, each a "
    "list of color characters exactly as in `.ascii`; `action` is 'UP', 'DOWN', 'LEFT', 'RIGHT', "
    "'SPACE', 'UNDO' or 'MOUSE(row=R, col=C)' (`mouse(r, c)` builds one). Return the predicted next "
    "board, or only the cells you predict to change as a dict {(row, col): color}.\n"
    "- `check_model(step)` replays every recorded transition of the current level through `step` "
    "and returns `checked`, `interior_ok` (boards match outside the 4-cell border, where budget bars "
    "live), `all_interior_ok`, and the first `mismatches` with the action and example cells "
    "`(row, col, actual, predicted)`. UNDO and RESET transitions are not replayed. Fix `step` until `all_interior_ok` is True; a simulator that has "
    "not passed `check_model` is only a guess.\n"
    "- `plan_with_model(step, goal, actions)` searches breadth-first inside your simulator from "
    "`current_frame` and returns `{'plan': [...], 'length': n}` with the shortest action list that "
    "reaches a board where `goal(grid)` is True (or that equals a goal board inside the border), or "
    "`{'plan': None, 'reason': ...}`. `actions` is the list of labels to try in every state. Capped at "
    "4000 states and 8 s (`max_nodes=`, `time_limit=`); all calls in one snippet share a ~24 s budget. Execute the plan with `action(...)`; if the "
    "real board diverges from the prediction, your `step` is wrong: re-run `check_model`.\n"
    "- Keep everything `step` needs (constants, lookup tables, helpers) inside `step` itself so it is "
    "retained across calls. Use these tools when the mechanics look deterministic and you can state "
    "them in code; they turn exploration into verified planning and save actions, which improves the "
    "score.\n"
)

COLA_PROMPTS = (
    "\n\n# --- Objetos Vivos: ayuda de check_model / plan_with_model ---\n"
    "FRAME_DIFF_HINT_ADDENDUM = FRAME_DIFF_HINT_ADDENDUM + " + repr(AYUDA_PROMPT) + "\n"
)


# Modo auto: después de cada snippet del modelo, si hay un `step` (propio o retenido) y el
# snippet no llamó a check_model, el sandbox lo contrasta solo y agrega una línea a la salida.
# Aislado (revisión 2026-10-17): stdout propio, tope duro con SIGALRM, action() deshabilitada
# y BaseException atrapada, para que un step con print, bucle o acciones no rompa el snippet.
COLA_AUTO = r'''

# --- Objetos Vivos: auto-chequeo del step retenido después de cada snippet ---
_VIVOS_AUTO_DEF = (
    "\nclass _VivosCorte(BaseException):\n    pass\n"
    "\ndef _vivos_auto(_salida, _g, _codigo):\n"
    "    import signal as _sig\n"
    "    _accion = _g.get('action')\n"
    "    _previo = None\n"
    "    try:\n"
    "        if 'check_model' in _codigo or not callable(_g.get('step')):\n"
    "            return\n"
    "        def _corte(*_a):\n"
    "            raise _VivosCorte()\n"
    "        def _sin_accion(*_a, **_k):\n"
    "            raise RuntimeError('action() is disabled inside step during the auto-check')\n"
    "        _g['action'] = _sin_accion\n"
    "        _previo = _sig.signal(_sig.SIGALRM, _corte)\n"
    "        import time as _t\n"
    "        _resta = 26.5 - (_t.monotonic() - _VIVOS['_T0'])\n"
    "        if _resta < 0.3:\n"
    "            return\n"
    "        _sig.setitimer(_sig.ITIMER_REAL, min(2.5, _resta))\n"
    "        with contextlib.redirect_stdout(io.StringIO()):\n"
    "            _m = _VIVOS['auto_check'](_g['step'], _g.get('transitions'), _g.get('current_frame'))\n"
    "        _sig.setitimer(_sig.ITIMER_REAL, 0)\n"
    "        if _m:\n"
    "            _salida.write('\\n' + _m + '\\n')\n"
    "    except BaseException:\n"
    "        pass\n"
    "    finally:\n"
    "        try:\n"
    "            _sig.setitimer(_sig.ITIMER_REAL, 0)\n"
    "            if _previo is not None:\n"
    "                _sig.signal(_sig.SIGALRM, _previo)\n"
    "        except BaseException:\n"
    "            pass\n"
    "        if _accion is not None:\n"
    "            _g['action'] = _accion\n"
)
_VIVOS_ANCLA_FINAL = '        _send(\n            {\n                "type": "final",'
if (_SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_FINAL) == 1 and "_VIVOS = {}" in _SANDBOX_BOOTSTRAP
        and _SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_INICIO) == 1):
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_INICIO, _VIVOS_AUTO_DEF + _VIVOS_ANCLA_INICIO, 1)
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_FINAL,
        "        _vivos_auto(stdout, runtime_globals, _code_text)\n" + _VIVOS_ANCLA_FINAL, 1)
else:
    logging.getLogger(__name__).warning("objetos_vivos: anclas del auto-chequeo no encontradas")
'''

RECORDATORIO_SIN_STEP = (
    "[World model] No `step(grid, action)` simulator is retained yet. As soon as you have a "
    "hypothesis of what the actions do, write it as `def step(grid, action)` in a Python call and "
    "run `print(check_model(step))`: it replays every observed transition of this level at zero "
    "action cost and shows exactly where the hypothesis is wrong. A partial theory is fine: `step` "
    "may return only the cells it changes, as a dict {(row, col): color}."
)
RECORDATORIO_CON_STEP = (
    "[World model] Your `step` simulator is retained. After new observations re-run "
    "`print(check_model(step))`; once `all_interior_ok` is True, call "
    "`plan_with_model(step, goal, actions)` and execute the returned plan instead of exploring by hand."
)

# Modo fuerte: un recordatorio de una línea al final de cada mensaje de turno. Envuelve
# ToolAgent._retained_function_context (harness de Franzen); si falla, devuelve lo original.
COLA_AGENTE = """

# --- Objetos Vivos: recordatorio por turno de check_model / plan_with_model ---
def _vivos_envolver_contexto():
    _orig = ToolAgent._retained_function_context
    _sin, _con = __SIN__, __CON__

    def _retained_function_context(self):
        base = _orig(self)
        try:
            kept = getattr(self, "_kept_functions", None) or {}
            return base + "\\n\\n" + (_con if "step" in kept else _sin)
        except Exception:
            return base

    ToolAgent._retained_function_context = _retained_function_context


try:
    _vivos_envolver_contexto()
except Exception:  # la corrida oficial sigue sin el recordatorio
    pass
"""


def aplicar(src_root, vivos_src: str, fuerte: bool = False, auto: bool = False) -> str:
    """Nunca lanza: si algo no coincide, avisa y deja el harness como estaba.
    auto=True implica fuerte=True."""
    try:
        return _aplicar(src_root, vivos_src, fuerte or auto, auto)
    except Exception as exc:  # la re-ejecución oficial tiene que seguir aunque falle el injerto
        return f"objetos_vivos: injerto OMITIDO ({type(exc).__name__}: {exc})"


def _aplicar(src_root, vivos_src: str, fuerte: bool = False, auto: bool = False) -> str:
    raiz = Path(src_root) / "ARC3-Inference" / "inference" / "agent"
    sandbox, prompts = raiz / "python_tool_sandbox.py", raiz / "prompts.py"
    s = sandbox.read_text(encoding="utf-8")
    if "_VIVOS_SRC" in s:
        return "objetos_vivos: injerto ya aplicado"
    if "_SANDBOX_BOOTSTRAP" not in s:
        raise RuntimeError("objetos_vivos: python_tool_sandbox.py no tiene _SANDBOX_BOOTSTRAP")
    # Sin las anclas el sandbox no recibe las herramientas: tampoco hay que anunciarlas en el prompt.
    for ancla in ('runtime_globals["frame_diff"] = frame_diff', "HOST_STDOUT = sys.stdout"):
        if s.count(ancla) != 1:
            raise RuntimeError(f"objetos_vivos: ancla del sandbox no encontrada: {ancla!r}")
    p = prompts.read_text(encoding="utf-8")
    if "FRAME_DIFF_HINT_ADDENDUM = (" not in p:
        raise RuntimeError("objetos_vivos: prompts.py no tiene FRAME_DIFF_HINT_ADDENDUM")
    compile(vivos_src, "<vivos_simulador>", "exec")  # que al menos sea Python válido
    cola_agente = ""
    agente = raiz / "tool_agent.py"
    if fuerte:
        a = agente.read_text(encoding="utf-8")
        if "class ToolAgent" not in a or "def _retained_function_context(self)" not in a:
            raise RuntimeError("objetos_vivos: tool_agent.py sin _retained_function_context")
        cola_agente = (COLA_AGENTE.replace("__SIN__", repr(RECORDATORIO_SIN_STEP))
                       .replace("__CON__", repr(RECORDATORIO_CON_STEP)))
        compile(a + cola_agente, str(agente), "exec")
    cola = COLA_SANDBOX.replace("__VIVOS_SRC_REPR__", repr(vivos_src)) + (COLA_AUTO if auto else "")
    compile(s + cola, str(sandbox), "exec")
    sandbox.write_text(s + cola, encoding="utf-8")
    prompts.write_text(p + COLA_PROMPTS, encoding="utf-8")
    if cola_agente:
        agente.write_text(a + cola_agente, encoding="utf-8")
        return ("objetos_vivos: check_model / plan_with_model injertados + recordatorio por turno"
                + (" + auto-chequeo" if auto else ""))
    return "objetos_vivos: check_model / plan_with_model injertados"
