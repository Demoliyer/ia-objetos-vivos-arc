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
_VIVOS_MOTOR_SRC = __VIVOS_MOTOR_SRC_REPR__  # candidata M: predictor puro del motor ("" = apagado)
_VIVOS_GLOBALS = (
    '        runtime_globals["check_model"] = lambda step, **kw: _VIVOS["check_model"]('
    'step, runtime_globals.get("transitions"), runtime_globals.get("current_frame"), **kw)\n'
    '        runtime_globals["plan_with_model"] = lambda step, goal, actions, start=None, **kw: '
    '_VIVOS["plan"](step, goal, start if start is not None else runtime_globals.get("current_frame"), '
    'actions, **kw)\n'
    '        runtime_globals["mouse"] = _VIVOS["mouse"]\n'
    '        runtime_globals["check_state"] = lambda observe, advance, **kw: _VIVOS["check_state"]('
    'observe, advance, runtime_globals.get("transitions"), runtime_globals.get("current_frame"), **kw)\n'
    '        runtime_globals["plan_with_state"] = lambda observe, advance, goal, actions, start=None, **kw: '
    '_VIVOS["plan_state"](observe, advance, goal, start if start is not None else '
    'runtime_globals.get("current_frame"), actions, **kw)\n'
)
_VIVOS_GLOBALS += (
    '        _vivos_m = (initial.get("state") or {}).get("_vivos_motor")\n'
    '        if isinstance(_vivos_m, dict) and _VIVOS_M is not None:\n'
    '            try:\n'
    '                _VIVOS["_motor"] = _VIVOS_M["preparar"](_vivos_m)\n'
    '                runtime_globals["engine_step"] = lambda grid, action: _VIVOS_M["engine_step"]('
    '_VIVOS["_motor"], grid, action)\n'
    '            except Exception:\n'
    '                pass\n'
) if _VIVOS_MOTOR_SRC else ""
_VIVOS_ANCLA_GLOBALS = '        runtime_globals["frame_diff"] = frame_diff\n'
_VIVOS_ANCLA_INICIO = "\nHOST_STDOUT = sys.stdout"
if (_SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_GLOBALS) == 1
        and _SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_INICIO) == 1):
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_INICIO,
        "\n_VIVOS = {}\nexec(compile(" + repr(_VIVOS_SRC) + ", '<vivos_simulador>', 'exec'), _VIVOS)\n"
        + ("_VIVOS_M = {}\ntry:\n    exec(compile(" + repr(_VIVOS_MOTOR_SRC) + ", '<vivos_motor>', 'exec'), _VIVOS_M)\n"
           "except Exception:\n    _VIVOS_M = None\n" if _VIVOS_MOTOR_SRC else "_VIVOS_M = None\n")
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
    "4000 states and 8 s (`max_nodes=`, `time_limit=`); all calls in one snippet share a ~24 s budget. With `cost=True`, `goal` "
    "returns a number instead, a distance to the goal (0 = reached, lower is better): the search is best-first and, if it "
    "cannot reach 0, it still returns `best_plan` / `best_cost`, the path that gets closest. Execute the plan with `action(...)`; if the "
    "real board diverges from the prediction, your `step` is wrong: re-run `check_model`.\n"
    "- Often easier than a full-board `step`: a state model. `observe(grid)` returns a small state, "
    "best a dict of named fields such as {'player': (r, c), 'keys': 2, 'doors': frozenset(...)}, found "
    "by color/shape so it works on any level; `advance(state, action)` returns the next state. "
    "`check_state(observe, advance)` replays the recorded transitions and checks every field "
    "separately (which field is wrong, with which action), and reports blind spots: board changes "
    "your state does not capture. `plan_with_state(observe, advance, goal, actions)` searches over "
    "states (`goal(state)` -> True; `cost=True` as above), far faster than over boards. `advance` "
    "never sees the board, so walls or obstacles that block moves must be in the state. Fix only "
    "the field that fails instead of rewriting everything.\n"
    "- Only `def` functions persist between snippets; top-level variables (e.g. `DIRS = {...}`) "
    "disappear. Keep everything `step` (or `observe` / `advance`) needs inside those functions or in "
    "helper `def`s. Use these tools when the mechanics look deterministic and you can state "
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
    "\ndef _vivos_auto(_salida, _g, _codigo, _final=None, _modo=None):\n"
    "    import signal as _sig\n"
    "    _accion = _g.get('action')\n"
    "    _previo = None\n"
    "    import threading as _hilos\n"
    "    _listo = _hilos.Event()\n"
    "    try:\n"
    "        _obs, _adv = _g.get('observe'), _g.get('advance')\n"
    "        _estado = callable(_obs) and callable(_adv)\n"
    "        if _estado and callable(_g.get('step')):\n"
    "            if 'def step(' in _codigo:\n"
    "                _estado = False\n"
    "            elif not ('def observe(' in _codigo or 'def advance(' in _codigo) and _modo == 'step':\n"
    "                _estado = False\n"
    "        if _estado and 'check_state' in _codigo:\n"
    "            return\n"
    "        if not _estado and ('check_model' in _codigo or not callable(_g.get('step'))):\n"
    "            return\n"
    "        def _corte(*_a):\n"
    "            raise _VivosCorte()\n"
    "        def _sin_accion(*_a, **_k):\n"
    "            raise RuntimeError('action() is disabled inside your model functions during the auto-check')\n"
    "        _g['action'] = _sin_accion\n"
    "        _previo = _sig.signal(_sig.SIGALRM, _corte)\n"
    "        import time as _t\n"
    "        _resta = 26.5 - (_t.monotonic() - _VIVOS['_T0'])\n"
    "        if _resta < 0.3:\n"
    "            return\n"
    "        _tope = min(2.5, _resta)\n"
    "        if _final is not None:\n"
    "            # vigía: si un `except:` pelado se traga la alarma, entrega lo ya hecho y cierra\n"
    "            import threading as _hilos\n"
    "            def _vigia():\n"
    "                if not _listo.wait(_tope + 0.6):\n"
    "                    try:\n"
    "                        _final()\n"
    "                    finally:\n"
    "                        import os as _os\n"
    "                        _os._exit(0)\n"
    "            _hilos.Thread(target=_vigia, daemon=True).start()\n"
    "        _sig.setitimer(_sig.ITIMER_REAL, _tope)\n"
    "        with contextlib.redirect_stdout(io.StringIO()):\n"
    "            if _estado:\n"
    "                _m = _VIVOS['auto_check_state'](_obs, _adv, _g.get('transitions'), _g.get('current_frame'), time_limit=2.0)\n"
    "            else:\n"
    "                _m = _VIVOS['auto_check'](_g['step'], _g.get('transitions'), _g.get('current_frame'), time_limit=2.0)\n"
    "        _sig.setitimer(_sig.ITIMER_REAL, 0)\n"
    "        _fs = [_obs, _adv] if _estado else [_g.get('step')]\n"
    "        _fs += [_v for _k, _v in _g.items() if callable(_v) and getattr(getattr(_v, '__code__', None), 'co_filename', '') == '<python_tool>' and _v not in _fs]\n"
    "        _perd = _VIVOS['nombres_que_se_pierden'](_fs, _codigo)\n"
    "        if _perd:\n"
    "            _m = (_m or '[auto-check]') + (' Warning: your functions use ' + ', '.join(_perd[:4]) + ', defined at top level in this snippet; only def functions are kept between snippets, so next time it will raise NameError. Move it inside the function that uses it.')\n"
    "        if _m and _estado and 'plan_with_state' in _codigo:\n"
    "            _m = _m.replace(' It explains everything seen so far: plan_with_state(observe, advance, goal, actions) searches over states without spending actions.', '')\n"
    "        if _m:\n"
    "            _salida.write('\\n' + _m + '\\n')\n"
    "    except BaseException:\n"
    "        pass\n"
    "    finally:\n"
    "        _listo.set()\n"
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
# Candidata M, "tirar" en vez de "ofrecer" (27-oct: en la práctica el modelo no llamó nunca a
# engine_step): si la malla está disponible y el snippet no la usó, el sandbox la contrasta solo
# y, si reproduce todo lo registrado del nivel (>= 3), agrega una línea con el plan a mano.
_VIVOS_ENGINE_AUTO_DEF = (
    "\ndef _vivos_engine_auto(_salida, _g, _codigo):\n"
    "    try:\n"
    "        _es = _g.get('engine_step')\n"
    "        if not callable(_es) or 'engine_step' in _codigo:\n"
    "            return\n"
    "        import time as _t\n"
    "        _resta = 26.5 - (_t.monotonic() - _VIVOS['_T0'])\n"
    "        if _resta < 1.5:\n"
    "            return\n"
    "        _r = _VIVOS['check_model'](_es, _g.get('transitions'), _g.get('current_frame'), max_report=1, time_limit=min(1.5, _resta - 0.5))\n"
    "        _n = _r.get('checked', 0)\n"
    "        if _n >= 3 and _r.get('interior_ok') == _n and 'note' not in _r:\n"
    "            _salida.write('\\n[auto-check of engine_step] the engine rule reproduces ' + str(_n) + '/' + str(_n) + "
    "' recorded transitions of this level (interior). plan_with_model(engine_step, goal, actions) "
    "searches for the shortest plan without spending actions; it can still be wrong on states not seen yet.\\n')\n"
    "    except BaseException:\n"
    "        pass\n"
) if _VIVOS_MOTOR_SRC else ""
if (_SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_FINAL) == 1 and "_VIVOS = {}" in _SANDBOX_BOOTSTRAP
        and _SANDBOX_BOOTSTRAP.count(_VIVOS_ANCLA_INICIO) == 1):
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_INICIO, _VIVOS_AUTO_DEF + _VIVOS_ENGINE_AUTO_DEF + _VIVOS_ANCLA_INICIO, 1)
    _SANDBOX_BOOTSTRAP = _SANDBOX_BOOTSTRAP.replace(
        _VIVOS_ANCLA_FINAL,
        "        _vivos_auto(stdout, runtime_globals, _code_text, lambda: _send({'type': 'final', "
        "**_collect_retention_payload(), 'stdout': stdout.getvalue(), "
        "'result': _json_safe(runtime_globals.get('result')), "
        "'action_results': _json_safe(action_results)}), "
        "(initial.get('state') or {}).get('_vivos_modo'))\n"
        + ("        _vivos_engine_auto(stdout, runtime_globals, _code_text)\n" if _VIVOS_ENGINE_AUTO_DEF else "")
        + _VIVOS_ANCLA_FINAL, 1)
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
# Freno (revisión de Argon + tu93 del 17-oct: más eficiente por nivel pero sin tiempo para
# terminar): si el step cambió en varios turnos seguidos, dejar de depurarlo y jugar.
RECORDATORIO_FRENO = (
    "[World model] {n} turns in a row without a single game action while modelling. Time is "
    "limited: stop refining `step` now, take a few direct actions to gather evidence, and come "
    "back to it only if it would clearly save actions."
)
RECORDATORIO_CON_ESTADO = (
    "[World model] Your `observe` / `advance` state model is retained. After new observations "
    "re-run `print(check_state(observe, advance))` and fix only the field that fails; once every "
    "field is correct, call `plan_with_state(observe, advance, goal, actions)` and execute the plan."
)
RECORDATORIO_CON_STEP = (
    "[World model] Your `step` simulator is retained. After new observations re-run "
    "`print(check_model(step))`; once `all_interior_ok` is True, call "
    "`plan_with_model(step, goal, actions)` and execute the returned plan instead of exploring by hand."
)

# Candidata F (Astra, 21-oct; diagnóstico de ls20): una vez por nivel, tras el primer intento
# fallido del nivel o 64 acciones sin cambiar de nivel, pedir separar reglas de movimiento de la
# condición de victoria y diseñar una prueba que distinga dos hipótesis. Sin nada propio de un juego.
RECORDATORIO_HIPOTESIS = (
    "[Hypothesis check] Before repeating the attempt, separate movement rules from completion "
    "conditions. Inspect before/after observations of an uncertain interaction, including edge "
    "displays. State two hypotheses and the different observable outcomes they predict. Choose a "
    "short, low-cost test that distinguishes them. After the test, record what was observed and "
    "which hypothesis remains possible. A matching simulation does not establish the game's "
    "completion condition."
)
UMBRAL_HIPOTESIS = 64

# Candidata M (27-oct): el motor de Objetos Vivos aprende solo las reglas del nivel con lo que ya
# jugó el modelo (vivos/mundo.py, sin tokens ni acciones) y, si su regla predijo >= 90 % de los
# cambios recientes antes de verlos, la ofrece como `engine_step`.
RECORDATORIO_MOTOR = (
    "[Engine] Without spending actions, the built-in object engine formed a hypothesis about this "
    "level's mechanics: it predicted {ok}/{n} of the recent board changes it had not seen before, "
    "before seeing them. It is available in every Python call as `engine_step(grid, action)`. It is "
    "only a hypothesis and can be wrong on states or actions not yet seen: run "
    "`print(check_model(engine_step))` first, and plan with `plan_with_model(engine_step, goal, "
    "actions)` only if it explains the recorded transitions."
)

# Modo fuerte: un recordatorio de una línea al final de cada mensaje de turno. Envuelve
# ToolAgent._retained_function_context (harness de Franzen); si falla, devuelve lo original.
COLA_AGENTE = """

# --- Objetos Vivos: recordatorio por turno de check_model / plan_with_model ---
def _vivos_envolver_contexto():
    _orig = ToolAgent._retained_function_context
    _sin, _con, _freno, _estado, _TOPE = __SIN__, __CON__, __FRENO__, __ESTADO__, 0  # 0 = freno apagado
    _hipo, _UMBRAL = __HIPO__, __UMBRAL__
    _motor_txt = __MOTOR__

    def _retained_function_context(self):
        base = _orig(self)
        try:
            kept = getattr(self, "_kept_functions", None) or {}
            con_estado = kept.get("observe") is not None and kept.get("advance") is not None
            fuente = kept.get("advance") if con_estado else kept.get("step")
            texto = _estado if con_estado else (_con if fuente is not None else _sin)
            try:  # turnos seguidos con un step y sin ninguna acción real del juego
                paso = getattr(self, "_vivos_paso", None)
                if fuente is not None and paso is not None and paso == getattr(self, "_vivos_paso_previo", None):
                    self._vivos_sin_jugar = getattr(self, "_vivos_sin_jugar", 0) + 1
                else:
                    self._vivos_sin_jugar = 0
                self._vivos_paso_previo = paso
                if _TOPE and self._vivos_sin_jugar >= _TOPE:
                    texto = _freno.format(n=self._vivos_sin_jugar)
            except Exception:
                pass
            try:  # candidata F: una vez por nivel, tras un intento fallido o _UMBRAL acciones
                nivel, paso = getattr(self, "_vivos_nivel", None), getattr(self, "_vivos_paso", None)
                muertes = len(getattr(self, "_death_ledger_attempts", None) or [])
                if nivel is not None and paso is not None:
                    if nivel != getattr(self, "_vivos_h_nivel", None):
                        self._vivos_h_nivel, self._vivos_h_paso0 = nivel, paso
                        self._vivos_h_muertes0, self._vivos_h_dado = muertes, False
                    murio = muertes > getattr(self, "_vivos_h_muertes0", muertes)
                    largo = paso - getattr(self, "_vivos_h_paso0", paso) >= _UMBRAL
                    if not getattr(self, "_vivos_h_dado", True) and (murio or largo):
                        self._vivos_h_dado = True
                        texto = texto + "\\n\\n" + _hipo
            except Exception:
                pass
            try:  # candidata M: el motor tiene una regla completa de este nivel
                est = getattr(self, "_vivos_motor_stats", None)
                if _motor_txt and est and est[0] is not None and est[0] == getattr(self, "_vivos_nivel", None) \\
                        and getattr(self, "_vivos_motor_datos", None) is not None \\
                        and getattr(self, "_vivos_motor_avisado", None) != est[0]:
                    self._vivos_motor_avisado = est[0]  # una vez por nivel (auditoría de Astra)
                    texto = texto + "\\n\\n" + _motor_txt.format(ok=est[1], n=est[2])
            except Exception:
                pass
            return base + "\\n\\n" + texto
        except Exception:
            return base

    ToolAgent._retained_function_context = _retained_function_context

    _orig_msg = ToolAgent._build_user_message

    def _build_user_message(self, user_prompt, current_frame):
        try:  # Frame.step = acciones del juego hasta ahora: sirve para saber si hubo jugada
            self._vivos_paso = getattr(current_frame, "step", None)
            self._vivos_nivel = getattr(current_frame, "level", None)
        except Exception:
            pass
        return _orig_msg(self, user_prompt, current_frame)

    ToolAgent._build_user_message = _build_user_message


try:
    _vivos_envolver_contexto()
except Exception:  # la corrida oficial sigue sin el recordatorio
    pass
"""


# Modo auto: si el auto-chequeo dice exactamente lo mismo que después del snippet anterior, se
# reemplaza por una línea corta (en lp85 el modelo recibió el mismo aviso 5 veces seguidas).
# El sandbox es un proceso nuevo por snippet: la memoria vive en el agente (host).
COLA_SIN_REPETIR = """

# --- Objetos Vivos: no repetir el mismo auto-chequeo ---
def _vivos_no_repetir(agente, res):
    # Si el auto-chequeo al final del stdout es idéntico al anterior de este agente, lo acorta.
    try:
        salida = res.get("stdout") if isinstance(res, dict) else None
        if agente is None or not isinstance(salida, str):
            return res
        i = salida.rfind("\\n[auto-check of your ")
        if i < 0:
            agente._vivos_ultimo_auto = None
            return res
        mensaje = salida[i:].strip()
        if mensaje == getattr(agente, "_vivos_ultimo_auto", None) and "explains everything" not in mensaje:
            # texto neutral: el informe repetido puede ser positivo (2/2, transferencia entre
            # niveles); decir "still fails" convertía un acierto en un fallo (revisión de Astra)
            res["stdout"] = salida[:i] + "\\n[auto-check] UNCHANGED: same report as after your previous snippet.\\n"
        agente._vivos_ultimo_auto = mensaje
    except Exception:
        pass
    return res


_VIVOS_MOTOR_FUENTES = __MOTOR_FUENTES__


def _vivos_cargar_motor():
    if not _VIVOS_MOTOR_FUENTES:
        return None
    try:
        import sys as _sys
        import types as _types
        paquete = _types.ModuleType("vivos_motor")
        paquete.__path__ = []
        _sys.modules["vivos_motor"] = paquete
        for nombre in ("motor_puro", "percepcion", "mundo", "motor_llm"):
            m = _types.ModuleType("vivos_motor." + nombre)
            m.__package__ = "vivos_motor"
            _sys.modules[m.__name__] = m
            exec(compile(_VIVOS_MOTOR_FUENTES[nombre], "<vivos_motor/" + nombre + ">", "exec"), m.__dict__)
            setattr(paquete, nombre, m)
        # caché de segmentaciones acotada solo en la copia del host (auditoría de Astra: 8192
        # entradas podían ocupar GB con tableros de muchos componentes)
        _sys.modules["vivos_motor.percepcion"]._CACHE_SEG_MAX = 256
        return _sys.modules["vivos_motor.motor_llm"]
    except Exception:
        print("objetos_vivos: motor (candidata M) no cargó; se juega como F", flush=True)
        return None


_VIVOS_MOTOR = _vivos_cargar_motor()
_VIVOS_MOTOR_T = [0.0, 0]


def _vivos_motor_registro(agente, codigo, res):
    # Contadores de uso para la práctica: a stdout la primera vez de cada evento y cada 50
    # snippets a /kaggle/working/vivos_motor_stats.json (sin lanzar nunca).
    try:
        if _VIVOS_MOTOR is None:
            return
        c = _VIVOS_MOTOR.contar
        if "engine_step" in codigo:
            c("snippets_engine_step")
            if "check_model(engine_step" in codigo:
                c("check_model_engine")
            if "plan_with_model(engine_step" in codigo:
                c("plan_engine")
                acciones = len((res or {}).get("action_results") or []) if isinstance(res, dict) else 0
                if acciones:
                    c("acciones_tras_plan_engine", acciones)
        salida = (res or {}).get("stdout") if isinstance(res, dict) else None
        if isinstance(salida, str) and "engine_step" in salida and "Traceback" in salida:
            c("engine_error_en_snippet")
        marca = "\\n[auto-check of engine_step]"
        if isinstance(salida, str) and marca in salida:
            nivel = (getattr(agente, "_vivos_motor_stats", None) or (None,))[0]
            if getattr(agente, "_vivos_engine_auto_nivel", object()) == nivel:
                i = salida.index(marca)
                j = salida.find("\\n", i + len(marca))
                res["stdout"] = salida[:i] + (salida[j:] if j >= 0 else "")
            else:
                agente._vivos_engine_auto_nivel = nivel
                c("engine_auto_mostrado")
        _VIVOS_MOTOR_T[1] += 1
        if _VIVOS_MOTOR_T[1] % 50 == 0:
            import json as _json
            import os as _os
            ruta = "/kaggle/working/vivos_motor_stats.json" if _os.path.isdir("/kaggle/working") else "vivos_motor_stats.json"
            with open(ruta, "w", encoding="utf-8") as f:
                _json.dump(dict(_VIVOS_MOTOR.CONTADORES, segundos_host=round(_VIVOS_MOTOR_T[0], 2),
                                snippets=_VIVOS_MOTOR_T[1]), f)
    except Exception:
        pass


def _vivos_motor_estado(agente, k):
    # Reglas del motor para el nivel actual; caché por (nivel, largo de la historia).
    try:
        st = k.get("initial_state")
        if _VIVOS_MOTOR is None or agente is None or not isinstance(st, dict):
            return
        hist = st.get("history") or []
        nivel = (st.get("current_frame") or {}).get("level")
        clave = (nivel, len(hist))
        if getattr(agente, "_vivos_motor_clave", None) != clave:
            import time as _t
            t0 = _t.monotonic()
            datos, (ok, n) = _VIVOS_MOTOR.motor_para(hist, nivel)
            _VIVOS_MOTOR_T[0] += _t.monotonic() - t0
            agente._vivos_motor_clave = clave
            agente._vivos_motor_datos = datos
            agente._vivos_motor_stats = (nivel, ok, n)
        if getattr(agente, "_vivos_motor_datos", None) is not None:
            k["initial_state"] = dict(k["initial_state"], _vivos_motor=agente._vivos_motor_datos)
    except Exception:
        try:
            _VIVOS_MOTOR.contar("host_failed")
        except Exception:
            pass


def _vivos_sin_repetir():
    import threading as _hilos
    _ctx = _hilos.local()
    _orig_run = ToolAgent._run_python_tool
    _orig_sandbox = run_sandboxed_python

    def _run_python_tool(self, *a, **k):
        _ctx.agente = self
        try:
            return _orig_run(self, *a, **k)
        finally:
            _ctx.agente = None

    def _sandbox(*a, **k):
        agente = getattr(_ctx, "agente", None)
        try:  # último tipo de modelo que definió el modelo: el auto-chequeo sigue ese
            codigo = str(k.get("code", ""))
            if agente is not None:
                if "def step(" in codigo:
                    agente._vivos_modo = "step"
                elif "def observe(" in codigo or "def advance(" in codigo:
                    agente._vivos_modo = "state"
                modo = getattr(agente, "_vivos_modo", None)
                if modo and isinstance(k.get("initial_state"), dict):
                    k["initial_state"] = dict(k["initial_state"], _vivos_modo=modo)
        except Exception:
            pass
        _vivos_motor_estado(agente, k)
        res = _vivos_no_repetir(agente, _orig_sandbox(*a, **k))
        _vivos_motor_registro(agente, str(k.get("code", "")), res)
        return res

    ToolAgent._run_python_tool = _run_python_tool
    globals()["run_sandboxed_python"] = _sandbox


try:
    _vivos_sin_repetir()
except Exception:
    pass
"""


def aplicar(src_root, vivos_src: str, fuerte: bool = False, auto: bool = False, motor=None) -> str:
    """Nunca lanza: si algo no coincide, avisa y deja el harness como estaba.
    auto=True implica fuerte=True. motor: fuentes {motor_puro, percepcion, mundo, motor_llm}
    para la candidata M (requiere auto)."""
    try:
        return _aplicar(src_root, vivos_src, fuerte or auto, auto, motor if auto else None)
    except Exception as exc:  # la re-ejecución oficial tiene que seguir aunque falle el injerto
        return f"objetos_vivos: injerto OMITIDO ({type(exc).__name__}: {exc})"


def _aplicar(src_root, vivos_src: str, fuerte: bool = False, auto: bool = False, motor=None) -> str:
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
        if ("class ToolAgent" not in a or "def _retained_function_context(self)" not in a
                or "def _build_user_message(self, user_prompt: str, current_frame" not in a):
            raise RuntimeError("objetos_vivos: tool_agent.py sin _retained_function_context")
        cola_agente = (COLA_AGENTE.replace("__SIN__", repr(RECORDATORIO_SIN_STEP))
                       .replace("__CON__", repr(RECORDATORIO_CON_STEP))
                       .replace("__FRENO__", repr(RECORDATORIO_FRENO))
                       .replace("__ESTADO__", repr(RECORDATORIO_CON_ESTADO))
                       .replace("__HIPO__", repr(RECORDATORIO_HIPOTESIS))
                       .replace("__UMBRAL__", str(UMBRAL_HIPOTESIS))
                       .replace("__MOTOR__", repr(RECORDATORIO_MOTOR if motor else "")))
        if auto and "run_sandboxed_python(" in a and "def _run_python_tool(self" in a:
            cola_agente += COLA_SIN_REPETIR.replace("__MOTOR_FUENTES__", repr(dict(motor) if motor else {}))
        compile(a + cola_agente, str(agente), "exec")
    if motor:
        for nombre in ("motor_puro", "percepcion", "mundo", "motor_llm"):
            compile(motor[nombre], nombre, "exec")
    cola = (COLA_SANDBOX.replace("__VIVOS_SRC_REPR__", repr(vivos_src))
            .replace("__VIVOS_MOTOR_SRC_REPR__", repr(motor["motor_puro"] if motor else ""))
            + (COLA_AUTO if auto else ""))
    compile(s + cola, str(sandbox), "exec")
    sandbox.write_text(s + cola, encoding="utf-8")
    prompts.write_text(p + COLA_PROMPTS, encoding="utf-8")
    if cola_agente:
        agente.write_text(a + cola_agente, encoding="utf-8")
        return ("objetos_vivos: check_model / plan_with_model injertados + recordatorio por turno"
                + (" + auto-chequeo" if auto else "") + (" + motor (candidata M)" if motor else ""))
    return "objetos_vivos: check_model / plan_with_model injertados"
