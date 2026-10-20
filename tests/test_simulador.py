"""Pruebas del verificador/planificador para simuladores escritos por el modelo."""
from types import SimpleNamespace as NS

from vivos.simulador import (auto_check, auto_check_state, check_model, check_state, mouse,
                             parse_action, plan, plan_state, shift_test, state_shift_test,
                             nombres_que_se_pierden)

N = 12  # tablero chico con franja de borde de 4: interior 4x4 (filas/cols 4..7)
MOV = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def tablero(pos, barra=N):
    g = [[0] * N for _ in range(N)]
    g[pos[0]][pos[1]] = 3
    for c in range(barra):  # barra de pasos en la fila 0 (franja de borde)
        g[0][c] = 5
    return g


def fisica(grid, action):
    """Juego real de prueba: el avatar (3) se mueve dentro del interior; la barra se achica."""
    g = [f[:] for f in grid]
    r, c = next((r, c) for r in range(N) for c in range(N) if g[r][c] == 3)
    dr, dc = MOV.get(action, (0, 0))
    nr, nc = r + dr, c + dc
    if 4 <= nr < 8 and 4 <= nc < 8:
        g[r][c], g[nr][nc] = 0, 3
    largo = sum(1 for x in g[0] if x == 5)
    if largo:
        g[0][largo - 1] = 0
    return g


def historia(acciones, pos=(4, 4)):
    frames = [NS(_grid=tablero(pos), level=1)]
    trans = []
    for a in acciones:
        nuevo = NS(_grid=fisica(frames[-1]._grid, a), level=1)
        trans.append(NS(action=a, before_frame=frames[-1], after_frame=nuevo, result={}))
        frames.append(nuevo)
    return trans, frames[-1]


def test_parse_action():
    assert parse_action("MOUSE(row=3, col=7)") == ("MOUSE", 3, 7)
    assert parse_action(mouse(10, 2)) == ("MOUSE", 10, 2)
    assert parse_action("up") == ("UP", None, None)


def test_modelo_correcto_explica_el_interior():
    trans, actual = historia(["RIGHT", "DOWN", "RIGHT", "UP", "LEFT"])

    def hipotesis(grid, action):  # sabe moverse pero ignora la barra
        g = [f[:] for f in grid]
        r, c = next((r, c) for r in range(N) for c in range(N) if g[r][c] == 3)
        dr, dc = MOV.get(action, (0, 0))
        if 4 <= r + dr < 8 and 4 <= c + dc < 8:
            g[r][c], g[r + dr][c + dc] = 0, 3
        return g

    inf = check_model(hipotesis, trans, actual)
    assert inf["checked"] == 5
    assert inf["all_interior_ok"]
    assert inf["exact"] == 0
    assert check_model(fisica, trans, actual)["exact"] == 5


def test_modelo_equivocado_muestra_donde():
    trans, actual = historia(["RIGHT", "DOWN"])
    al_reves = lambda g, a: fisica(g, {"RIGHT": "LEFT", "LEFT": "RIGHT"}.get(a, a))  # noqa: E731
    inf = check_model(al_reves, trans, actual)
    assert not inf["all_interior_ok"]
    assert inf["mismatches"][0]["action"] == "RIGHT"
    assert inf["mismatches"][0]["examples_r_c_actual_predicted"]


def test_step_que_explota_no_rompe():
    trans, actual = historia(["RIGHT"])
    inf = check_model(lambda g, a: 1 / 0, trans, actual)
    assert inf["errors"] == 1
    assert "ZeroDivisionError" in inf["mismatches"][0]["error"]


def test_ignora_muertes_resets_y_otros_niveles():
    trans, actual = historia(["RIGHT", "DOWN"])
    trans[0].result = {"game_over": True}
    trans.append(NS(action="RESET", before_frame=actual, after_frame=actual, result={}))
    viejo = NS(_grid=tablero((4, 4)), level=0)
    trans.append(NS(action="UP", before_frame=viejo, after_frame=viejo, result={}))
    assert check_model(fisica, trans, actual)["checked"] == 1


def test_plan_camino_mas_corto_ignorando_la_barra():
    inicio = NS(_grid=tablero((4, 4)))
    meta = lambda g: g[7][7] == 3  # noqa: E731
    r = plan(fisica, meta, inicio, ["UP", "DOWN", "LEFT", "RIGHT"])
    assert r["length"] == 6 and sorted(r["plan"]) == ["DOWN"] * 3 + ["RIGHT"] * 3
    # meta como grilla objetivo: la barra (borde) no cuenta
    r2 = plan(fisica, tablero((5, 4), barra=0), inicio, ["UP", "DOWN", "LEFT", "RIGHT"])
    assert r2["plan"] == ["DOWN"]


def test_plan_inalcanzable_y_limites():
    inicio = NS(_grid=tablero((4, 4)))
    r = plan(fisica, lambda g: False, inicio, ["UP", "DOWN", "LEFT", "RIGHT"])
    assert r["plan"] is None and r["reason"] == "goal unreachable in the simulator"
    assert plan(fisica, lambda g: False, inicio, [])["reason"] == "no actions given"
    r = plan(fisica, lambda g: False, inicio, ["DOWN", "RIGHT"], max_nodes=3)
    assert r["reason"] == "node limit"


def test_check_model_usa_la_vista_ascii():
    f = lambda texto: NS(ascii=texto, level=1)  # noqa: E731
    t = [NS(action="RIGHT", before_frame=f("ab\ncd"), after_frame=f("ba\ncd"), result={})]
    inf = check_model(lambda g, a: [g[0][::-1], g[1]], t, f("ba\ncd"), border=0)
    assert inf["exact"] == 1


def test_filas_como_texto_se_aceptan():
    trans, actual = historia(["RIGHT"])
    inf = check_model(lambda g, a: ["".join(map(str, fila)) for fila in fisica(g, a)], trans, actual)
    # las filas vuelven como str de dígitos: forma válida aunque los valores no coincidan
    assert "wrong shape" not in str(inf["mismatches"])


def test_presupuesto_agotado_corta_sin_lanzar():
    trans, actual = historia(["RIGHT", "DOWN"])
    assert check_model(fisica, trans, actual, time_limit=-1)["checked"] == 0
    inicio = NS(_grid=tablero((4, 4)))
    assert plan(fisica, lambda g: False, inicio, ["UP"], time_limit=-1)["reason"] == "time limit"


def test_plan_casos_borde_devuelven_motivo():
    inicio = NS(_grid=tablero((4, 4)))
    assert plan(fisica, lambda g: True, inicio, ["UP"])["plan"] == []
    assert plan(fisica, lambda g: False, None, ["UP"])["reason"] == "no start frame"
    r = plan(fisica, lambda g: False, inicio, ["DOWN", "RIGHT"], max_depth=1)
    assert r["reason"] == "depth limit"
    r = plan(lambda g, a: 1 / 0, lambda g: False, inicio, ["UP"])
    assert r["step_errors"] == 1
    no_hasheable = plan(lambda g, a: [[[0]] * N for _ in range(N)], lambda g: False, inicio, ["UP"])
    assert no_hasheable["step_errors"] == 1 and no_hasheable["plan"] is None


def test_forma_incorrecta_y_sin_transiciones():
    trans, actual = historia(["RIGHT"])
    assert "wrong shape" in check_model(lambda g, a: [], trans, actual)["mismatches"][0]["error"]
    assert check_model(fisica, [], actual)["all_interior_ok"] is False


import unittest  # noqa: E402


def fisica_delta(grid, action):
    """La misma física, devuelta solo como cambios del interior (la barra no se modela)."""
    r, c = next((r, c) for r in range(N) for c in range(N) if grid[r][c] == 3)
    dr, dc = MOV.get(action, (0, 0))
    nr, nc = r + dr, c + dc
    if not (4 <= nr < 8 and 4 <= nc < 8):
        return None
    return {(r, c): 0, (nr, nc): 3}


def test_step_parcial_como_dict_de_cambios():
    trans, actual = historia(["RIGHT", "DOWN", "LEFT", "LEFT", "UP"])
    r = check_model(fisica_delta, trans, actual)
    assert r["checked"] == 5
    assert r["all_interior_ok"] is True
    p = plan(fisica_delta, lambda g: g[7][7] == 3, actual, ["UP", "DOWN", "LEFT", "RIGHT"])
    assert p["plan"] is not None
    assert p["length"] == 6


def test_auto_check_resume_en_una_linea():
    trans, actual = historia(["RIGHT", "DOWN", "LEFT"])
    bien = auto_check(fisica_delta, trans, actual)
    assert "3/3" in bien
    assert "plan_with_model" in bien
    mal = auto_check(lambda g, a: None, trans, actual)
    assert "0/3" in mal
    assert "First miss: RIGHT" in mal
    assert auto_check(fisica_delta, [], actual) == ""
    assert auto_check(lambda g, a: {(99, 99): 1}, trans, actual).count("First miss") == 1


def test_step_que_muta_en_el_lugar_y_devuelve_none():
    def en_el_lugar(g, a):
        nuevo = fisica(g, a)
        for r in range(N):
            g[r][:] = nuevo[r]
    trans, actual = historia(["RIGHT", "DOWN"])
    assert check_model(en_el_lugar, trans, actual)["all_interior_ok"] is True
    p = plan(en_el_lugar, lambda g: g[7][7] == 3, actual, ["DOWN", "RIGHT"])
    assert p["plan"] is not None


def test_indices_negativos_son_error():
    trans, actual = historia(["RIGHT"])
    r = check_model(lambda g, a: {(-1, -1): 9}, trans, actual)
    assert r["errors"] == 1
    assert r["all_interior_ok"] is False


def test_auto_check_avisa_si_no_llego_a_revisar_todo():
    import time as _time
    trans, actual = historia(["RIGHT", "DOWN", "LEFT", "UP", "RIGHT"])

    def lento(g, a):
        _time.sleep(0.3)
        return fisica_delta(g, a)
    texto = auto_check(lento, trans, actual, time_limit=0.5)
    assert "could be checked" in texto
    assert "explains everything" not in texto


def _distancia_a_esquina(g):
    r, c = next((r, c) for r in range(N) for c in range(N) if g[r][c] == 3)
    return abs(r - 7) + abs(c - 7)


def test_plan_por_costo_llega_a_cero():
    _, actual = historia(["RIGHT"])
    p = plan(fisica_delta, _distancia_a_esquina, actual, ["UP", "DOWN", "LEFT", "RIGHT"], cost=True)
    assert p["plan"] is not None
    assert p["cost"] == 0
    assert p["length"] == 5


def test_plan_por_costo_devuelve_el_mejor_parcial():
    _, actual = historia(["RIGHT"])
    p = plan(fisica_delta, lambda g: _distancia_a_esquina(g) + 0.5, actual,
             ["UP", "DOWN", "LEFT", "RIGHT"], cost=True)
    assert p["plan"] is None
    assert p["best_cost"] == 0.5
    assert p["start_cost"] == 5.5
    assert p["best_length"] == 5


def test_meta_booleana_sigue_siendo_bfs():
    _, actual = historia(["RIGHT"])
    p = plan(fisica_delta, lambda g: _distancia_a_esquina(g) == 0, actual,
             ["UP", "DOWN", "LEFT", "RIGHT"])
    assert p["length"] == 5
    assert "cost" not in p


M = 20  # tablero con paredes dibujadas (color 1) y borde de 4: interior 4..15


def _tablero_paredes(pos):
    g = [[0] * M for _ in range(M)]
    for i in range(M):
        g[4][i] = g[15][i] = g[i][4] = g[i][15] = 1  # paredes del recinto
    g[pos[0]][pos[1]] = 3
    return g


def _general(g, a):
    """Física leída del tablero: el avatar (3) avanza si la celda destino es piso (0)."""
    r, c = next((r, c) for r in range(M) for c in range(M) if g[r][c] == 3)
    dr, dc = MOV.get(a, (0, 0))
    if g[r + dr][c + dc] != 0:
        return None
    return {(r, c): 0, (r + dr, c + dc): 3}


def _fijo(g, a):
    """Lo mismo, pero con las posiciones del nivel escritas a mano."""
    tabla = {((8, 8), "RIGHT"): (8, 9), ((8, 9), "DOWN"): (9, 9), ((9, 9), "LEFT"): (9, 8)}
    for (pos, accion), destino in tabla.items():
        if g[pos[0]][pos[1]] == 3 and accion == a:
            return {pos: 0, destino: 3}
    return None


def _historia_paredes(acciones, nivel=1):
    frames = [NS(_grid=_tablero_paredes((8, 8)), level=nivel)]
    trans = []
    for a in acciones:
        g = [f[:] for f in frames[-1]._grid]
        for (r, c), v in (_general(g, a) or {}).items():
            g[r][c] = v
        nuevo = NS(_grid=g, level=nivel)
        trans.append(NS(action=a, before_frame=frames[-1], after_frame=nuevo, result={}))
        frames.append(nuevo)
    return trans, frames[-1]


def test_shift_test_distingue_general_de_coordenadas_fijas():
    trans, actual = _historia_paredes(["RIGHT", "DOWN", "LEFT"])
    assert check_model(_fijo, trans, actual)["all_interior_ok"] is True
    assert shift_test(_general, trans, actual) is True
    assert shift_test(_fijo, trans, actual) is False
    assert shift_test(_general, [], actual) is None


def test_auto_check_al_empezar_nivel_valida_el_step_anterior():
    trans, _ = _historia_paredes(["RIGHT", "DOWN", "LEFT"])
    nuevo = NS(_grid=_tablero_paredes((10, 10)), level=2)  # nivel 2 sin transiciones todavía
    bien = auto_check(_general, trans, nuevo)
    assert "position-independent" in bien
    mal = auto_check(_fijo, trans, nuevo)
    assert "shifted by one cell" in mal
    assert "plan_with_model" not in mal


def test_auto_check_avisa_si_step_lee_el_tablero_global():
    trans, actual = _historia_paredes(["RIGHT"])

    def espia(grid, action):
        return current_frame  # noqa: F821 - es justamente el error que se detecta

    assert "live global current_frame" in auto_check(espia, trans, actual)


def test_minimo_local_se_avisa():
    _, actual = historia(["RIGHT"])
    p = plan(lambda g, a: None, _distancia_a_esquina, actual, ["UP"], cost=True)
    assert p["plan"] is None
    assert "local minimum" in p["reason"]


def test_meta_entera_sin_cost_se_lee_como_verdad():
    # H1: un goal que devuelve 1/0 (p. ej. int(llegó)) no es una distancia.
    _, actual = historia(["RIGHT"])
    p = plan(fisica_delta, lambda g: int(_distancia_a_esquina(g) == 0), actual,
             ["UP", "DOWN", "LEFT", "RIGHT"])
    assert p["length"] == 5
    assert "cost" not in p


def test_distancia_sin_cost_pide_cost_true():
    _, actual = historia(["RIGHT"])
    for meta in (_distancia_a_esquina, lambda g: float("nan")):
        p = plan(fisica_delta, meta, actual, ["UP"])
        assert p["plan"] is None and "cost=True" in p["reason"]


def test_costo_nan_o_no_numerico():
    # H5: NaN no es un costo; al inicio se rechaza, en la búsqueda se cuenta como error.
    _, actual = historia(["RIGHT"])
    p = plan(fisica_delta, lambda g: float("nan"), actual, ["UP"], cost=True)
    assert p["plan"] is None and "finite number" in p["reason"]
    p = plan(fisica_delta, lambda g: "lejos", actual, ["UP"], cost=True)
    assert p["plan"] is None and "finite number" in p["reason"]
    primero = {"v": True}

    def costo(g):
        if primero["v"]:
            primero["v"] = False
            return 3
        return float("nan")
    p = plan(fisica_delta, costo, actual, ["UP", "DOWN"], cost=True)
    assert p["plan"] is None and p["step_errors"] >= 1


def test_shift_test_ignora_el_hud_exterior():
    # H2: una barra de pasos en la franja exterior no debe descartar el caso.
    trans, actual = _historia_paredes(["RIGHT", "DOWN", "LEFT"])
    for t in trans:
        g = [f[:] for f in t.after_frame._grid]
        g[0][0] = 9
        t.after_frame = NS(_grid=g, level=t.after_frame.level)

    def general_hud(g, a):
        d = _general(g, a)
        if d is not None:
            d[(0, 0)] = 9
        return d
    assert shift_test(general_hud, trans, actual) is True
    assert shift_test(_fijo, trans, actual) is False


def test_shift_test_prediccion_rara_es_false():
    # H4: una predicción con filas no comparables no lanza.
    trans, actual = _historia_paredes(["RIGHT"])
    assert shift_test(lambda g, a: [None] * M, trans, actual) is False


def test_aviso_de_global_se_suma_al_reporte():
    # H3: el aviso no tapa el informe normal.
    trans, actual = _historia_paredes(["RIGHT", "DOWN", "LEFT"])

    def espia(grid, action, _f=_general):
        if False:
            return current_frame  # noqa: F821
        return _f(grid, action)
    texto = auto_check(espia, trans, actual)
    assert "reproduces 3/3" in texto
    assert "live global current_frame" in texto


# --- v4: informe por familia, cambios faltantes/inventados y modelo de estado ---

def _historia_mixta():
    """Flechas que mueven el avatar y un SPACE que pinta (12,12): mecánicas distintas."""
    trans, _ = _historia_paredes(["RIGHT", "DOWN"])
    antes = trans[-1].after_frame
    g = [f[:] for f in antes._grid]
    g[12][12] = 7
    despues = NS(_grid=g, level=1)
    trans.append(NS(action="SPACE", before_frame=antes, after_frame=despues, result={}))
    return trans, despues


def test_informe_por_familia_y_cambio_no_predicho():
    trans, actual = _historia_mixta()
    r = check_model(_general, trans, actual)
    assert r["by_family"] == {"arrows": [2, 2], "SPACE": [0, 1]}
    m = r["mismatches"][0]
    assert m["missed_changes"] == 1 and m["missed_example"] == (12, 12, 0, 7)
    assert m["invented_changes"] == 0
    texto = auto_check(_general, trans, actual)
    assert "By action: SPACE 0/1, arrows 2/2" in texto
    assert "real changes you did not predict, e.g. (12,12) went 0->7" in texto


def test_cambio_inventado():
    trans, actual = _historia_paredes(["RIGHT"])

    def inventa(g, a):
        d = _general(g, a) or {}
        d[(12, 12)] = 5
        return d
    m = check_model(inventa, trans, actual)["mismatches"][0]
    assert m["invented_changes"] == 1 and m["invented_example"] == (12, 12, 0, 5)


def _observe(g):
    r, c = next((r, c) for r in range(M) for c in range(M) if g[r][c] == 3)
    return {"player": (r, c), "walls": sum(f.count(1) for f in g)}


def _advance(s, a):
    dr, dc = MOV.get(a, (0, 0))
    r, c = s["player"]
    return {"player": (r + dr, c + dc), "walls": s["walls"]}


def test_check_state_por_campo():
    trans, actual = _historia_paredes(["RIGHT", "DOWN", "LEFT"])
    r = check_state(_observe, _advance, trans, actual)
    assert r["all_ok"] and r["state_ok"] == 3 and r["blind"] == 0
    assert r["fields"] == {"player": [3, 3], "walls": [3, 3]}

    def mal(s, a):
        n = _advance(s, a)
        n["walls"] += 1
        return n
    r = check_state(_observe, mal, trans, actual)
    assert r["fields"]["walls"] == [0, 3] and r["fields"]["player"] == [3, 3]
    assert r["field_misses"]["walls"]["index"] == 0


def test_check_state_punto_ciego_y_errores():
    trans, actual = _historia_paredes(["RIGHT", "DOWN"])
    ciego = check_state(lambda g: {"x": 1}, lambda s, a: s, trans, actual)
    assert ciego["constant"] and not ciego["all_ok"] and ciego["blind"] == 2
    assert ciego["blind_example"]["cells_changed"] == 2
    texto = auto_check_state(lambda g: {"x": 1}, lambda s, a: s, trans, actual)
    assert "captures nothing yet" in texto and "plan_with_state" not in texto and "2/2" not in texto
    # parcialmente ciego: ve al jugador pero no un cambio extra
    trans2, actual2 = _historia_mixta()
    parcial = check_state(_observe, lambda s, a: _advance(s, a) if a != "SPACE" else s, trans2, actual2)
    assert not parcial["constant"] and parcial["blind"] == 1

    def lanza(g):
        raise ValueError("boom")
    r = check_state(lanza, _advance, trans, actual)
    assert r["errors"] == 2 and "boom" in r["first_error"]["error"]
    no_hash = check_state(lambda g: {"x": [{1, 2}, {"a": [1]}]}, lambda s, a: s, trans, actual)
    assert no_hash["errors"] == 0  # listas, sets y dicts se congelan


def test_auto_check_state_textos():
    trans, actual = _historia_paredes(["RIGHT", "DOWN", "LEFT"])
    ok = auto_check_state(_observe, _advance, trans, actual)
    assert "3/3" in ok and "plan_with_state" in ok
    nuevo = NS(_grid=_tablero_paredes((10, 10)), level=2)
    assert "new level" in auto_check_state(_observe, _advance, trans, nuevo)
    assert auto_check_state(_observe, _advance, [], actual) == ""
    assert auto_check_state(None, None, trans, actual).startswith("[auto-check")  # errores, no lanza


def test_plan_state():
    _, actual = _historia_paredes(["RIGHT"])

    def avanza(s, a):  # sin paredes en el modelo: el interior es 5..14
        n = _advance(s, a)
        r, c = n["player"]
        return n if 5 <= r <= 14 and 5 <= c <= 14 else s
    p = plan_state(_observe, avanza, lambda s: s["player"] == (12, 12), actual,
                   ["UP", "DOWN", "LEFT", "RIGHT"])
    assert p["length"] == 7
    assert plan_state(_observe, avanza, lambda s: float("nan"), actual, ["UP"], max_nodes=30)["plan"] is None
    assert "observe failed" in plan_state(lambda g: 1 / 0, avanza, lambda s: True, actual, ["UP"])["reason"]
    assert plan_state(_observe, avanza, lambda s: True, actual, [])["reason"] == "no actions given"
    tupla = plan_state(_observe, avanza, lambda s: s["player"][1] > 12 and s["player"], actual, ["RIGHT"])
    assert tupla["length"] == 4  # una tupla vale por su verdad de Python
    assert plan_state(_observe, avanza, lambda s: s.get("won"), actual, ["RIGHT"], max_nodes=20)["plan"] is None
    dist = plan_state(_observe, avanza, lambda s: abs(s["player"][1] - 12), actual, ["RIGHT"])
    assert dist["plan"] is None and "cost=True" in dist["reason"]
    costo = plan_state(_observe, avanza, lambda s: abs(s["player"][0] - 12) + abs(s["player"][1] - 12),
                       actual, ["UP", "DOWN", "LEFT", "RIGHT"], cost=True)
    assert costo["length"] == 7 and costo["cost"] == 0
    lejos = plan_state(_observe, avanza, lambda s: abs(s["player"][1] - 40) + 0.5, actual, ["RIGHT"], cost=True)
    assert lejos["plan"] is None and lejos["best_cost"] == 26.5 and lejos["best_length"] == 5
    assert "goal failed" in plan_state(_observe, avanza, lambda s: s["nada"], actual, ["UP"])["reason"]


def test_campos_con_claves_no_texto():
    trans, actual = _historia_paredes(["RIGHT"])
    r = check_state(lambda g: {1: "a", "1": "b"}, lambda s, a: s, trans, actual)
    assert set(r["fields"]) == {"1", "1:int"}


def test_celdas_ambientales_no_son_punto_ciego():
    # un reloj en (12,5) que cambia en cada transición, con acciones distintas
    trans, actual = _historia_paredes(["RIGHT", "DOWN", "LEFT", "UP"])
    for k, t in enumerate(trans):
        for fr, v in ((t.before_frame, k % 2), (t.after_frame, (k + 1) % 2)):
            fr._grid = [f[:] for f in fr._grid]
            fr._grid[12][5] = 8 + v
    r = check_state(_observe, _advance, trans, actual)
    assert r["ambient_cells"] == 1 and r["blind"] == 0 and r["all_ok"]


def test_state_shift_test_detecta_coordenadas_fijas():
    g = _tablero_paredes((8, 8))
    general = lambda g: {"player": next((r, c) for r in range(M) for c in range(M) if g[r][c] == 3)}  # noqa: E731
    fijo = lambda g: {"player": (8, 8) if g[8][8] == 3 else None, "n": 1}  # noqa: E731
    clavado = lambda g: {"player": (8, 8)}  # noqa: E731
    assert state_shift_test(general, g) == []
    assert state_shift_test(fijo, g) == ["player"]
    assert state_shift_test(clavado, g) == ["player"]
    assert state_shift_test(lambda g: 1 / 0, g) == []
    paredes = lambda g: {"walls": frozenset((r, c) for r in range(M) for c in range(M) if g[r][c] == 1)}  # noqa: E731
    assert state_shift_test(paredes, g) == []
    fijas = lambda g: {"walls": frozenset((4, c) for c in range(4, 16))}  # noqa: E731
    assert state_shift_test(fijas, g) == ["walls"]


def test_nivel_nuevo_estado():
    trans, _ = _historia_paredes(["RIGHT", "DOWN", "LEFT"])
    nuevo = NS(_grid=_tablero_paredes((10, 10)), level=2)
    assert "observe() reads this board" in auto_check_state(_observe, _advance, trans, nuevo)
    sin_jugador = NS(_grid=[[0] * M for _ in range(M)], level=2)
    assert "crashed" in auto_check_state(_observe, _advance, trans, sin_jugador)

    def obs_vacia(g):
        pos = [(r, c) for r in range(M) for c in range(M) if g[r][c] == 3]
        return {"player": pos[0] if pos else None, "walls": sum(f.count(1) for f in g)}
    assert "returns nothing for player" in auto_check_state(obs_vacia, _advance, trans, sin_jugador)

    def obs_fija(g):  # lee el jugador solo en la fila 8, columnas 8..10 del nivel 1
        for c in (8, 9, 10):
            if g[8][c] == 3:
                return {"player": (8, c), "walls": sum(f.count(1) for f in g)}
        for c in (8, 9, 10):
            if g[9][c] == 3:
                return {"player": (9, c), "walls": sum(f.count(1) for f in g)}
        return {"player": (8, 8), "walls": sum(f.count(1) for f in g)}
    nuevo_fijo = NS(_grid=_tablero_paredes((8, 10)), level=2)
    assert "fixed coordinates" in auto_check_state(obs_fija, _advance, trans, nuevo_fijo)


def test_choques_sin_cambio_no_son_estado_vacio():
    # Argon: un inicio con choques contra la pared (el tablero no cambia) no es "no captura nada"
    trans, actual = _historia_paredes(["RIGHT"])
    quieto = trans[-1].after_frame
    for _ in range(2):
        trans.append(NS(action="UP", before_frame=quieto, after_frame=quieto, result={}))
    solo_choques = trans[1:]
    r = check_state(_observe, lambda s, a: s, solo_choques, actual)
    assert r["changed"] == 0 and not r["constant"] and r["all_ok"]


def test_nombres_que_se_pierden():
    codigo = "DIRS = {'UP': (-1, 0)}\nN = 3\nX: int = 2\ndef advance(s, a):\n    return s"
    ns = {}
    exec(compile("DIRS = {'UP': (-1, 0)}\nN = 3\n"
                 "def advance(s, a):\n    d = DIRS[a]\n    return [x for x in range(N)]\n"
                 "def observe(g):\n    return {}\n", "<python_tool>", "exec"), ns)
    assert nombres_que_se_pierden([ns["advance"], ns["observe"]], codigo) == ["DIRS", "N"]
    assert nombres_que_se_pierden([ns["observe"]], codigo) == []
    assert nombres_que_se_pierden([ns["advance"]], "def advance(s, a):\n    return s") == []
    assert nombres_que_se_pierden([None], "x = (") == []
    ns2 = {}
    exec(compile("x = 1\ndef obs(s):\n    return s.x\n", "<python_tool>", "exec"), ns2)
    assert nombres_que_se_pierden([ns2["obs"]], "x = 1\ndef obs(s):\n    return s.x") == []  # atributo


class PruebasSimulador(unittest.TestCase):
    """El repo corre con unittest: cada test_ de arriba es un caso."""

    def setUp(self):
        # En el sandbox cada snippet es un proceso nuevo con 24 s de presupuesto desde la carga
        # del módulo; en la suite completa el módulo vive más que eso. Reiniciar el reloj.
        import time as _time
        from vivos import simulador
        simulador._T0 = _time.monotonic()


for _nombre, _f in list(globals().items()):
    if _nombre.startswith("test_") and callable(_f):
        setattr(PruebasSimulador, _nombre, lambda self, f=_f: f())
del _nombre, _f  # si no, `_f` queda apuntando a la clase y unittest la corre dos veces

if __name__ == "__main__":
    unittest.main()


# --- Revisión de Astra (20-oct): tres defectos reproducidos y corregidos -----------------------

_GRAFO = {("S", "a"): "A", ("A", "b"): "B", ("B", "c"): "X",
          ("S", "d"): "C", ("C", "e"): "X", ("X", "f"): "G"}
_COSTO = {"S": 10, "A": 1, "B": 1, "X": 1, "C": 5, "G": 0}
_CODIGO = {"S": 1, "A": 2, "B": 3, "X": 4, "C": 5, "G": 6}
_NOMBRE = {v: k for k, v in _CODIGO.items()}


class PruebasRevisionAstra(unittest.TestCase):
    def setUp(self):  # el presupuesto de tiempo del simulador es por proceso (ver PruebasSimulador)
        import time as _time
        from vivos import simulador
        simulador._T0 = _time.monotonic()

    def test_estado_vacio_con_jugador_que_se_mueve_no_explica_todo(self):
        # El jugador cambia las mismas celdas en cada transición con acciones distintas: el
        # filtro de celdas ambientales las descartaba y observe() -> {} salía "explica todo".
        trans, actual = historia(["RIGHT", "LEFT", "RIGHT"])
        r = check_state(lambda g: {}, lambda s, a: s, trans, actual)
        self.assertTrue(r["constant"])
        self.assertFalse(r["all_ok"])
        self.assertIsNotNone(r["blind_example"])
        texto = auto_check_state(lambda g: {}, lambda s, a: s, trans, actual)
        self.assertIn("captures nothing", texto)
        self.assertNotIn("explains everything", texto)

    def test_estado_correcto_sigue_explicando_todo(self):
        trans, actual = historia(["RIGHT", "LEFT", "RIGHT"])

        def observe(g):
            return {"p": next((r, c) for r in range(4, 8) for c in range(4, 8) if g[r][c] == 3)}

        def advance(s, a):
            dr, dc = MOV.get(a, (0, 0))
            r, c = s["p"][0] + dr, s["p"][1] + dc
            return {"p": (r, c) if 4 <= r < 8 and 4 <= c < 8 else s["p"]}

        r = check_state(observe, advance, trans, actual)
        self.assertTrue(r["all_ok"])
        self.assertFalse(r["constant"])

    def test_plan_estado_por_costo_reabre_estado_alcanzado_por_camino_mas_corto(self):
        # S->A->B->X (barato) marca X a profundidad 3; S->C->X llega a profundidad 2 y X->G.
        r = plan_state(lambda g: "S", lambda s, a: _GRAFO.get((s, a), s), lambda s: _COSTO[s],
                       [[0]], list("abcdef"), max_depth=3, cost=True)
        self.assertEqual(r["plan"], ["d", "e", "f"])

    def test_plan_por_costo_reabre_estado_alcanzado_por_camino_mas_corto(self):
        def grilla(nombre):
            g = [[0] * N for _ in range(N)]
            g[5][5] = _CODIGO[nombre]
            return g

        def step(g, a):
            return grilla(_GRAFO.get((_NOMBRE[g[5][5]], a), _NOMBRE[g[5][5]]))

        r = plan(step, lambda g: _COSTO[_NOMBRE[g[5][5]]], grilla("S"), list("abcdef"),
                 max_depth=3, cost=True)
        self.assertEqual(r["plan"], ["d", "e", "f"])
