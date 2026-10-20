"""Pruebas del verificador/planificador para simuladores escritos por el modelo."""
from types import SimpleNamespace as NS

from vivos.simulador import auto_check, check_model, mouse, parse_action, plan

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
