"""IA de Objetos Vivos (docs/IA_OBJETOS_VIVOS.md): percepción, colonia, agente."""

from __future__ import annotations

import json
import unittest

import numpy as np

from vivos.percepcion import DetectorDeRuido, efecto, fondo, segmentar
from vivos.puntaje import de_juego, por_nivel
from vivos.reglas import (
    ATRAVIESA,
    CAMBIA,
    DEGRADED,
    FATAL,
    HEALTHY,
    INERTE,
    MUERTE,
    QUARANTINED,
    Colonia,
    Observacion,
    Ranura,
    Transicion,
)

try:
    import arcengine  # noqa: F401

    HAY_ARCENGINE = True
except ImportError:  # pragma: no cover - depende del entorno
    HAY_ARCENGINE = False


def lienzo(bg: int = 0) -> np.ndarray:
    return np.full((64, 64), bg, dtype=np.int16)


def bloque(g: np.ndarray, x: int, y: int, color: int, lado: int = 4) -> np.ndarray:
    g[y : y + lado, x : x + lado] = color
    return g


class PercepcionTest(unittest.TestCase):
    def test_fondo_ignora_el_letterbox(self):
        g = lienzo(3)
        g[20:44, 2:62] = 0  # área de juego más chica que el borde
        bloque(g, 10, 24, 5, 8)
        self.assertEqual(fondo(g), 0)
        marcos = [o for o in segmentar(g) if o.es_marco]
        self.assertEqual({o.color for o in marcos}, {3})

    def test_efecto_mueve_y_recolorea(self):
        a = bloque(bloque(lienzo(), 8, 8, 8), 40, 40, 2)
        d = bloque(bloque(lienzo(), 12, 8, 8), 40, 40, 4)
        e = efecto(a, d)
        self.assertEqual(e.mueve, [(8, 4, 0)])
        self.assertEqual(e.tam_mueve, [(4, 4)])
        self.assertEqual(e.recolorea, [(2, 4)])
        self.assertFalse(e.otro)

    def test_sprite_sobre_nodos_no_hace_mover_al_nodo_que_destapa(self):
        # Regresión de tu93: un bloque 9 (con un indicador 4 que gira) avanza de un nodo 0
        # a otro. El nodo destapado no se movió; el que se movió es el bloque.
        g = lienzo(5)
        for x in range(0, 60, 6):
            bloque(g, x, 20, 0, lado=3)  # fila de nodos 0 (abundantes)
        antes = g.copy()
        bloque(antes, 12, 20, 9, lado=3)
        antes[21, 14] = 4  # indicador mirando a la derecha
        despues = g.copy()
        bloque(despues, 18, 20, 9, lado=3)
        despues[22, 19] = 4  # giró: ahora mira abajo
        e = efecto(antes, despues)
        self.assertEqual(e.mueve[0], (9, 6, 0))
        self.assertNotIn(0, [c for c, _, _ in e.mueve])

    def test_avatar_que_se_mueve_sobre_un_piso_raro_no_es_recoloreo(self):
        # Piso (0) abundante pero que no es el fondo estimado: el avatar (8) se
        # mueve a una celda de piso del mismo tamaño. Se mueve el raro.
        a = lienzo(5)
        a[0:40, 0:40] = 0
        d = a.copy()
        bloque(a, 8, 8, 8)
        bloque(d, 12, 8, 8)
        e = efecto(a, d)
        self.assertEqual(e.mueve, [(8, 4, 0)])
        self.assertEqual(e.recolorea, [])

    def test_reloj_que_avanza_se_enmascara_y_el_avatar_no(self):
        det = DetectorDeRuido()
        acciones = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION2", "ACTION4"]
        avatar = [(20, 20), (20, 24), (20, 24), (24, 24), (24, 28), (28, 28)]
        grid = bloque(bloque(lienzo(), 0, 0, 11), *avatar[0], 8)
        for i, acc in enumerate(acciones[1:], start=1):
            nuevo = bloque(bloque(lienzo(), 4 * i, 0, 11), *avatar[i], 8)
            det.observar(grid, nuevo, acc)
            grid = nuevo
        self.assertEqual(det.relojes, [11])
        self.assertTrue(det.mascara[0:4, 0:24].all())  # donde anduvo el reloj
        self.assertFalse(det.mascara[0:4, 40:].any())  # el resto de la fila sigue visible
        self.assertFalse(det.mascara[20:40, 20:40].any())

    def test_reloj_con_color_del_tablero_no_enmascara_el_tablero(self):
        # Regresión de vc33: una barra de pasos (7 que se vuelve 4) en la fila 0, y
        # piezas de esos mismos colores en el tablero que a veces se mueven.
        det = DetectorDeRuido()
        g = lienzo(3)
        g[0, :] = 7
        bloque(g, 20, 30, 4, lado=6)
        bloque(g, 40, 30, 7, lado=6)
        for paso in range(1, 13):
            n = g.copy()
            n[0, 64 - paso] = 4  # la barra se gasta una celda por acción
            if paso % 3 == 0:  # el botón mueve las piezas del tablero
                n[20:50, 10:60] = 3
                dy = 4 if (paso // 3) % 2 else 0
                bloque(n, 20, 30 + dy, 4, lado=6)
                bloque(n, 40, 30 - dy, 7, lado=6)
            det.observar(g, n, f"ACTION6@{paso % 4},{paso % 5}")
            g = n
        self.assertIn(4, det.relojes)
        self.assertTrue(det.mascara[0, 60:].all())  # la parte gastada de la barra
        self.assertFalse(det.mascara[10:, :].any())  # el tablero sigue visible

    def test_reloj_heredado_no_tapa_su_color_en_todo_el_nivel_nuevo(self):
        # Regresión de ls20: el color de la barra es el piso del nivel 2; heredarlo
        # por color tapaba media pantalla y el agente quedaba ciego.
        det = DetectorDeRuido()
        g = lienzo(0)
        g[20:60, 0:64] = 3  # el piso del nivel nuevo es del color del reloj
        g[0:2, 0:30] = 3  # y la barra sigue arriba
        viejo = np.zeros((64, 64), dtype=bool)
        viejo[0:2, 10:30] = True
        det.heredar([3], g, {3: viejo})
        self.assertTrue(det.mascara[0:2, 10:30].all())
        self.assertFalse(det.mascara[20:60, :].any())

    def test_barra_pegada_a_un_piso_del_mismo_color_no_lo_tapa(self):
        # Regresión de tn36 (nivel 2): la barra toca un piso de su color; extenderla
        # por conexión tapaba media pantalla. Un reloj de HUD es delgado.
        det = DetectorDeRuido()
        g = lienzo(0)
        g[2:40, 0:30] = 3  # piso del mismo color que la barra, pegado a ella
        for paso in range(0, 10):
            n = g.copy()
            n[0:2, paso] = 3  # la barra se llena una columna por acción
            det.observar(g, n, f"ACTION6@{paso % 3},{paso % 7}")
            g = n
        self.assertIn(3, det.relojes)
        self.assertTrue(det.mascara[0:2, 1:9].all())
        self.assertFalse(det.mascara[10:40, 0:30].any())

    def test_barra_que_se_gasta_es_reloj_y_el_piso_que_descubre_no(self):
        det = DetectorDeRuido()
        g = lienzo(5)
        g[0:40, :] = 0
        g[60:64, 0:40] = 11
        bloque(g, 8, 8, 8)
        x, y = 8, 8
        pasos = {"ACTION4": (4, 0), "ACTION2": (0, 4), "ACTION1": (0, -4)}
        for i, acc in enumerate(["ACTION4", "ACTION2", "ACTION4", "ACTION1", "ACTION4"]):
            n = g.copy()
            n[60:64, 36 - 4 * i : 40 - 4 * i] = 0  # la barra pierde un tramo
            n[y : y + 4, x : x + 4] = 0
            x, y = x + pasos[acc][0], y + pasos[acc][1]
            n[y : y + 4, x : x + 4] = 8
            det.observar(g, n, acc)
            g = n
        self.assertEqual(det.relojes, [11])
        self.assertFalse(det.mascara[8:16].any())


def transicion(n: int, accion: str, antes, despues, **kw) -> Transicion:
    mascara = np.zeros((64, 64), dtype=bool)
    return Transicion(
        n=n,
        accion=accion,
        click=kw.get("click"),
        antes=antes,
        despues=despues,
        mascara=mascara,
        efecto=efecto(antes, despues, mascara),
        game_over=kw.get("game_over", False),
        sube_nivel=kw.get("sube", False),
        clave_antes=kw.get("estado", f"s{n}"),
        clave_despues="",
    )


class ColoniaTest(unittest.TestCase):
    def test_inerte_contradicha_se_repara_a_mueve_con_lkg(self):
        col = Colonia()
        quieto = bloque(lienzo(), 0, 0, 8)
        col.registrar(transicion(1, "ACTION4", quieto, quieto.copy()))  # chocó contra el borde
        r = col.ranuras["accion:ACTION4"]
        self.assertEqual(r.regla, INERTE)
        a = bloque(lienzo(), 8, 8, 8)
        d = bloque(lienzo(), 12, 8, 8)
        col.registrar(transicion(2, "ACTION4", a, d))
        self.assertEqual(r.regla, "mueve:8:1:0")
        self.assertEqual(r.estado, HEALTHY)
        self.assertEqual(r.lkg, INERTE)
        self.assertTrue(any("degraded" in h for h in r.historial))
        self.assertEqual(col.avatar(), 8)

    def test_anomalia_rara_no_tumba_una_regla_bien_confirmada(self):
        # Regresión de ls20: el avatar reaparece en el inicio al perder una vida y eso
        # parecía un salto; una sola contradicción tumbaba "mueve" a "cambia".
        r = Ranura("accion:ACTION1")
        for i in range(20):
            r.agregar(i, Observacion("mueve:9:0:-1"), f"s{i}")
        r.agregar(20, Observacion("mueve:9:3:5"), "s20")
        self.assertEqual(r.regla, "mueve:9:0:-1")
        self.assertEqual(r.estado, HEALTHY)
        self.assertEqual(r.anomalias, 1)

    def test_contradicciones_frecuentes_si_tumban_la_regla(self):
        r = Ranura("accion:ACTION1")
        for i in range(10):
            r.agregar(i, Observacion("mueve:9:0:-1"), f"s{i}")
        for i in range(10, 13):
            r.agregar(i, Observacion("mueve:9:3:5"), f"s{i}")
        self.assertNotEqual(r.regla, "mueve:9:0:-1")

    def test_cambia_se_afina_cuando_la_evidencia_sostiene_un_mueve(self):
        r = Ranura("accion:ACTION3")
        r.agregar(0, Observacion("mueve:9:-1:0"), "s0")
        r.agregar(1, Observacion("mueve:9:2:5"), "s1")  # anomalía temprana: cae a "cambia"
        self.assertEqual(r.regla, CAMBIA)
        for i in range(2, 40):
            r.agregar(i, Observacion("mueve:9:-1:0"), f"s{i}")
        self.assertEqual(r.regla, "mueve:9:-1:0")

    def test_tres_reparaciones_fallidas_ponen_en_cuarentena(self):
        r = Ranura("contacto:4")
        r.agregar(1, Observacion(ATRAVIESA), "a")
        r.agregar(2, Observacion("bloquea"), "b")  # contradice y ninguna explica ambas
        self.assertEqual(r.estado, DEGRADED)
        r.agregar(3, Observacion("bloquea"), "c")
        r.agregar(4, Observacion(ATRAVIESA), "d")
        self.assertEqual(r.estado, QUARANTINED)
        self.assertFalse(r.usable)

    def test_confirmaciones_cuentan_estados_distintos(self):
        r = Ranura("accion:ACTION1")
        for n in range(5):
            r.agregar(n, Observacion(INERTE), "esquina")
        self.assertEqual(r.confirmaciones, 1)
        self.assertFalse(r.usable)
        r.agregar(6, Observacion(INERTE), "otro")
        r.agregar(7, Observacion(INERTE), "otro mas")
        self.assertTrue(r.usable)

    def test_una_muerte_no_refuta_lo_que_se_sabia_del_contacto(self):
        r = Ranura("contacto:0")
        r.agregar(1, Observacion(ATRAVIESA), "a")
        r.agregar(2, Observacion(ATRAVIESA), "b")
        r.agregar(3, Observacion(MUERTE), "c")  # la energía se acabó justo ahí
        self.assertEqual((r.regla, r.estado), (ATRAVIESA, HEALTHY))
        nueva = Ranura("contacto:2")
        nueva.agregar(1, Observacion(MUERTE), "a")
        self.assertEqual(nueva.regla, FATAL)
        self.assertTrue(nueva.usable)

    def test_conjetura_por_simetria_y_refutacion(self):
        col = Colonia()
        for i, (acc, dx, dy) in enumerate([("ACTION1", 0, -4), ("ACTION2", 0, 4), ("ACTION4", 4, 0)]):
            a = bloque(lienzo(), 20, 20, 8)
            d = bloque(lienzo(), 20 + dx, 20 + dy, 8)
            col.registrar(transicion(i, acc, a, d))
        self.assertTrue(col.es_conjetura("ACTION3"))
        self.assertEqual(col.desplazamientos(bloque(lienzo(), 20, 20, 8), np.zeros((64, 64), bool))["ACTION3"], (8, -4, 0))
        col.refutadas.add("ACTION3")
        self.assertNotIn("ACTION3", col.movimientos())

    def test_exportar_es_json(self):
        col = Colonia()
        g = bloque(lienzo(), 8, 8, 8)
        col.registrar(transicion(1, "ACTION1", g, g.copy()))
        json.dumps(col.exportar())


class PuntajeTest(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(por_nivel(10, 20), 0.25)
        self.assertAlmostEqual(por_nivel(10, 5), 1.15**2)
        self.assertEqual(por_nivel(10, None), 0.0)
        # nivel 2 pesa el doble que el 1
        self.assertAlmostEqual(de_juego([10, 40], [10, 20]), (1 * 1.0 + 2 * 0.25) / 3)
        self.assertIsNone(de_juego([5], None))


@unittest.skipUnless(HAY_ARCENGINE, "requiere arcengine (requirements-arc.txt)")
class AgenteEnJuegosTest(unittest.TestCase):
    """Aceptación de docs/IA_OBJETOS_VIVOS.md §6 sobre el banco local."""

    def test_gana_todos_los_juegos_del_banco_con_pocas_acciones(self):
        from vivos.agente import AgenteVivo
        from vivos.arena import AgenteAleatorio, jugar
        from vivos.juegos import JUEGOS, EntornoLocal

        for nombre in JUEGOS:
            for semilla in range(3):
                with self.subTest(juego=nombre, semilla=semilla):
                    res = jugar(EntornoLocal.crear(nombre), AgenteVivo(semilla), max_acciones=600, juego=nombre)
                    self.assertEqual(res.estado, "WIN", res.linea())
                    self.assertLessEqual(res.acciones, 300, res.linea())
                    azar = jugar(EntornoLocal.crear(nombre), AgenteAleatorio(semilla), max_acciones=600)
                    self.assertLess(res.acciones, azar.acciones, f"{res.linea()} vs azar {azar.linea()}")

    def test_lo_aprendido_en_un_nivel_se_usa_en_el_siguiente(self):
        from vivos.agente import AgenteVivo
        from vivos.arena import jugar
        from vivos.juegos import EntornoLocal

        ia = AgenteVivo(0)
        res = jugar(EntornoLocal.crear("laberinto"), ia, max_acciones=600)
        self.assertEqual(res.niveles, 2)
        # El nivel 2 es más largo, pero ya sabe qué teclas mueven, qué color frena y
        # qué color hace ganar: va casi derecho (camino mínimo: 28 movimientos).
        self.assertLessEqual(res.acciones_por_nivel[1], 35, res.linea())
        self.assertIn(9, ia.colonia.colores_meta())
        self.assertEqual(ia.colonia.regla("contacto:5"), "bloquea")
        self.assertIn(11, ia.ruido.relojes)  # el reloj se heredó del nivel 1

    def test_aprende_que_la_lava_mata_y_la_evita(self):
        from vivos.agente import AgenteVivo
        from vivos.arena import jugar
        from vivos.juegos import LAVA, EntornoLocal

        ia = AgenteVivo(0)
        res = jugar(EntornoLocal.crear("trampa"), ia, max_acciones=600)
        self.assertEqual(res.estado, "WIN")
        self.assertLessEqual(res.reinicios, 2)
        self.assertEqual(ia.colonia.regla(f"contacto:{LAVA}"), FATAL)

    def test_tope_de_tiempo_corta_el_juego(self):
        from vivos.agente import AgenteVivo
        from vivos.arena import jugar
        from vivos.juegos import EntornoLocal

        res = jugar(EntornoLocal.crear("laberinto"), AgenteVivo(0), max_acciones=600, segundos_max=0.0)
        self.assertEqual(res.estado, "TIEMPO")
        self.assertEqual(res.acciones, 0)

    def test_adaptador_de_arc_agi_3_agents(self):
        from vivos.adaptador_agents import crear_clase
        from vivos.juegos import EntornoLocal

        class AgentMinimo:  # la forma de agents.agent.Agent que usa el adaptador
            def __init__(self, entorno):
                self.entorno = entorno

        Clase = crear_clase(AgentMinimo)
        entorno = EntornoLocal.crear("interruptores")
        agente = Clase(entorno)
        obs = entorno.reset()
        for _ in range(100):
            if agente.is_done([], obs):
                break
            accion = agente.choose_action([], obs)
            obs = entorno.step(accion, accion.action_data.model_dump())
        self.assertEqual(obs.state.value, "WIN")


class TestOrdenDeClicks(unittest.TestCase):
    def test_lo_distinto_entra_antes_que_un_patron_repetido(self):
        # Regresión de lp85 (nivel 2): con más objetos que MAX_CLICKS, cortar por
        # tamaño dejaba afuera los botones.
        from vivos.agente import MAX_CLICKS, AgenteVivo

        g = lienzo(0)
        for i in range(12):
            for j in range(8):
                g[4 + i * 4, 4 + j * 4] = 5  # 96 celditas iguales: un patrón
        bloque(g, 56, 30, 8, lado=3)  # dos botones distintos, más grandes
        bloque(g, 56, 40, 14, lado=3)
        agente = AgenteVivo(0)
        nodo = agente.grafo.nodo(g, agente.ruido.mascara)
        cand = agente._candidatas(nodo, [6])
        self.assertLessEqual(len(cand), MAX_CLICKS)
        self.assertIn("ACTION6@57,31", cand)
        self.assertIn("ACTION6@57,41", cand)


class TestPresupuesto(unittest.TestCase):
    def _obs(self):
        from types import SimpleNamespace

        return SimpleNamespace(frame=[lienzo(3).tolist()], available_actions=[1, 2, 3, 4],
                               levels_completed=0, win_levels=1, state="NOT_FINISHED")

    def test_presupuesto_solo_con_muertes_regulares(self):
        from vivos.agente import AgenteVivo

        a = AgenteVivo(0)
        a.vidas = [42, 42]
        self.assertEqual(a.presupuesto(), 42)
        a.vidas = [5, 42]  # una muerte por lava y una por contador: no alcanza
        self.assertIsNone(a.presupuesto())
        a.vidas = [40, 42, 7]
        self.assertEqual(a.presupuesto(), 42)
        a.paso_vida = 50  # esta vida ya duró más: hubo recarga
        self.assertIsNone(a.presupuesto())

    def test_nunca_reinicia_en_el_paso_cero(self):
        # Con 0 acciones desde el inicio del nivel, RESET reinicia el JUEGO (nivel 1).
        from vivos.agente import AgenteVivo, Decision

        a = AgenteVivo(0)
        a._elegir = lambda *args: Decision("RESET")
        self.assertNotEqual(a.decidir(self._obs()).accion, "RESET")
        a.paso_vida = 3
        self.assertEqual(a.decidir(self._obs()).accion, "RESET")

    def test_reinicia_si_desde_aca_no_se_llega_a_nada_por_probar(self):
        from vivos.agente import AgenteVivo

        a = AgenteVivo(0)
        grids = []
        for i in range(4):
            g = lienzo(3)
            bloque(g, 4 + 8 * i, 4, 9)
            grids.append(g)
        nodos = [a.grafo.nodo(g, a.ruido.mascara) for g in grids]
        for i in range(3):  # cadena irreversible: raíz -> 1 -> 2 -> 3
            a.grafo.arista(nodos[i], "ACTION4", nodos[i + 1])
        a.raiz = nodos[0].id
        a.paso_vida = 3
        meta = lambda n: n.id == nodos[1].id  # noqa: E731  lo pendiente quedó atrás
        d = a._reiniciar_conviene(nodos[3], None, meta)
        self.assertIsNotNone(d)
        self.assertEqual(d.accion, "RESET")
        self.assertEqual([x.clave for x in a._plan_tras_reset], ["ACTION4"])
        a.paso_vida = 0
        self.assertIsNone(a._reiniciar_conviene(nodos[3], None, meta))


class TestMetaAprendida(unittest.TestCase):
    def test_unidad_de_movimiento_es_la_vista_al_moverse_no_la_mas_grande(self):
        # Regresión de ls20: un ícono del mismo color, más alto que el avatar, duplicaba el paso.
        col = Colonia()
        col._tam_visto[9] = {(5, 3): 10}
        g = lienzo(0)
        bloque(g, 10, 10, 9, lado=5)  # ícono 5x5 de color 9
        g[40:43, 30:35] = 9  # avatar 5x3
        self.assertEqual(col.tamanio_avatar(g, 9, np.zeros((64, 64), dtype=bool)), (5, 3))

    def test_meta_es_lo_que_pisa_la_jugada_ganadora_y_no_el_piso(self):
        from vivos.agente import AgenteVivo, Decision

        a = AgenteVivo(0)
        g = lienzo(0)
        g[20:60, 0:64] = 5  # piso abundante
        bloque(g, 30, 30, 9, lado=2)  # avatar
        bloque(g, 30, 32, 3, lado=2)  # meta justo abajo
        a.colonia.avatar = lambda: 9
        a.colonia.desplazamientos = lambda grid, m: {"ACTION2": (9, 0, 2)}
        a._aprender_meta(g, np.zeros((64, 64), dtype=bool), Decision("ACTION2"))
        self.assertEqual(list(a.metas_color), [3])


class TestNovedad(unittest.TestCase):
    def test_un_estado_sin_atomos_nuevos_no_es_novedoso(self):
        from vivos.agente import AgenteVivo

        a = AgenteVivo(0)
        m = np.zeros((64, 64), dtype=bool)
        g1, g2, g3 = lienzo(0), lienzo(0), lienzo(0)
        bloque(g1, 10, 10, 9)
        bloque(g2, 20, 10, 9)
        bloque(g3, 10, 10, 9)
        bloque(g3, 20, 10, 9)  # solo combina lo ya visto (IW(1)): no es novedoso
        for g in (g1, g2, g3):
            a._registrar_novedad(a.grafo.nodo(g, m), g, m)
        n1, n2, n3 = (a.grafo.nodo(g, m) for g in (g1, g2, g3))
        self.assertTrue(a.novedad[n1.id])
        self.assertTrue(a.novedad[n2.id])
        self.assertFalse(a.novedad[n3.id])


class TestSinCandidatas(unittest.TestCase):
    """Regresión de tn36: un juego de solo clicks sin objetos dejaba al agente sin acciones."""

    def _obs(self, disponibles):
        from types import SimpleNamespace

        return SimpleNamespace(frame=[lienzo(3).tolist()], available_actions=disponibles,
                               levels_completed=0, win_levels=1, state="NOT_FINISHED")

    def test_solo_clicks_sin_objetos_sondea_la_grilla(self):
        from vivos.agente import AgenteVivo

        agente = AgenteVivo(0)
        obs = self._obs([6])
        for _ in range(80):  # más que la grilla de sondeo: también cubre el "todo probado"
            d = agente.decidir(obs)
            self.assertEqual(d.accion, "ACTION6")
            self.assertIsNotNone(d.x)
            agente.aprender(obs, d, obs)

    def test_todo_probado_no_repite_lo_que_no_cambia_nada(self):
        # Regresión de vc33: con todo probado elegía al azar entre acciones que dejan
        # el estado igual y se quedaba cientos de pasos en el mismo click.
        from vivos.agente import AgenteVivo

        agente = AgenteVivo(0)
        a, b = lienzo(3), lienzo(3)
        bloque(a, 10, 10, 9)
        bloque(b, 40, 40, 9)
        quieto, cambia = "ACTION6@41,41", "ACTION6@11,11"
        nodo_a = agente.grafo.nodo(a, agente.ruido.mascara)
        nodo_b = agente.grafo.nodo(b, agente.ruido.mascara)
        agente.grafo.arista(nodo_a, quieto, nodo_a)
        agente.grafo.arista(nodo_a, cambia, nodo_b)
        agente.grafo.arista(nodo_b, quieto, nodo_b)
        agente.grafo.arista(nodo_b, cambia, nodo_a)
        agente._candidatas = lambda nodo, disponibles: [quieto, cambia]
        elegidas = {agente._elegir(a, agente.ruido.mascara, nodo_a, [6]).clave for _ in range(30)}
        self.assertEqual(elegidas, {cambia})

    def test_sin_ninguna_accion_no_rompe(self):
        from vivos.agente import AgenteVivo

        agente = AgenteVivo(0)
        agente.ruido.mascara = np.ones((64, 64), dtype=bool)
        d = agente.decidir(self._obs([6]))
        self.assertEqual(d.accion, "ACTION6")


if __name__ == "__main__":
    unittest.main()
