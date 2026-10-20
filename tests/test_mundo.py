"""Modelo del mundo simulable (vivos/mundo.py): predice el cuadro siguiente."""

from __future__ import annotations

import unittest

import numpy as np

from vivos.mundo import ModeloMundo


def lienzo(bg: int = 0) -> np.ndarray:
    return np.full((64, 64), bg, dtype=np.int16)


def con_avatar(x: int, y: int, piso_x: int | None = None) -> np.ndarray:
    g = lienzo()
    g[30:34, 40:44] = 4  # pared
    if piso_x is not None:
        g[y : y + 2, piso_x : piso_x + 2] = 7  # piso de color debajo del camino
    g[y : y + 2, x : x + 2] = 9
    g[y, x + 1] = 3  # indicador: el sprite es multicolor
    return g


class MundoTest(unittest.TestCase):
    def test_sprite_rigido_generaliza_a_posiciones_no_vistas(self):
        m = ModeloMundo()
        for x in (2, 4, 6):  # aprende ACTION4 = mover 2 a la derecha
            m.observar(con_avatar(x, 10), "ACTION4", con_avatar(x + 2, 10), None)
        pred = m.predecir(con_avatar(20, 10), "ACTION4", None)  # posición nunca vista
        self.assertTrue(np.array_equal(pred, con_avatar(22, 10)))

    def test_sprite_frenado_por_un_color_que_solo_frena(self):
        m = ModeloMundo()
        for x in (2, 4, 6):
            m.observar(con_avatar(x, 10), "ACTION4", con_avatar(x + 2, 10), None)
        contra = con_avatar(38, 30)
        m.observar(contra, "ACTION4", contra, None)  # contra la pared 4: no se movió
        otra = con_avatar(38, 32)
        self.assertTrue(np.array_equal(m.predecir(otra, "ACTION4", None), otra))

    def test_parche_relativo_al_boton_clickeado(self):
        # Clickear un botón 5 prende la lámpara de la derecha (0 -> 8), donde esté el botón.
        def tablero(bx, encendida=False):
            g = lienzo()
            g[5:7, bx : bx + 2] = 5
            g[5:7, bx + 4 : bx + 6] = 8 if encendida else 2
            return g

        m = ModeloMundo()
        m.observar(tablero(10), "ACTION6@10,5", tablero(10, True), None)
        m.observar(tablero(30), "ACTION6@30,5", tablero(30, True), None)
        pred = m.predecir(tablero(50), "ACTION6@50,5", None)
        self.assertTrue(np.array_equal(pred, tablero(50, True)))
        # Si la lámpara ya está prendida, la condición ("antes" = 2) no se cumple: nada.
        ya = tablero(50, True)
        self.assertTrue(np.array_equal(m.predecir(ya, "ACTION6@50,5", None), ya))

    def test_una_regla_que_falla_pierde_la_palabra(self):
        m = ModeloMundo()
        for x in (2, 4, 6):
            m.observar(con_avatar(x, 10), "ACTION4", con_avatar(x + 2, 10), None)
        for x in (20, 24, 28, 32, 36):  # de golpe deja de moverse (otro modo de juego)
            g = con_avatar(x, 20)
            m.observar(g, "ACTION4", g, None)
        nueva = con_avatar(10, 40)
        self.assertTrue(np.array_equal(m.predecir(nueva, "ACTION4", None), nueva))


if __name__ == "__main__":
    unittest.main()
