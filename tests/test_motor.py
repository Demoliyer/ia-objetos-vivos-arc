"""Candidata M: el predictor puro del sandbox reproduce ModeloMundo.predecir (numpy)."""
import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vivos import motor_llm, motor_puro  # noqa: E402
from vivos.mundo import ModeloMundo, huella  # noqa: E402

PRACTICAS = Path(os.environ.get("VIVOS_TRAYECTORIAS", "/home/claude/exp-reglas/trayectorias/practicas"))


def tramos(ruta):
    d = np.load(ruta, allow_pickle=False)
    cuadros, acciones, cortes = d["cuadros"].astype(np.int16), [str(a) for a in d["acciones"]], d["corte"]
    tramo = []
    for t, a in enumerate(acciones):
        if cortes[t]:
            if tramo:
                yield tramo
            tramo = []
            continue
        tramo.append((cuadros[t], a, cuadros[t + 1]))
    if tramo:
        yield tramo


def comparar(test, tramo, muestras=25, semilla=0):
    r = motor_llm.aprender(tramo)
    modelo, mascara, _, _ = r
    modelo.memoria = {(huella(a, mascara), acc): d for a, acc, d in tramo[-motor_llm.MAX_MEMORIA:]}
    M = motor_puro.preparar(motor_llm.serializar(modelo, mascara, tramo))
    rnd = random.Random(semilla)
    acciones = sorted({acc for _, acc, _ in tramo})
    for _ in range(muestras):
        antes = tramo[rnd.randrange(len(tramo))][rnd.choice((0, 2))]
        acc = rnd.choice(acciones)
        esperado = modelo.predecir(antes, acc, mascara)
        obtenido = motor_puro.predecir(M, antes.tolist(), acc)
        fuera = ~mascara  # la memoria exacta conserva lo actual en las celdas enmascaradas (relojes)
        test.assertEqual(np.asarray(obtenido)[fuera].tolist(), esperado[fuera].tolist(), f"{acc}")


class Equivalencia(unittest.TestCase):
    def test_sintetico_sprite_y_click(self):
        g = np.zeros((64, 64), dtype=np.int16)
        g[20:22, 20:22] = 3
        g[40, 40] = 5
        tramo = []
        cur = g.copy()
        for _ in range(5):  # el bloque 3 baja de a 1 con ACTION2
            nxt = cur.copy()
            ys, xs = np.nonzero(cur == 3)
            nxt[cur == 3] = 0
            nxt[ys + 1, xs] = 3
            tramo.append((cur, "ACTION2", nxt))
            cur = nxt
        nxt = cur.copy()
        nxt[40, 40] = 7  # click que cambia el color del punto
        tramo.append((cur, "ACTION6@40,40", nxt))
        comparar(self, tramo, muestras=30)

    @unittest.skipUnless(PRACTICAS.is_dir(), "sin trayectorias de práctica locales")
    def test_trayectorias_reales(self):
        rutas = sorted(PRACTICAS.glob("*.npz"))
        rnd = random.Random(1)
        n = 0
        for ruta in rnd.sample(rutas, min(14, len(rutas))):
            for i, tramo in enumerate(tramos(ruta)):
                if len(tramo) < 4 or i % 3:
                    continue
                with self.subTest(ruta=ruta.name, tramo=i):
                    comparar(self, tramo, muestras=12, semilla=i)
                n += 1
        self.assertGreater(n, 10)


class Harness(unittest.TestCase):
    def _hist(self, acciones_grillas, nivel=1):
        h = []
        for acc, g in acciones_grillas:
            asc = "\n".join("".join(chr(v) for v in fila) for fila in g)
            h.append({"action": acc, "frame": {"ascii": asc, "level": nivel, "step": len(h)}, "result": {}})
        return h

    def test_motor_para_y_engine_step(self):
        B, R = ord("B"), ord("R")
        g = [[B] * 64 for _ in range(64)]
        for y in range(20, 22):
            for x in range(20, 22):
                g[y][x] = R
        pasos = [("", [f[:] for f in g])]
        cur = [f[:] for f in g]
        for _ in range(7):
            nxt = [[B] * 64 for _ in range(64)]
            for y in range(64):
                for x in range(64):
                    if cur[y][x] == R:
                        nxt[y][x + 3] = R  # sin superposición: traslación sin ambigüedad
            pasos.append(("RIGHT", nxt))
            cur = nxt
        datos, (ok, n) = motor_llm.motor_para(self._hist(pasos), 1)
        self.assertIsNotNone(datos, (ok, n))
        self.assertGreaterEqual(n, 3)
        M = motor_puro.preparar(datos)
        letras = [[chr(v) for v in f] for f in cur]
        pred = motor_puro.engine_step(M, letras, "RIGHT")
        self.assertEqual([i for i, c in enumerate(pred[20]) if c == "R"], [44, 45])
        # nivel sin regla: None, no lanza
        self.assertEqual(motor_llm.motor_para(self._hist(pasos), 2)[0], None)
        self.assertEqual(motor_llm.motor_para([None, 3, {"x": 1}], 1)[0], None)

    def test_extraccion_acotada_igual_a_la_completa(self):
        import time
        B = ord("B")
        def fr(v, lv):
            g = [[B] * 8 for _ in range(8)]
            g[3][v % 8] = ord("R")
            return {"ascii": "\n".join("".join(chr(c) for c in f) for f in g), "level": lv}
        h = [{"action": "", "frame": fr(0, 1)}]
        acciones = ["RIGHT", "RESET", "UNDO", "LEFT", "MOUSE(row=1, col=2)"]
        for i in range(1, 400):
            lv = 1 if i < 300 else 2
            res = {"game_over": True} if i % 97 == 0 else {}
            h.append({"action": acciones[i % 5], "frame": fr(i, lv), "result": res})
        for nivel in (1, 2):
            todo = motor_llm.transiciones(h, nivel)
            for tope in (1, 7, 150, 10_000):
                parcial = motor_llm.transiciones(h, nivel, tope)
                self.assertEqual(len(parcial), min(tope, len(todo)))
                for (a, x, b), (c, y, d) in zip(parcial, todo[-len(parcial):]):
                    self.assertEqual(x, y)
                    self.assertTrue(np.array_equal(a, c) and np.array_equal(b, d))
        g = [[B] * 64 for _ in range(64)]
        asc = "\n".join("".join(chr(c) for c in f) for f in g)
        largo = [{"action": "", "frame": {"ascii": asc, "level": 1}}] + [
            {"action": "UP", "frame": {"ascii": asc, "level": 1}, "result": {}} for _ in range(3000)]
        t = time.monotonic()
        self.assertEqual(len(motor_llm.transiciones(largo, 1, motor_llm.MAX_TRANS)), motor_llm.MAX_TRANS)
        self.assertLess(time.monotonic() - t, 2.0)

    def test_memoria_exacta_no_cuenta_para_el_criterio(self):
        # ida y vuelta entre dos estados: todo acierto sale de la memoria, no hay regla que ofrecer
        B, R = ord("B"), ord("R")
        def g(x):
            m = np.full((64, 64), B, dtype=np.int16)
            m[20:22, x:x + 2] = R
            return m
        tramo = []
        for i in range(10):
            a, b = (g(10), g(13)) if i % 2 == 0 else (g(13), g(10))
            tramo.append((a, "RIGHT" if i % 2 == 0 else "LEFT", b))
        _, _, ok, n = motor_llm.aprender(tramo)
        self.assertLess(n, motor_llm.MINIMO)

    def test_mouse(self):
        self.assertEqual(motor_puro.accion_motor("MOUSE(row=3, col=7)"), "MOUSE@7,3")
        self.assertEqual(motor_puro.accion_motor("up"), "UP")


if __name__ == "__main__":
    unittest.main()
