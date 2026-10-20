# SPDX-License-Identifier: MIT-0
"""Pruebas del congelado por huella (kaggle/huella_notebook.py y kaggle/experimento.sh).

Las de experimento.sh copian kaggle/ a un directorio temporal con un `kaggle` simulado
(nunca tocan el kit real ni Kaggle). Necesitan el notebook de Franzen en
kaggle/franzen_nb/original.ipynb (no versionado); sin él se saltean.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "kaggle"))
import fuentes_salida as fs  # noqa: E402
import huella_notebook as hn  # noqa: E402

ORIGEN = RAIZ / "kaggle" / "franzen_nb" / "original.ipynb"
HUELLA_ORIGEN = "4056fbfc69bb"
# Condiciones que se repiten en el set oculto: Franzen sin injerto y vivos-auto (la de 29,28 y 25,71).
# Si un cambio en esta rama altera alguno de estos notebooks, la prueba falla antes de entregar.
HUELLAS = {(): "5c66785823b6", ("--vivos-auto",): "413a0b289a64"}  # rama v4s: --vivos-auto produce v4s
TIPADAS = json.loads((RAIZ / "kaggle" / "identidades_candidatos.json").read_text())


def _nb(ruta, fuentes, **extra):
    celdas = [{"cell_type": "code", "source": s, "metadata": {}, "outputs": extra.get("outputs", []),
               "execution_count": extra.get("n")} for s in fuentes]
    Path(ruta).write_text(json.dumps({"cells": celdas, "metadata": extra.get("meta", {}), "nbformat": 4,
                                      "nbformat_minor": 5}))


class TestHuella(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_ignora_salidas_y_metadatos(self):
        _nb(self.d / "a.ipynb", ["x = 1\n", "print(x)"])
        _nb(self.d / "b.ipynb", ["x = 1\n", "print(x)"], outputs=[{"text": "1"}], n=7, meta={"k": 1})
        self.assertEqual(hn.huella(self.d / "a.ipynb"), hn.huella(self.d / "b.ipynb"))

    def test_lista_o_texto_dan_lo_mismo(self):
        _nb(self.d / "a.ipynb", [["x = 1\n", "y = 2"]])
        _nb(self.d / "b.ipynb", ["x = 1\ny = 2"])
        self.assertEqual(hn.huella(self.d / "a.ipynb"), hn.huella(self.d / "b.ipynb"))

    def test_un_caracter_o_el_orden_cambian_la_huella(self):
        _nb(self.d / "a.ipynb", ["x = 1", "y = 2"])
        _nb(self.d / "b.ipynb", ["x = 2", "y = 2"])
        _nb(self.d / "c.ipynb", ["y = 2", "x = 1"])
        h = {hn.huella(self.d / f"{n}.ipynb") for n in "abc"}
        self.assertEqual(len(h), 3)

    def test_main_codigo_de_salida(self):
        _nb(self.d / "a.ipynb", ["x"])
        h = hn.huella(self.d / "a.ipynb")
        self.assertEqual(hn.main(["", str(self.d / "a.ipynb"), h]), 0)
        self.assertEqual(hn.main(["", str(self.d / "a.ipynb"), "000000000000"]), 1)


GIT = """git status:
  ARC3-Inference                   aa69123     DIRTY  add-kaggle-share-flag     Add flag
  tufa-arc-agi-framework           fe9f7c4     clean  submission-share-mode-bugfix  Fix
  re-arc-3                         57e46d619d  pinned  master                    (non-editable)"""


class TestFuentes(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d)

    def _log(self, texto):
        (self.d / "arc-agi-3-vivos.log").write_text(texto)

    def test_saltos_de_linea_reales_no_se_pegan_al_valor(self):
        (self.d / "git_status.txt").write_text(GIT)
        self._log("harness patch applied\nsglang 0.5.19+gd00d88efc8d6\nlinea_siguiente\n"
                  "torch 2.13.0+cu130\notra\n"
                  "Target: /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/"
                  "transformers/default/1\nSIGUIENTE\n"
                  "Draft source: /kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/"
                  "transformers/default/1\nSIGUIENTE\n")
        self.assertEqual(fs.fuentes(self.d), fs.REFERENCIA)
        e, dif, falta = fs.estado(self.d)
        self.assertTrue(e.startswith("ok"), e)
        self.assertIn("dirty sin identificar: ARC3-Inference", e)

    def test_escapes_json_tambien(self):
        (self.d / "git_status.txt").write_text(GIT)
        self._log('"harness patch applied\\nsglang 0.5.19+gd00d88efc8d6\\ntorch 2.13.0+cu130\\n'
                  'Target: /kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/'
                  'transformers/default/1 | shards: 38\\nDraft source: /kaggle/input/models/dfranzen/'
                  'albucino-qwen3-8-flash-next-drafter/transformers/default/1\\n"')
        e, dif, falta = fs.estado(self.d)
        self.assertTrue(e.startswith("ok"), e)

    def test_incompleto_no_es_lo_mismo_que_distinto(self):
        self._log("sglang 0.5.19+gd00d88efc8d6\n")  # sin git_status ni lo demás
        e, dif, falta = fs.estado(self.d)
        self.assertTrue(e.startswith("INCOMPLETO"), e)
        self.assertFalse(dif)
        self.assertEqual(fs.main([str(self.d)]), 2)
        self._log("sglang 0.6.0\n")
        e, dif, falta = fs.estado(self.d)
        self.assertTrue(e.startswith("DIFIERE"), e)
        self.assertEqual(fs.main([str(self.d)]), 3)  # difiere y además falta registro


@unittest.skipUnless(ORIGEN.exists(), "falta kaggle/franzen_nb/original.ipynb")
class TestExperimento(unittest.TestCase):
    """experimento.sh con un `kaggle` simulado: pull copia un notebook de origen, push solo avisa."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        shutil.copytree(RAIZ / "kaggle", self.d / "kaggle",
                        ignore=shutil.ignore_patterns("kit", "experimento", "token*", ".kaggle", "usuario.txt"))
        shutil.copytree(RAIZ / "vivos", self.d / "vivos")
        bin_ = self.d / "kaggle" / "kit" / ".venv" / "bin"
        bin_.mkdir(parents=True)
        (self.d / "kaggle" / "kit" / ".kaggle").mkdir()
        (self.d / "kaggle" / "kit" / ".kaggle" / "access_token").write_text("simulado")
        (self.d / "kaggle" / "usuario.txt").write_text("prueba\n")
        os.symlink(sys.executable, bin_ / "python")
        self.origen = self.d / "origen" / "arc-agi-3-milestone-2-solution.ipynb"
        self.origen.parent.mkdir()
        shutil.copy(ORIGEN, self.origen)
        (self.origen.parent / "kernel-metadata.json").write_text(json.dumps(
            {"id": "dfranzen/x", "machine_shape": "NvidiaRtxPro6000"}))
        k = bin_ / "kaggle"
        k.write_text(f"""#!/bin/bash
case "$1 $2" in
  "kernels pull") while [ $# -gt 0 ]; do [ "$1" = -p ] && D="$2"; shift; done
                  mkdir -p "$D"; cp {self.origen.parent}/* "$D"/ ;;
  "kernels push") echo "PUSH LLAMADO"; exit 1 ;;
  *) echo "inesperado: $*"; exit 2 ;;
esac
""")
        k.chmod(k.stat().st_mode | stat.S_IEXEC)

    def tearDown(self):
        shutil.rmtree(self.d)

    def correr(self, *args):
        r = subprocess.run(["bash", "kaggle/experimento.sh", "prueba", *args], cwd=self.d,
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_origen_esperado_y_huellas_conocidas(self):
        self.assertEqual(hn.huella(ORIGEN), HUELLA_ORIGEN)
        for opciones, esperada in HUELLAS.items():
            rc, out = self.correr(*opciones)
            self.assertIn(f"huella_origen={HUELLA_ORIGEN}", out)
            self.assertIn("PUSH LLAMADO", out)  # llegó a subir: el push simulado falla a propósito
            self.assertIn(f"huella_notebook={esperada}", out, opciones)
            self.assertIn(f"tipada_notebook={TIPADAS['candidatos'][esperada]}", out, opciones)
            self.assertIn(f"tipada_origen={TIPADAS['origen'][HUELLA_ORIGEN]}", out)

    def test_huella_correcta_sube_e_incorrecta_corta_antes(self):
        rc, out = self.correr("--huella=5c66785823b6")
        self.assertIn("PUSH LLAMADO", out)
        rc, out = self.correr("--huella=000000000000")
        self.assertNotEqual(rc, 0)
        self.assertNotIn("PUSH LLAMADO", out)
        self.assertIn("no se sube", out)

    def test_origen_cambiado_corta_y_origen_explicito_lo_acepta(self):
        nb = json.loads(self.origen.read_text())
        nb["cells"][3]["source"] = "# cambio\n" + "".join(nb["cells"][3]["source"])
        self.origen.write_text(json.dumps(nb))
        rc, out = self.correr()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("PUSH LLAMADO", out)
        self.assertIn("El notebook de Franzen cambió", out)
        rc, out = self.correr(f"--origen={hn.huella(self.origen)}")
        self.assertIn("PUSH LLAMADO", out)

    def test_celda_de_codigo_pasada_a_raw_corta_antes_de_subir(self):
        # En el origen: la huella v1 no cambia, la tipada sí.
        nb = json.loads(self.origen.read_text())
        i = next(k for k, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
        nb["cells"][i]["cell_type"] = "raw"
        self.origen.write_text(json.dumps(nb))
        self.assertEqual(hn.huella(self.origen), HUELLA_ORIGEN)
        rc, out = self.correr()
        self.assertNotEqual(rc, 0)
        self.assertNotIn("PUSH LLAMADO", out)
        self.assertIn("cambió el tipo de alguna celda", out)

    def test_candidato_con_celda_raw_corta_antes_de_subir(self):
        # Envuelve preparar_franzen: corre el original y deja la última celda de código como raw.
        k = self.d / "kaggle"
        (k / "preparar_franzen.py").rename(k / "preparar_original.py")
        (k / "preparar_franzen.py").write_text("""import json, runpy, sys
destino = sys.argv[2]
try:
    runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("preparar_original.py")), run_name="__main__")
except SystemExit as e:
    if e.code not in (0, None):
        raise
nb = json.load(open(destino))
k = max(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
nb["cells"][k]["cell_type"] = "raw"
json.dump(nb, open(destino, "w"))
""")
        rc, out = self.correr("--vivos-auto", "--huella=413a0b289a64")
        self.assertIn("huella_notebook=413a0b289a64", out)
        self.assertNotIn("PUSH LLAMADO", out)
        self.assertIn("cambió el tipo de alguna celda", out)

    def test_huella_y_origen_no_van_a_la_descripcion_como_opciones(self):
        script = (self.d / "kaggle" / "experimento.sh").read_text()
        self.assertIn('DESC="$DESC [nb $H_NB]"', script)
        self.assertIn("--huella=*|--origen=*) ;;", script)


if __name__ == "__main__":
    unittest.main()
