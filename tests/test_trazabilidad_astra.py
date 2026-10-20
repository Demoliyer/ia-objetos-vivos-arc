# SPDX-License-Identifier: MIT-0
import contextlib
import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import fuentes_salida as fs
import identidad_notebook as nbid


class TestFuentes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        git = "\n".join(f"  {r} {fs.REFERENCIA[r]} DIRTY" for r in ("ARC3-Inference", "tufa-arc-agi-framework", "re-arc-3"))
        (self.d / "git_status.txt").write_text(git)
        self.log = (
            f"sglang {fs.REFERENCIA['sglang']}\nNext line\n"
            f"torch {fs.REFERENCIA['torch']}\nNext line\n"
            f"Target: /kaggle/input/models/dfranzen/{fs.REFERENCIA['modelo']} | shards: 4\n"
            f"Draft source: /kaggle/input/models/dfranzen/{fs.REFERENCIA['borrador']}\n"
            "SGLang draft view: /tmp/view (4 linked weight shards)\n"
            "harness patch applied successfully\n"
        )
        (self.d / "arc-agi-3-vivos.log").write_text(self.log)

    def tearDown(self):
        self.tmp.cleanup()

    def codigo(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = fs.main([self.d])
        return rc, out.getvalue()

    def test_multilinea_no_captura_siguiente_linea(self):
        self.assertEqual(fs.fuentes(self.d), fs.REFERENCIA)
        self.assertEqual(self.codigo()[0], 0)

    def test_gzip_equivale_a_texto(self):
        for name in ["git_status.txt", "arc-agi-3-vivos.log"]:
            p = self.d / name
            with gzip.open(str(p) + ".gz", "wt") as f:
                f.write(p.read_text())
            p.unlink()
        self.assertEqual(fs.fuentes(self.d), fs.REFERENCIA)

    def test_faltantes_no_son_diferencias_observadas(self):
        (self.d / "git_status.txt").unlink()
        faltan, dif = fs.comparar(fs.fuentes(self.d))
        self.assertEqual(len(faltan), 3)
        self.assertEqual(dif, {})
        rc, out = self.codigo()
        self.assertEqual(rc, 2)
        self.assertIn("INCOMPLETO", out)
        self.assertNotIn("DIFIERE", out)

    def test_cambio_real_se_distingue(self):
        (self.d / "arc-agi-3-vivos.log").write_text(self.log.replace(fs.REFERENCIA['sglang'], '0.5.20'))
        self.assertEqual(self.codigo()[0], 1)

    def test_cambio_y_faltante_no_se_ocultan(self):
        (self.d / "git_status.txt").unlink()
        (self.d / "arc-agi-3-vivos.log").write_text(self.log.replace(fs.REFERENCIA['sglang'], '0.5.20'))
        self.assertEqual(self.codigo()[0], 3)

    def test_tabulacion_y_barra_separan_version(self):
        log = self.log.replace(f"sglang {fs.REFERENCIA['sglang']}\n", f"sglang\t{fs.REFERENCIA['sglang']}|torch {fs.REFERENCIA['torch']}\n")
        (self.d / "arc-agi-3-vivos.log").write_text(log)
        self.assertEqual(fs.fuentes(self.d), fs.REFERENCIA)


class TestIdentidad(unittest.TestCase):
    def test_detecta_tipo_y_conserva_huella_v1(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.ipynb"
            obj = {"cells": [{"cell_type": "code", "source": "print(1)", "outputs": []}]}
            p.write_text(json.dumps(obj))
            a = nbid.identidad(p)
            obj["cells"][0]["cell_type"] = "raw"
            p.write_text(json.dumps(obj))
            b = nbid.identidad(p)
            self.assertEqual(a["huella_v1"], b["huella_v1"])
            self.assertNotEqual(a["sha256_celdas_tipadas"], b["sha256_celdas_tipadas"])

    def test_celdas_tipadas_ignora_outputs_normaliza_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.ipynb"
            obj = {"cells": [{"cell_type": "code", "source": ["x=1\n", "print(x)"], "outputs": []}]}
            p.write_text(json.dumps(obj))
            a = nbid.identidad(p)
            obj["cells"][0]["source"] = "x=1\nprint(x)"
            obj["cells"][0]["outputs"] = [{"text": "1"}]
            p.write_text(json.dumps(obj))
            b = nbid.identidad(p)
            self.assertEqual(a["sha256_celdas_tipadas"], b["sha256_celdas_tipadas"])
            self.assertNotEqual(a["sha256_archivo"], b["sha256_archivo"])


if __name__ == "__main__":
    unittest.main()
