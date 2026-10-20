# SPDX-License-Identifier: MIT-0
"""Recupera versiones históricas de un notebook de Kaggle: fuente, metadatos y salida chica.

Solo lectura. La CLI 2.2.4 manda la versión dentro del slug y Kaggle responde 403 (pull) o
devuelve la última versión (output/status). La API acepta el campo `version_label="v<N>"`
con el slug limpio (Kaggle/kaggle-cli issue #1200, PR #1201, todavía sin publicar), y
kagglesdk 2.2.4 ya trae ese campo; acá se usa directamente.

    KAGGLE_API_TOKEN=... python kaggle/recuperar_versiones.py <owner/kernel> <desde> <hasta> <dir>
        [--salida v1,v2,...|todas] [--max-mb 5]

Por versión escribe <dir>/v<N>/: notebook.ipynb (fuente tal cual), meta.json (metadatos,
estado y huella de la fuente) y, con --salida, salida/ con los archivos de salida de hasta
--max-mb MB (se saltean *_requests.jsonl, .pkl, eventos y videos) más el log del kernel.
Al final <dir>/resumen.json y un control: si dos versiones distintas devuelven la misma
fuente y el mismo current_version_number, la API no está respetando la versión.
"""
import json
import sys
import time
from pathlib import Path

SALTEAR = ("_requests.jsonl", ".pkl", "_events.jsonl", ".mp4", ".gif", ".webm")


def _a_dict(obj):
    if obj is None:
        return None
    out = {}
    for k in dir(obj):
        if k.startswith("_") or callable(getattr(type(obj), k, None)) and not isinstance(getattr(type(obj), k, None), property):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return out


def _huella_texto(fuente):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import identidad_notebook as idn
    import tempfile
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as f:
            f.write(fuente)
        i = idn.identidad(f.name)
        return {"huella": i["huella_v1"], "sha256_celdas_tipadas": i["sha256_celdas_tipadas"],
                "sha256_archivo": i["sha256_archivo"]}
    except Exception as e:  # la fuente no es un notebook JSON
        return {"huella": f"no disponible ({type(e).__name__})", "sha256_celdas_tipadas": "no disponible"}
    finally:
        Path(f.name).unlink(missing_ok=True)


def main(argv):
    if len(argv) < 5:
        sys.exit(__doc__)
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest)

    kernel, desde, hasta, dest = argv[1], int(argv[2]), int(argv[3]), Path(argv[4])
    owner, slug = kernel.split("/", 1)
    con_salida = set()
    if "--salida" in argv:
        val = argv[argv.index("--salida") + 1]
        con_salida = set(range(desde, hasta + 1)) if val == "todas" else {int(x.lstrip("v")) for x in val.split(",")}
    max_bytes = float(argv[argv.index("--max-mb") + 1] if "--max-mb" in argv else 5) * 1e6
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    resumen = []
    with api.build_kaggle_client() as k:
        cli = k.kernels.kernels_api_client
        for v in range(desde, hasta + 1):
            fila = {"version": v}
            d = dest / f"v{v}"
            try:
                req = ApiGetKernelRequest()
                req.user_name, req.kernel_slug, req.version_label = owner, slug, f"v{v}"
                r = cli.get_kernel(req)
                fuente = r.blob.source if r.blob else ""
                d.mkdir(parents=True, exist_ok=True)
                (d / "notebook.ipynb").write_text(fuente)
                meta = _a_dict(r.metadata) or {}
                meta.pop("source", None)
                fila.update(_huella_texto(fuente))
                fila["current_version_number"] = meta.get("current_version_number")
                fila["last_run_time"] = meta.get("last_run_time")
                fila["dataset_data_sources"] = meta.get("dataset_data_sources")
                try:
                    sreq = ApiGetKernelSessionStatusRequest()
                    sreq.user_name, sreq.kernel_slug, sreq.version_label = owner, slug, f"v{v}"
                    s = cli.get_kernel_session_status(sreq)
                    fila["estado"] = str(s.status)
                    fila["falla"] = s.failure_message or ""
                except Exception as e:
                    fila["estado"] = f"no disponible ({type(e).__name__}: {str(e)[:120]})"
                (d / "meta.json").write_text(json.dumps({**meta, **fila}, indent=1, default=str))
            except Exception as e:
                fila["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                resumen.append(fila)
                print(json.dumps(fila, default=str), flush=True)
                continue
            if v in con_salida:
                bajados, salteados = [], []
                token = None
                try:
                    while True:
                        oreq = ApiListKernelSessionOutputRequest()
                        oreq.user_name, oreq.kernel_slug, oreq.version_label = owner, slug, f"v{v}"
                        if token:
                            oreq.page_token = token
                        o = cli.list_kernel_session_output(oreq)
                        if o.log and not token:
                            (d / "salida").mkdir(exist_ok=True)
                            (d / "salida" / "kernel.log").write_text(o.log)
                        for it in o.files or []:
                            nombre = it.file_name
                            if nombre.endswith(SALTEAR):
                                salteados.append(nombre)
                                continue
                            resp = requests.get(it.url, stream=True, timeout=60)
                            tam = int(resp.headers.get("content-length") or 0)
                            if tam > max_bytes:
                                salteados.append(f"{nombre} ({tam} B)")
                                resp.close()
                                continue
                            destino = (d / "salida" / nombre).resolve()
                            if not str(destino).startswith(str((d / "salida").resolve())):
                                salteados.append(f"{nombre} (ruta fuera)")
                                continue
                            destino.parent.mkdir(parents=True, exist_ok=True)
                            destino.write_bytes(resp.content)
                            bajados.append(nombre)
                        token = o.next_page_token
                        if not token:
                            break
                except Exception as e:
                    fila["salida_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                fila["salida_bajados"] = len(bajados)
                fila["salida_salteados"] = salteados[:50]
            resumen.append(fila)
            print(json.dumps({k: fila.get(k) for k in ("version", "huella", "current_version_number",
                              "last_run_time", "estado", "salida_bajados", "error")}, default=str), flush=True)
            time.sleep(0.5)
    # Control de que la versión se respeta: fuentes distintas entre versiones.
    ok = [f for f in resumen if "error" not in f]
    distintas = len({f.get("huella") for f in ok})
    control = {"versiones_ok": len(ok), "huellas_distintas": distintas,
               "version_respetada": "sí" if distintas > 1 or len(ok) <= 1 else "NO: todas iguales"}
    (dest / "resumen.json").write_text(json.dumps({"kernel": kernel, "control": control, "versiones": resumen},
                                                  indent=1, default=str))
    print("CONTROL", json.dumps(control))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
