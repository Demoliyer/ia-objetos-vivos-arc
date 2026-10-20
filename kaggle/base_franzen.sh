#!/usr/bin/env bash
# Reproduce en tu cuenta la solución pública del hito 2 (Daniel Franzen, 27,9 %) y la entrega.
# Es la base medida sobre la que se injertan las piezas de la IA de Objetos Vivos.
#
#   bash kaggle/base_franzen.sh            # copia el notebook, lo corre, espera y lo entrega
#   bash kaggle/base_franzen.sh --sin-entregar
#
# Usa el kit ya instalado por kaggle/entregar.sh (kaggle/kit/.venv y kaggle/kit/.kaggle).
# Crédito: solución de Daniel Franzen sobre el Duck Harness de Tufa Labs (código abierto).
set -euo pipefail
cd "$(dirname "$0")/.."
KIT="kaggle/kit"
ORIGEN="dfranzen/arc-agi-3-milestone-2-solution"
COMPE="arc-prize-2026-arc-agi-3"
SLUG="arc-agi-3-base-franzen"
DIR="kaggle/base_franzen"
ENTREGAR=1
[ "${1:-}" = "--sin-entregar" ] && ENTREGAR=0

[ -x "$KIT/.venv/bin/kaggle" ] || { echo "Falta el kit: corré antes bash kaggle/entregar.sh" >&2; exit 1; }
[ -s "$KIT/.kaggle/access_token" ] || { echo "Falta el token en $KIT/.kaggle/access_token" >&2; exit 1; }
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(cat "$KIT/.kaggle/access_token")
K="$KIT/.venv/bin/kaggle"
USUARIO=$(tr -d '[:space:]' < kaggle/usuario.txt)
USUARIO_MIN=$(echo "$USUARIO" | tr '[:upper:]' '[:lower:]')

echo "== 1/4 Bajando el notebook de $ORIGEN (con sus datasets, modelo y acelerador)"
rm -rf "$DIR" && mkdir -p "$DIR"
"$K" kernels pull "$ORIGEN" -p "$DIR" -m

echo "== 2/4 Pasándolo a tu cuenta ($USUARIO_MIN/$SLUG)"
"$KIT/.venv/bin/python" - "$DIR/kernel-metadata.json" "$USUARIO_MIN/$SLUG" <<'EOF'
import json, sys
ruta, nuevo = sys.argv[1], sys.argv[2]
m = json.load(open(ruta))
m["id"] = nuevo
m["title"] = "ARC-AGI-3 base Franzen"
m.pop("id_no", None)
m["is_private"] = True
m["enable_internet"] = False  # obligatorio para entregar
json.dump(m, open(ruta, "w"), indent=2)
print(json.dumps({k: m.get(k) for k in ("id", "kernel_type", "enable_gpu", "enable_internet",
      "machine_shape", "accelerator", "dataset_sources", "model_sources", "competition_sources")}, indent=2))
EOF

echo "== 3/4 Subiendo y corriendo (Save & Run All)"
SALIDA=$("$K" kernels push -p "$DIR" 2>&1); echo "$SALIDA"
VERSION=$(echo "$SALIDA" | grep -oE "version [0-9]+" | head -1 | grep -oE "[0-9]+" || true)
[ -n "$VERSION" ] || { echo "No se pudo subir el notebook (ver arriba)" >&2; exit 1; }
echo "Esperando a que termine (puede tardar bastante: carga un modelo de 27B)..."
while true; do
  ESTADO=$("$K" kernels status "$USUARIO_MIN/$SLUG" 2>&1 || true)
  echo "  $(date +%H:%M) $ESTADO"
  case "$ESTADO" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) echo "La corrida falló. Mirá el registro en https://www.kaggle.com/code/$USUARIO_MIN/$SLUG" >&2; exit 1 ;;
  esac
  sleep 120
done

if [ "$ENTREGAR" = 0 ]; then
  echo "Listo, sin entregar. Para entregar: Submit to Competition en https://www.kaggle.com/code/$USUARIO_MIN/$SLUG"
  exit 0
fi

echo "== 4/4 Entregando a la competencia"
"$K" competitions submit "$COMPE" -k "$USUARIO_MIN/$SLUG" -v "$VERSION" -f submission.parquet \
  -m "Base: solución hito 2 de D. Franzen (Duck Harness, Tufa Labs), sin cambios"
echo
echo "Entregado. Puntaje en: https://www.kaggle.com/competitions/$COMPE/submissions"
