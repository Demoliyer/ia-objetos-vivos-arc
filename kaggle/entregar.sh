#!/usr/bin/env bash
# Entrega la IA de Objetos Vivos a la competencia ARC Prize 2026 - ARC-AGI-3 en Kaggle.
#
# Requisitos (una sola vez):
#   1. Cuenta de Kaggle con las reglas de la competencia aceptadas:
#      https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
#   2. Token de API: kaggle.com → Settings → API → "Create New Token".
#      Guardá el token (la cadena que muestra) en kaggle/token.txt (está en .gitignore).
#   3. Tu usuario de Kaggle en kaggle/usuario.txt.
#
# Uso:
#   bash kaggle/entregar.sh            # arma el notebook y lo sube (corre "Save & Run All")
#   bash kaggle/entregar.sh --solo-armar
#
# Después: abrí el notebook en kaggle.com, esperá que termine, tocá
# "Submit to Competition" y elegí submission.parquet. El puntaje aparece en el
# leaderboard cuando termina la re-ejecución sobre los juegos ocultos.
set -euo pipefail
cd "$(dirname "$0")/.."
RAIZ=$(pwd)
KIT="$RAIZ/kaggle/kit"
SOLO_ARMAR=0
[ "${1:-}" = "--solo-armar" ] && SOLO_ARMAR=1

if [ ! -d "$KIT" ]; then
  git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter "$KIT"
fi

# El agente es un solo archivo con vivos/ adentro.
.venv/bin/python empaquetar_kaggle.py --salida "$KIT/agent/my_agent.py"

# Sin GPU: el agente corre en CPU y así no se gasta cuota.
sed -i 's/^ACCELERATOR = ".*"/ACCELERATOR = "cpu"/' "$KIT/scripts/build_notebook.py"

if [ "$SOLO_ARMAR" = 1 ]; then
  (cd "$KIT" && python3 scripts/build_notebook.py)
  echo "Notebook armado en $KIT/notebooks/submission.ipynb"
  exit 0
fi

[ -s kaggle/token.txt ] || { echo "Falta kaggle/token.txt (ver requisitos arriba)" >&2; exit 1; }
[ -s kaggle/usuario.txt ] || { echo "Falta kaggle/usuario.txt con tu usuario de Kaggle" >&2; exit 1; }
USUARIO=$(tr -d '[:space:]' < kaggle/usuario.txt)
mkdir -p "$KIT/.kaggle"
install -m 600 kaggle/token.txt "$KIT/.kaggle/access_token"
sed -i "s/REPLACE_WITH_YOUR_USERNAME/$USUARIO/" "$KIT/notebooks/kernel-metadata.json"

cd "$KIT"
make setup
make submit
echo
echo "Listo. Seguimiento: (cd $KIT && make status)"
echo "Cuando termine: kaggle.com → tu notebook → Submit to Competition → submission.parquet"
