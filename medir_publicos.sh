#!/usr/bin/env bash
# Mide la IA en los juegos públicos de ARC-AGI-3 sin red, en paralelo.
# Requiere environment_files/ (lo baja `jugar_arc.py --todos` en modo online la primera vez).
#   bash medir_publicos.sh            # todos, salida en medicion_local.txt
#   bash medir_publicos.sh ls20 vc33  # solo esos
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -d environment_files ] || { echo "Falta environment_files/: corré antes jugar_arc.py --todos con red" >&2; exit 1; }
if [ $# -gt 0 ]; then
  juegos=("$@")
else
  mapfile -t juegos < <(ls environment_files)
fi
salida=medicion_local_${ETIQUETA:+${ETIQUETA}_}s${SEMILLA:-0}.txt
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' "${juegos[@]}" | xargs -P "${PARALELO:-$(nproc)}" -I{} sh -c \
  "OPERATION_MODE=offline $PY jugar_arc.py --juego {} --semilla ${SEMILLA:-0} --segundos \${SEGUNDOS:-300} --exportar vivos-data 2>/dev/null | grep 'niveles,' > '$tmp/{}.txt' || echo '{}: ERROR' > '$tmp/{}.txt'"
cat "$tmp"/*.txt | sort > "$salida"
cat "$salida"
$PY - "$salida" <<'EOF'
import re, sys
lineas = open(sys.argv[1]).read().splitlines()
niv = sum(int(m.group(1)) for l in lineas if (m := re.search(r": (\d+)/\d+ niveles", l)))
tot = sum(int(m.group(1)) for l in lineas if (m := re.search(r": \d+/(\d+) niveles", l)))
pts = [float(m.group(1)) for l in lineas if (m := re.search(r"puntaje estimado ([\d.]+)", l))]
print(f"TOTAL: {niv}/{tot} niveles en {len(lineas)} juegos; puntaje estimado medio {sum(pts)/max(1,len(pts)):.4f}")
EOF
