#!/usr/bin/env bash
# Un experimento sobre la solución del hito 2 de Franzen = una entrega diaria a Kaggle.
#
#   bash kaggle/experimento.sh base                                  # Franzen sin cambios
#   bash kaggle/experimento.sh undo-off EXPOSE_UNDO=off              # una variable distinta
#   bash kaggle/experimento.sh x ARC3_A=1 ARC3_B=2                   # varias
#   bash kaggle/experimento.sh vivos --vivos                         # + check_model / plan_with_model
#   bash kaggle/experimento.sh rend cfg.MAXREQ=14 cfg.SPEC_STEPS=4   # CFG del servidor sglang
#   SIN_ENTREGAR=1 bash kaggle/experimento.sh medir --vivos --juegos=ls20,vc33 --minutos=60
#   SIN_ENTREGAR=1 bash kaggle/experimento.sh base                   # solo guarda, no entrega
#   ACELERADOR=<nombre> bash kaggle/experimento.sh ...               # fuerza el acelerador del push
#   bash kaggle/experimento.sh vivos-auto --vivos-auto --huella=ff42d7f18a5a   # exige ese notebook
#
# Congelado: el notebook de Franzen se baja en cada corrida, así que se exige su huella
# (ORIGEN_HUELLA; --origen=<huella> la cambia a propósito). La huella del notebook preparado
# se imprime y va en la descripción de la entrega; con --huella=<h> se corta antes de subir
# si no coincide. Huella = kaggle/huella_notebook.py (fuentes de las celdas).
#
# Guardado rápido (~30-40 min de GPU) en vez de las 9 h del notebook original: ver
# kaggle/preparar_franzen.py. La re-ejecución oficial sobre los juegos ocultos no cambia
# salvo las variables pasadas.
#
# Requiere el kit instalado por kaggle/entregar.sh (kaggle/kit/.venv y el token).
# Crédito: solución de Daniel Franzen sobre el Duck Harness de Tufa Labs (código abierto).
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { sed -n 2,12p "$0"; exit 1; }
NOMBRE="$1"; shift
KIT="kaggle/kit"
ORIGEN="dfranzen/arc-agi-3-milestone-2-solution"
COMPE="arc-prize-2026-arc-agi-3"
SLUG="arc-agi-3-vivos"
DIR="kaggle/experimento"
# Notebook de Franzen verificado el 20-oct-2026 (idéntico a la copia del 15-oct).
ORIGEN_HUELLA="4056fbfc69bb"
HUELLA=""

[ -x "$KIT/.venv/bin/kaggle" ] || { echo "Falta el kit: corré antes bash kaggle/entregar.sh" >&2; exit 1; }
[ -s "$KIT/.kaggle/access_token" ] || { echo "Falta el token en $KIT/.kaggle/access_token" >&2; exit 1; }
[ -s kaggle/usuario.txt ] || { echo "Falta kaggle/usuario.txt con tu usuario de Kaggle" >&2; exit 1; }
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(cat "$KIT/.kaggle/access_token")
K="$KIT/.venv/bin/kaggle"
PY="$KIT/.venv/bin/python"
USUARIO=$(tr -d '[:space:]' < kaggle/usuario.txt | tr '[:upper:]' '[:lower:]')

ENVS=()
for par in "$@"; do
  case "$par" in
    --vivos) ENVS+=(--vivos) ;;
    --vivos-fuerte) ENVS+=(--vivos-fuerte) ;;
    --vivos-auto) ENVS+=(--vivos-auto) ;;
    --juegos=*) ENVS+=(--juegos "${par#--juegos=}") ;;    # solo el guardado rápido
    --minutos=*) ENVS+=(--minutos "${par#--minutos=}") ;;  # solo el guardado rápido
    cfg.*=*) ENVS+=(--cfg "${par#cfg.}") ;;                # CFG del servidor (también la oficial)
    --huella=*) HUELLA="${par#--huella=}" ;;
    --origen=*) ORIGEN_HUELLA="${par#--origen=}" ;;
    -*) echo "Opción desconocida: $par" >&2; exit 1 ;;
    *) ENVS+=(--env "$par") ;;
  esac
done
OPCS=()
for par in "$@"; do case "$par" in --huella=*|--origen=*) ;; *) OPCS+=("$par") ;; esac; done
DESC="Vivos/$NOMBRE: Franzen hito 2${OPCS[*]:+ + ${OPCS[*]}}"

echo "== 1/5 Bajando $ORIGEN"
rm -rf "$DIR" && mkdir -p "$DIR/origen"
"$K" kernels pull "$ORIGEN" -p "$DIR/origen" -m
NB_ORIGEN=$(ls "$DIR"/origen/*.ipynb | head -1)
H_ORIGEN=$("$PY" kaggle/huella_notebook.py "$NB_ORIGEN" "$ORIGEN_HUELLA") || {
  echo "El notebook de Franzen cambió (huella $H_ORIGEN, esperada $ORIGEN_HUELLA): no se mezcla con lo medido. Revisalo y pasá --origen=$H_ORIGEN si es a propósito." >&2
  exit 1; }
echo "huella_origen=$H_ORIGEN"
T_ORIGEN=$("$PY" kaggle/guarda_identidad.py "$NB_ORIGEN" origen) || {
  echo "El notebook de Franzen conserva la huella pero cambió el tipo de alguna celda: no se sube." >&2; exit 1; }
echo "tipada_origen=$T_ORIGEN"

echo "== 2/5 Preparando ($NOMBRE)"
"$PY" kaggle/preparar_franzen.py "$NB_ORIGEN" "$DIR/notebook.ipynb" ${ENVS[@]+"${ENVS[@]}"}
H_NB=$("$PY" kaggle/huella_notebook.py "$DIR/notebook.ipynb" ${HUELLA:+"$HUELLA"}) || {
  echo "El notebook preparado tiene huella $H_NB, no $HUELLA: no se sube." >&2; exit 1; }
echo "huella_notebook=$H_NB"
# Identidad tipada (incluye el tipo de celda): obligatoria para los candidatos conocidos
# (kaggle/identidades_candidatos.json); la huella v1 sigue siendo la etiqueta de las entregas.
T_NB=$("$PY" kaggle/guarda_identidad.py "$DIR/notebook.ipynb" candidatos) || {
  echo "El notebook preparado conserva la huella $H_NB pero cambió el tipo de alguna celda: no se sube." >&2; exit 1; }
echo "tipada_notebook=$T_NB"
DESC="$DESC [nb $H_NB]"
"$PY" - "$DIR/origen/kernel-metadata.json" "$DIR/kernel-metadata.json" "$USUARIO/$SLUG" <<'EOF'
import json, sys
origen, destino, nuevo = sys.argv[1:4]
m = json.load(open(origen))
m["id"] = nuevo
m["title"] = "ARC-AGI-3 Vivos"
m.pop("id_no", None)
m["code_file"] = "notebook.ipynb"
m["is_private"] = True
m["enable_gpu"] = True
m["enable_internet"] = False
for k in ("dataset_sources", "model_sources", "kernel_sources", "competition_sources", "keywords"):
    m[k] = m.get(k) or []
if "arc-prize-2026-arc-agi-3" not in m.get("competition_sources", []):
    m.setdefault("competition_sources", []).append("arc-prize-2026-arc-agi-3")
json.dump(m, open(destino, "w"), indent=2)
print(json.dumps({k: m.get(k) for k in ("id", "enable_gpu", "enable_internet", "machine_shape",
      "dataset_sources", "model_sources", "competition_sources")}, indent=2))
# Sin machine_shape (ni ACELERADOR en el entorno) Kaggle asignaría la GPU por defecto y la
# corrida se perdería: se corta acá, antes de subir nada.
if not m.get("machine_shape") and not __import__("os").environ.get("ACELERADOR"):
    sys.exit("machine_shape vacío en los metadatos de Franzen: corré con ACELERADOR=<nombre> "
             "(el que muestra Kaggle para 'GPU RTX Pro 6000') o elegila a mano en el notebook.")
EOF

echo "== 3/5 Subiendo y guardando (Save & Run, modo rápido)"
ACC=()
[ -n "${ACELERADOR:-}" ] && ACC=(--accelerator "$ACELERADOR")
SALIDA=$("$K" kernels push -p "$DIR" ${ACC[@]+"${ACC[@]}"} 2>&1) || true; echo "$SALIDA"
VERSION=$(echo "$SALIDA" | grep -oiE "version [0-9]+" | head -1 | grep -oE "[0-9]+" || true)
[ -n "$VERSION" ] || { echo "No se pudo subir el notebook (ver arriba)" >&2; exit 1; }

echo "== 4/5 Esperando que termine la versión $VERSION (~30-40 min)"
# Se compara con el texto exacto del CLI ('has status "KernelWorkerStatus.X"'): un error de red
# pasajero ("503 Server Error") no debe cortar la espera. Y no se acepta COMPLETE sin haber visto
# antes la versión nueva en cola o corriendo (el estado puede ser el de una versión anterior).
VISTA_NUEVA=0
while true; do
  sleep 120
  ESTADO=$("$K" kernels status "$USUARIO/$SLUG" 2>&1 || true)
  echo "  $(date +%H:%M) $ESTADO"
  case "$ESTADO" in
    *KernelWorkerStatus.QUEUED*|*KernelWorkerStatus.RUNNING*) VISTA_NUEVA=1 ;;
    *KernelWorkerStatus.COMPLETE*)
      [ "$VISTA_NUEVA" = 1 ] && break
      echo "  (COMPLETE sin haber visto correr la versión $VERSION: puede ser la anterior; sigo esperando)"
      VISTA_NUEVA=$((VISTA_NUEVA - 1))
      [ "$VISTA_NUEVA" -le -3 ] && break ;;  # terminó antes del primer sondeo
    *KernelWorkerStatus.ERROR*|*KernelWorkerStatus.CANCEL*)
      echo "Falló. Registro: https://www.kaggle.com/code/$USUARIO/$SLUG (pestaña Logs)" >&2; exit 1 ;;
  esac
done

if [ -n "${SIN_ENTREGAR:-}" ]; then
  echo "Guardado sin entregar: https://www.kaggle.com/code/$USUARIO/$SLUG"; exit 0
fi

echo "== 5/5 Entregando: $DESC"
# La versión ya está corrida: un error de red no tiene que hacer perder la entrega.
for INTENTO in 1 2 3 4 5; do
  if "$K" competitions submit "$COMPE" -k "$USUARIO/$SLUG" -v "$VERSION" -f submission.parquet -m "$DESC"; then
    break
  fi
  if [ "$INTENTO" = 5 ]; then
    echo "No se pudo entregar. Reintentá a mano:" >&2
    echo "  $K competitions submit $COMPE -k $USUARIO/$SLUG -v $VERSION -f submission.parquet -m \"$DESC\"" >&2
    exit 1
  fi
  echo "  Falló la entrega (intento $INTENTO/5); reintento en 60 s"; sleep 60
done
echo
echo "Entregado. Puntaje en unas horas: https://www.kaggle.com/competitions/$COMPE/submissions"
