# IA de Objetos Vivos — ARC Prize 2026 (ARC-AGI-3)

Código de Franco González ([Demoliyer](https://github.com/Demoliyer)) para la competencia
ARC Prize 2026 – ARC-AGI-3 en Kaggle. Licencia **MIT-0** (ver `LICENSE`).

La idea: una creencia sobre el juego se usa solo si explica todo lo observado. Se aplica de
dos formas:

1. **Motor de objetos vivos** (`vivos/`): una población de reglas chicas y legibles, cada una
   verificada y reparada contra la evidencia (percepción con máscara de relojes, colonia de
   reglas, grafo de estados, novedad IW(1), presupuesto de pasos y un modelo del mundo
   simulable).
2. **Injerto en un agente LLM** (`vivos/simulador.py`, `kaggle/injerto_vivos.py`): sobre el
   harness público de Daniel Franzen, el modelo escribe su hipótesis como simulador
   `step(grid, action)`. Herramientas sin costo de acciones la verifican contra cada
   transición observada (`check_model`, auto-chequeo después de cada snippet) y planifican
   dentro de ella (`plan_with_model`).

El paper (borrador) está en `docs/paper/borrador.md`; el registro completo de mediciones y
decisiones, en `docs/mediciones.md`.

## Qué hay

| Ruta | Contenido |
|---|---|
| `vivos/` | Motor: `percepcion.py`, `reglas.py`, `mundo.py`, `grafo.py`, `agente.py`, `simulador.py` |
| `kaggle/preparar_franzen.py` | Prepara el notebook de Franzen: guardado rápido y opciones `--vivos` / `--vivos-auto` |
| `kaggle/injerto_vivos.py` | Injerta el simulador verificado en el sandbox del harness |
| `kaggle/experimento.sh` | Un experimento = una entrega; exige las huellas del notebook antes de subir |
| `kaggle/huella_notebook.py`, `identidad_notebook.py`, `guarda_identidad.py`, `identidades_candidatos.json` | Identidad de cada notebook entregado |
| `kaggle/recuperar_versiones.py`, `fuentes_salida.py`, `analizar_salida.py` | Recuperación de versiones de Kaggle y análisis de salidas |
| `experimentos/reglas/` | Experimento: reglas escritas por el LLM frente a reglas aprendidas por el motor |
| `evaluar_mundo.py`, `comparar_borde.py`, `grabar_trayectorias.py` | Mediciones del modelo del mundo y de la máscara de relojes |
| `jugar_arc.py`, `bench_arc.py`, `medir_publicos.sh`, `empaquetar_kaggle.py` | Agente propio sobre los juegos públicos y su paquete para Kaggle |

## Versiones entregadas

Cada configuración se identifica por la huella de su notebook preparado
(`kaggle/huella_notebook.py`):

| Configuración | Huella |
|---|---|
| Harness de Franzen sin cambios | 5c66785823b6 |
| `--vivos` (herramientas descritas en el prompt) | be36c273659a |
| `--vivos-auto` (auto-chequeo) | ff42d7f18a5a |
| `vivos-v3` | 5e0b90762a34 |
| `vivos-v4s` original (no entregada) | 413a0b289a64 |
| `vivos-v4s` revisada (rama `v4s`) | 9d748dcf6c1a |
| F: v4s + aviso de hipótesis (rama `f2b`; entregas F-2b 31,52 y F-2b-r), `--vivos-auto` | de260c82176f |
| M: F + malla de objetos vivos `engine_step` (rama `motor`; entrega M-1 24,08), `--vivos-auto --motor` | 97652a37e9be |

Origen: notebook de Franzen con huella 4056fbfc69bb. Los puntajes están en
`docs/mediciones.md`.

## Uso

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-arc.txt
.venv/bin/python -m unittest discover -s tests

# Notebook de Franzen con el injerto (el original se baja de Kaggle: dfranzen/arc-agi-3-milestone-2-solution)
python3 kaggle/preparar_franzen.py original.ipynb salida.ipynb --vivos-auto
python3 kaggle/huella_notebook.py salida.ipynb     # ff42d7f18a5a
```

Las pruebas que necesitan el harness o el notebook de Franzen se saltean si no están presentes
(`VIVOS_HARNESS=<ruta a ARC3-Inference>`, `kaggle/franzen_nb/original.ipynb`).

## Créditos

- **Daniel Franzen**, *ARC-AGI-3 solution (Milestone 2)*: notebook y harness sobre los que se
  injerta este código (<https://github.com/da-fr/arc-agi-3-solution>). Su código no está
  incluido en este repositorio; `kaggle/preparar_franzen.py` lo modifica al prepararlo.
- **Tufa Labs** (Jeroen Cottaar), *Duck harness*, base del harness de Franzen
  (<https://github.com/Tufalabs/duck-harness>).
- **ARC Prize Foundation**: juegos, kit oficial y competencia. Los juegos no se incluyen.
- Revisiones externas con modelos de IA (Gemini "Argon", GPT "Astra") y desarrollo asistido
  por Claude (Anthropic); las decisiones, experimentos y entregas los dirigió el autor.
