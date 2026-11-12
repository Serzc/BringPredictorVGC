> **BORRADOR asistido por IA (Worker Agent, Claude Opus 5.5, 11-11-2026).** Reporte de la reestructuración de BringPredictorVGC a numpy + pandas. No se hizo ningún commit ni push.

# Reestructuración de BringPredictorVGC (numpy + pandas, sin torch)

- Rama: `restructure-numpy`, creada desde `implementation` (`d3b1d7d test1`).
- Todo está en el índice (`git mv`) o sin seguimiento, listo para que lo revisen y hagan commit ustedes.

## Fase 1: layout

**Movimientos con `git mv`** (el historial se conserva):

| Antes | Ahora |
|---|---|
| `configs/*.yaml` | `src/config/{data,bc,vhat,arena,rl}.yaml`; `rl_smoke.yaml` → `smoke.yaml` (ahora pisa todos los pasos) |
| `src/bringvgc/config.py` | `src/config/config.py` |
| `src/bringvgc/parsing/showdown.py` | `src/dataPrep/parser.py` |
| `src/bringvgc/sim/validate.py` | `src/dataPrep/validar.py` |
| `src/bringvgc/data/splits.py` | `src/dataPrep/particion.py` |
| `scripts/build_dataset.py` | `src/dataPrep/construir.py` |
| `src/bringvgc/data/dataset.py` | `src/dataPrep/cargar.py` |
| `src/bringvgc/brings.py`, `data/usage.py`, `data/vocab.py`, `features/spreads.py`, `features/encode.py`, `eval/stats.py` | `src/utils/{brings,usage,vocab,spreads,features,estadistica}.py` |
| `src/bringvgc/baselines/methods.py` | `src/models/base.py` (clase padre `MetodoBring`); los baselines se separaron en `src/baselines/{random,usage,tipos,planner}.py` (nuevos) |
| `src/bringvgc/models/policy.py`, `models/vhat.py` | `src/models/bc.py`, `src/models/vhat.py` (reescritos en numpy); `src/models/rl_agent.py` (nuevo) |
| `src/bringvgc/sim/arena.py` | `src/sim/arena.py`; además `src/sim/jugador.py` y `src/sim/showdown.py` (nuevos) |
| `src/bringvgc/train/{bc,vhat,rl,common}.py` | `src/training/{entrenar_bc,entrenar_vhat,entrenar_rl,ejemplos}.py` |
| `src/bringvgc/eval/{run_arena,aggregate,diagnostics}.py` | `src/training/eval/{evaluar_arena,tablas,diagnosticos}.py` |
| `scripts/validate_teams.js` | `src/scripts/validate_teams.js` |
| `scripts/{phase0_arena_spikes,extract_teams,smoke_min}.py`, `run_all.ps1`, `phase3_val_baselines.ps1` | `src/scripts/fase0/` (históricos; los `.ps1` se renombraron a `*_torch.ps1`) |
| `splits/*` | `src/dataset/splits/` (sigue versionado) |
| `tests/*` (con `fixtures/`) | `src/tests/` |
| `results/*` (lo versionado) | `src/output/legacy_torch/` |
| `src/bringvgc/**/__init__.py` | `__init__.py` de los paquetes nuevos |

**Nuevos:**
- `src/main.py`, `src/scripts/setup.py`
- `src/tests/test_gradients.py`, `src/tests/test_main.py`
- `src/output/legacy_torch/README.md`, `src/scripts/fase0/README.md`

**Quitado con `git rm`:** solo `src/bringvgc/models/encoder.py`, el DeepSets en torch, descartado por la decisión 1. Se puede recuperar desde `d3b1d7d`.

**Movidos con `Move-Item` (estaban ignorados; nada se borró):**
- los checkpoints `.pt`, las carpetas `results/arena_*`, `results/phase0/` y `results/rl_20261107-150017_s0/` → `src/output/legacy_torch/`;
- `data/processed/` y `data/sample_teams.txt` → `src/output/legacy_torch/data/`.

**Carpetas vacías eliminadas** (no tenían ningún archivo): `configs/`, `scripts/`, `splits/`, `data/` y `results/`.

**Raíz final:** `README.md`, `requirements.txt`, `.gitignore`, `.gitattributes`, `pyproject.toml`, `docs/`, `Research/`, `_IAR_2026S2_PY01_TrackRL.pdf`, `src/`, `.venv/`. Quedan además `tests/` y `src/bringvgc/`, con solo `__pycache__` (ver la sección de borrados propuestos).

**Pares congelados:**
- El SHA-256 de `train/val/test_pairs.json` coincide con `d3b1d7d`. Lo comprueba `test_frozen_pairs_match_d3b1d7d`.
- `setup` y `data` nunca regeneran los pares. `data` además verifica que `battles.csv` coincide con la partición que da el parser actual (6 905 batallas: coincide).
- Solo `src/dataset/raw/` y `clean/` están en `.gitignore`.

**Lo que se versiona en `src/output/`:** solo los `.md`, y lo que ya estaba versionado en `legacy_torch/`. Las corridas `smoke_*` y la marca `.test_run_done` se ignoran.

## Fase 2: modelos (numpy, gradientes a mano)

| Pieza | Qué es ahora |
|---|---|
| Features (`utils/features.py`) | Por Pokémon: log P(líder), log P(reserva), log P(fuera) y uso normalizado (train, LOO en BC); ventaja ofensiva y defensiva de tipos contra los 6 rivales; one-hot de especie (252 en el vocabulario de train) → F = 258 |
| $\rho_{BC}$ (`models/bc.py`) | Softmax lineal autorregresiva: 15 pares líderes y luego 6 pares de reservas entre los 4 que quedan (con máscara opcional por Pokémon). Pérdida (7) de etiquetas parciales, L2, Adam escrito a mano |
| Agente RL (`models/rl_agent.py`) | Hereda de la política de BC. Minimiza REINFORCE con baseline (`retorno_medio`) − β_H·entropía + β_KL·KL(π‖ρ_BC), con recorte de la norma del gradiente. Los tres términos usan el mismo `gradiente_ponderado()` |
| $\hat v$ (`models/vhat.py`) | Regresión logística sin sesgo sobre "features de b1 − features de b2" (líderes y reservas por separado), más la ventaja de tipos entre los dos brings. Es antisimétrico por construcción. Verosimilitud marginalizada (11) |
| Clase padre `MetodoBring` (`models/base.py`) | Random, Usage, Usage-argmax, Tipos, Planificador, BC y el agente heredan de ella. La arena los trata igual (`probs()` / `elegir()`) |

**Tests nuevos:**
- `test_gradients.py`: diferencias finitas centrales para la pérdida (7), la verosimilitud (11) y la actualización de RL, también con el paso 2 enmascarado. Error relativo menor que 1e-5.
- `test_vhat_antisymmetry_exact`: $v(b_1,b_2)+v(b_2,b_1)=1$ con atol 1e-12.
- `test_all_methods_inherit_metodobring...`: todos los métodos heredan de `MetodoBring` y devuelven distribuciones válidas.

**Decisión documentada** en `docs/EXPLICACION_ETAPA2.md` §3: por qué torch → lineal, las alternativas descartadas y lo que se pierde (sinergias entre Pokémon, objeto y habilidad, capacidad).

## Fase 3: runner y docs

**`src/main.py`** es el único punto de entrada: `setup | all | test` y un comando por paso (`data | bc | vhat | val | rl [--beta-H] | tablas`), con `--smoke`.

**Showdown** (`sim/showdown.py`):
- comprueba que el puerto 8000 esté libre;
- espera hasta 60 s a que arranque;
- lo apaga con `taskkill /PID … /T /F` en un `finally`.

En `--smoke` hay un tope de tiempo total de 1 800 s: la arena lo revisa antes de cada bloque, y un vigilante duro a los 1 920 s apaga Showdown y termina el proceso.

**`setup`:**
- Reutiliza el JSON local si pesa 49 775 900 bytes.
- Si no, descarga de Hugging Face en la revisión `74f2605…` con `urllib` y comprueba el tamaño.
- Reutiliza `../vgc-bench` si existe; si no, lo clona con el submódulo. Hace checkout de Showdown a `913da36` solo si no hay cambios locales.
- Corre `npm install` si falta `node_modules`.

**`main.py test`:**
- Escribe la marca `src/output/.test_run_done` **antes** de empezar.
- Se niega a correr si la marca existe; `--force` lo repite y lo registra. Lo cubre `test_main.py`.
- Usa 1000 batallas por par, Tabla 1 contra Usage-argmax y contra BC, y los diagnósticos de BC en test.
- **No se ejecutó.**

**`meta.json`:** cada corrida escribe el suyo (config, commit, semilla, smoke sí/no, wall time, hardware y versiones de numpy, pandas y poke-env).

**Docs actualizados:**
- `README.md`: los 4 comandos.
- `docs/EXPLICACION_ETAPA2.md`: reescrito, más corto, con el nuevo layout, la nota de decisión torch → lineal y el glosario (con términos nuevos).
- `docs/MDP.md`, `docs/EXPERIMENTS.md`: rutas y descripción de la política.
- `docs/AI_USAGE.md`: fila nueva para la sesión del 11-11.

**Comentarios:**
- Todos en español, con encabezados de sección en los bloques largos y nombres explícitos para las variables de entrenamiento (`retorno_medio`, `ventaja`, `peso_lider`…).
- En `parser.py` y `particion.py` (portados casi sin cambios) solo se agregaron comentarios de sección.

## Comprobaciones

### pytest (`python -m pytest`, `.venv` del dueño, Python 3.14.6)

**42 passed, 20 skipped.**
- Los 20 skips son `test_parser_annotated.py`: "anotacion a mano pendiente (estudiantes)". Las fixtures siguen en TODO y no se generaron con el parser.
- Los tests de leakage corren y pasan sobre los datos reales:
  - ninguna serie aparece en dos particiones;
  - ningún equipo de val o test aparece en train;
  - los pares de train salen de train;
  - las usage stats usan solo IDs de train;
  - test está bloqueado.
- `test_arena.py` corre: arranca su propio Showdown y comprueba que el servidor respeta el bring enviado.

### Sin torch

- `grep "import torch|from torch|torch\."` en `src/`: sin coincidencias.
- **Venv nuevo y temporal** (`%TEMP%\bringvgc_torchfree_venv`, `C:\Python314\python.exe` 3.14):
  - `pip install -r requirements.txt` no instala torch (`find_spec('torch')` = False);
  - `pytest`: **41 passed, 20 skipped** (antes de agregar `test_main.py`), con el test de arena incluido.
- No se desinstaló nada del `.venv` del dueño: torch 2.15.0 sigue ahí, sin uso.

### Smoke test: `python src/main.py all --smoke`, exit 0, 66 s

**SMOKE TEST: 2 pares de val × 8 batallas, 1 semilla. No son resultados.**

| Paso | Número (smoke) |
|---|---|
| data | 6 902/6 905 batallas válidas (99,96 %); particiones 4 831 / 1 036 / 1 035 |
| BC (3 épocas) | NLL $|\mathcal C|=1$ en val 4,225 (azar 4,50); acierto del par líder 16,5 % (piso 6,67 %); bring permisivo 5,2 % (piso 1,11 %) |
| $\hat v$ (2 épocas) | team: acc 0,552 / AUC 0,571; obs: acc 0,547 / AUC 0,560 |
| Showdown | arrancó en 6,0 s y se apagó al final |
| Arena (16 batallas por método) | Usage vs Usage 0,44; Usage-argmax espejo 0,31; Random 0,56; Usage-argmax 0,56; Tipos 0,63; Planificador 0,63; BC 0,63; agente 0,69 |
| RL (64 episodios, 2 actualizaciones) | entropía 4,22; KL a $\rho_{BC}$ ≈ 0; T de entrenamiento 40 episodios/s; 0 timeouts |

Tabla del smoke: `src/output/smoke_tabla_val_vs_usage_argmax.md` (ignorada por git). Con 2 pares y 8 batallas, ninguna de estas cifras dice nada.

## Instalaciones

| Qué | Por qué | Dónde |
|---|---|---|
| pandas 3.1.1 (+ python-dateutil 4.0.0, tzdata 2026.6) | Nueva dependencia pedida (tablas y CSV) | `.venv` del proyecto |
| Venv temporal con `requirements.txt` (fork de poke-env `379a046`, numpy 2.5.4, pandas 3.1.1, PyYAML 6.0.3, pytest 9.2.0 y sus dependencias) | Probar que el proyecto funciona sin torch | `%TEMP%\bringvgc_torchfree_venv` (fuera del repo) |
| Desinstalaciones | ninguna | — |

**Descarga:** el primer `setup` bajó el log set de Hugging Face (49 775 900 bytes, tamaño verificado) a `src/dataset/raw/`, porque la ruta `local_copy` tenía un `../` de más. Ya está corregida. El segundo `setup` dijo "log set ya presente", y el parseo y la partición coinciden con los de la copia local.

## Problemas abiertos

1. **Los planes de `Research/` (`ETAPA2_PLAN*.md`) ya no coinciden con el código:** dicen DeepSets/torch. No se editaron, porque estaba fuera del alcance. Hay que actualizarlos o declarar el cambio.
2. **El agente RL lineal podría no pasar la regla del 55 %.** Si pasa eso, se reporta como hallazgo (capacidad del modelo frente al problema). **No se ajusta para que pase.**
3. **El hallazgo del piloto torch sigue sin investigar:** Random y Tipos le ganaron a Usage. Hay que repetirlo con el código nuevo y corridas reales.
4. **Parser sin verificar:** faltan los 20 replays anotados a mano.
5. **Se quitó `--resume` del RL** (el piloto torch lo tenía). Si una corrida larga se corta, empieza de nuevo. Con un T de unos 40–50 episodios/s, cada semilla de 50 000 episodios lleva del orden de 20–30 min más las evaluaciones. Ese T hay que medirlo en serio.
6. **Ruido al apagar:** poke-env imprime una traza `ConnectionClosedError` cuando se mata Showdown al final. Es cosmético: el exit code es 0 y los resultados ya están escritos.
7. **Intervalo t degenerado:** con desviación 0 entre pares, el intervalo colapsa a un punto. En el smoke, BC y el Planificador (2 pares con el mismo win rate) salen como "supera el 55 %". Con 10 o 20 pares es improbable, pero la regla es mecánica. Revisen si quieren exigir n ≥ cierto número de pares.
8. **Barrido β_H = 0** del plan: no está dentro de `all`; se corre con `python src/main.py rl --beta-H 0`. Las 3 semillas de BC y de RL las corre `all` (sin `--smoke`), lo que no se hizo (estaba fuera del alcance).
9. **Las estadísticas descriptivas del paso `data`** incluyen la partición test (conteos, fechas, % de $v$), igual que en `d3b1d7d`. No hay modelos ni batallas en test.

## Archivos de José (revisar y hacer commit él)

- `src/models/vhat.py`
- `src/training/entrenar_vhat.py`
- `src/baselines/planner.py`
- `src/models/rl_agent.py`
- `src/training/entrenar_rl.py`

## Borrados propuestos (no se hicieron)

| Archivo / carpeta | Qué es | Por qué borrarlo |
|---|---|---|
| `tests/` (raíz) | Solo `__pycache__/*.pyc` y una carpeta `fixtures/` vacía | Generado; el contenido ya está en `src/tests/` |
| `src/bringvgc/` | Solo `__pycache__/*.pyc` | Generado; el paquete ya no existe |
| `src/scripts/fase0/run_all_torch.ps1`, `phase3_val_baselines_torch.ps1` (con seguimiento) | Runners de la versión torch | Ya no funcionan; los reemplaza `main.py` (se pueden recuperar de `d3b1d7d`) |
| `%TEMP%\bringvgc_torchfree_venv` | El venv temporal de la prueba sin torch | Ya cumplió su función |

## Comandos de commit (para el dueño)

Sergio hace commit de todo menos los archivos de José. Después José revisa y hace commit de los suyos. El primer commit no corre solo (le faltan $\hat v$, el Planificador y el RL), así que hagan push de los dos juntos.

```powershell
git switch restructure-numpy
git add .gitignore pyproject.toml requirements.txt README.md docs src
git restore --staged src/models/vhat.py src/training/entrenar_vhat.py src/baselines/planner.py src/models/rl_agent.py src/training/entrenar_rl.py
git commit -m "Reestructura a src/ con numpy y pandas (sin torch), main.py unico y tests de gradientes"

# José, después de revisar:
git add src/models/vhat.py src/training/entrenar_vhat.py src/baselines/planner.py src/models/rl_agent.py src/training/entrenar_rl.py
git commit -m "v-hat logistico antisimetrico, Planificador y agente RL en numpy"
git push -u origin restructure-numpy
```
