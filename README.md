> **BORRADOR asistido por IA.** Solo instalación y ejecución. La explicación está en `docs/EXPLICACION_ETAPA2.md`; el MDP en `docs/MDP.md`; el protocolo en `docs/EXPERIMENTS.md`.

# BringPredictorVGC: Etapa 2

IC-6200 Inteligencia Artificial (TEC), Track C. Sergio Zúñiga Castillo · José Ignacio Madriz Ortiz.

## Requisitos

- Python 3.14 (probado con 3.14.6, CPU), Node.js (probado con 24.18), git.
- Showdown local: `cameronangliss/pokemon-showdown` en el commit `913da36` (submódulo de vgc-bench).
- El log set Reg M-A de VGC-Bench (`logs_gen9championsvgc2026regma.json`, ~50 MB). **No está en el repo.**

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Showdown (en otra terminal, se deja corriendo):

```powershell
cd ..\vgc-bench\pokemon-showdown
node pokemon-showdown start --no-security      # puerto 8000
```

Rutas por defecto (se pueden cambiar con variables de entorno):

| Qué | Por defecto | Variable |
|---|---|---|
| Log set | `../Research Proyecto/data/logs_gen9championsvgc2026regma.json` (en `configs/data.yaml`) | `BRINGVGC_LOGS` |
| Showdown | `../vgc-bench/pokemon-showdown` | `BRINGVGC_SHOWDOWN` |

## Ejecución

Todo se corre desde la raíz del repo. `scripts/run_all.ps1` tiene una línea por tabla:

```powershell
.\scripts\run_all.ps1 -Stage data                     # Fase 1: dataset, splits, tests
.\scripts\run_all.ps1 -Stage offline -Bc0 ... -Bc1 ... -Bc2 ...   # Fase 2: rho_BC x3, v-hat
.\scripts\run_all.ps1 -Stage val -Bc0 ... -Bc1 ... -Bc2 ... -Vhat ...   # Fase 3: baselines en val
.\scripts\run_all.ps1 -Stage rl                       # Fase 4: RL (editar bc_ckpt en configs/rl.yaml)
.\scripts\run_all.ps1 -Stage test ...                 # Fase 5: UNA vez, tras commitear docs/EXPERIMENTS.md
```

Comandos sueltos (con `$env:PYTHONPATH="src"`):

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m bringvgc.train.bc --seed 0
.\.venv\Scripts\python.exe -m bringvgc.train.vhat --seed 0
.\.venv\Scripts\python.exe -m bringvgc.eval.run_arena --method tipos --pairs val
.\.venv\Scripts\python.exe -m bringvgc.train.rl --seed 0 [--resume results\rl_<id>] [--max-wall-min 120]
.\.venv\Scripts\python.exe -m bringvgc.eval.aggregate --pairs val
```

Cada corrida escribe `results/<run_id>/meta.json` (config, commit, semilla, tiempo de pared, hardware) y sus CSV.

## Estructura

```
configs/      data, bc, vhat, arena, rl (+ rl_smoke)
splits/       battles.csv (partición), {train,val,test}_pairs.json, pairs_report.json  (versionado)
src/bringvgc/ parsing, data, features, models, baselines, sim, train, eval
scripts/      build_dataset.py, run_all.ps1, validate_teams.js, phase0_arena_spikes.py, extract_teams.py
tests/        pytest (incluye tests de leakage y el test de 20 replays, pendiente de anotar)
docs/         MDP, EXPERIMENTS, AI_USAGE, EXPLICACION_ETAPA2
results/      salidas de las corridas (no se versionan los checkpoints .pt)
```
