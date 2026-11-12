> **BORRADOR asistido por IA.** Solo instalación y ejecución. La explicación está en `docs/EXPLICACION_ETAPA2.md`; el MDP en `docs/MDP.md`; el protocolo en `docs/EXPERIMENTS.md`.

# BringPredictorVGC: Etapa 2

IC-6200 Inteligencia Artificial (TEC), Track C. Sergio Zúñiga Castillo · José Ignacio Madriz Ortiz.

## Requisitos

- Python 3.14 (probado con 3.14.6, CPU), Node.js (probado con 24.18) y git.
- Nada más: `setup` descarga el log set y prepara Showdown (`cameronangliss/pokemon-showdown@913da36`, submódulo de vgc-bench, en `../vgc-bench`).

## Uso

Desde la raíz del repo (mejor dentro de un entorno virtual: `python -m venv .venv` y activarlo):

```powershell
pip install -r requirements.txt
python src/main.py setup     # log set (Hugging Face, revisión fijada, 49 775 900 bytes) + vgc-bench/Showdown + npm install
python src/main.py all       # data -> bc -> vhat -> baselines en val -> rl -> tabla. Arranca y apaga Showdown solo
python src/main.py test      # corrida FINAL en test: una sola vez (después de hacer commit de docs/EXPERIMENTS.md)
```

- `python src/main.py all --smoke`: lo mismo con presupuestos mínimos (unos 70 s). **No es un resultado.**
- Pasos sueltos: `python src/main.py data | bc | vhat | val | rl [--beta-H 0] | tablas`.
- Tests: `python -m pytest` (61 tests; los 20 del parser contra replays anotados se saltan hasta que se anoten a mano).
- `main.py test` no corre dos veces: si ya existe `src/output/.test_run_done`, se niega. Con `--force` repite la corrida y lo deja registrado.
- El puerto 8000 tiene que estar libre: `main.py` arranca su propio Showdown y lo apaga al terminar, aunque haya error.

Cada corrida escribe `src/output/<tipo>_<fecha>_s<semilla>/meta.json` (config, commit, semilla, tiempo de pared, hardware), sus CSV y los pesos `.npz`. Las tablas quedan en `src/output/tabla_*.md`.

## Estructura

```
src/
  main.py       punto de entrada único
  config/       config.py + data, bc, vhat, arena, rl, smoke (.yaml)
  scripts/      setup.py, validate_teams.js, fase0/ (spikes históricos de la Fase 0)
  dataPrep/     parser de logs, validador, partición por serie, construcción del dataset
  dataset/      raw/ y clean/ (no versionados), splits/ (versionado: pares congelados)
  utils/        brings y C(o), features, usage stats, regla A8, estadística
  models/       base.py (MetodoBring), bc.py, rl_agent.py, vhat.py   (numpy, gradientes a mano)
  baselines/    random.py, usage.py, tipos.py, planner.py
  sim/          showdown.py (arranque/apagado), jugador.py (BringPlayer), arena.py
  training/     entrenar_bc.py, entrenar_vhat.py, entrenar_rl.py, eval/ (arena, tablas, regla 55 %, diagnósticos)
  output/       salidas de las corridas (solo se versionan los .md); legacy_torch/ = piloto torch, no comparable
  tests/        pytest
docs/           MDP, EXPERIMENTS, AI_USAGE, EXPLICACION_ETAPA2
Research/       planes y survey (los planes todavía describen la versión torch)
```

Variable opcional: `BRINGVGC_SHOWDOWN` para usar otro Showdown en lugar de `../vgc-bench/pokemon-showdown`.
