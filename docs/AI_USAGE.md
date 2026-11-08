> **BORRADOR asistido por IA.** Registro para la *Declaración de uso de IA* de la Etapa 2. Los estudiantes deben completarlo, corregirlo y firmar qué verificaron a mano.

# Uso de IA en la Etapa 2

Herramienta: Claude Code (Anthropic), modelo Claude Opus 5.5, ejecutado localmente por Sergio a través de un "Worker Agent" (07-11-2026).

## Qué se hizo con IA

| Pieza | Qué hizo la IA | Qué deben verificar a mano los estudiantes | Estado |
|---|---|---|---|
| `Research/ETAPA2_PLAN.md`, `ETAPA2_PLAN_v2.md` | Borrador del plan | Cada decisión (A1–A9) y que la puedan defender | revisión pendiente |
| `scripts/extract_teams.py`, `scripts/phase0_arena_spikes.py` | Spikes de la Fase 0; hallazgo de que `SimpleHeuristicsPlayer` nunca megaevoluciona (`auto_mega`) | Decidir si se adopta `auto_mega` | pendiente |
| `src/bringvgc/parsing/showdown.py` | Parser del protocolo de Showdown, normalización de formas/apodos/Illusion | **Anotar a mano los 20 replays** de `tests/fixtures/annotated_replays.json` (hoy todos en TODO; el test se salta). Hasta entonces el parser **no está verificado** | **pendiente** |
| `data/splits.py`, `configs/data.yaml` | Partición por serie, cortes temporales, selección de pares, filtro de equipos vistos, validador | Revisar los cortes y el criterio de "equipo visto" (conjunto de especies) | pendiente |
| `data/usage.py` | Usage stats con cuentas fraccionales y LOO por batalla | Revisar el suavizado (5 pseudo-cuentas hacia 2/6) | pendiente |
| `features/spreads.py` (A8) | Regla `role_spread` | Decidir la regla (BORRADOR, decisión a defender) | pendiente |
| `models/encoder.py`, `models/policy.py`, `train/bc.py` | DeepSets, política autorregresiva, pérdida (7), early stopping por $|\mathcal C|=1$ | Revisar arquitectura e hiperparámetros | pendiente |
| `models/vhat.py`, `train/vhat.py` | $\hat v$ (8)–(11) y su entrenamiento | **José**: revisar y hacer commit | pendiente |
| `baselines/methods.py` | Random, Usage, Tipos, BC, Planificador-$\hat v$ | **José** revisa el Planificador; Tipos: revisar la regla | pendiente |
| `sim/arena.py`, `sim/validate.py`, `scripts/validate_teams.js` | Arena con `MethodPlayer`, tope de turnos, timeouts, validador de equipos | Revisar que el bring se respeta (Fase 0 lo comprobó con `bring_ok = 1.0`) | pendiente |
| `train/rl.py` | REINFORCE + baseline de media + entropía + KL, checkpoints y reanudación | **José**: revisar el bucle y hacer commit | pendiente |
| `eval/stats.py`, `eval/aggregate.py`, `eval/diagnostics.py` | Wilson, intervalo t, diferencia pareada, regla del 55 % | Comprobar los cuantiles t y la regla | pendiente |
| `tests/` | Tests unitarios y de leakage | Que cubren lo que dicen cubrir | pendiente |
| `docs/MDP.md`, `docs/EXPERIMENTS.md`, `README.md`, `docs/EXPLICACION_ETAPA2.md` | Borradores de documentación | Reescribir con sus palabras lo que vaya al artículo | pendiente |
| Corridas piloto (`results/`, 07-11-2026) | Las lanzó la IA: 1 semilla, solo val | Son smoke tests; **no citar como resultados** | — |

## Qué NO hizo la IA

- No corrió nada sobre test (ni modelos ni batallas). Solo se seleccionó la lista de pares.
- No hizo commits ni editó el artículo (`.tex`).
- No generó los valores esperados del test de 20 replays (se dejan en TODO a propósito).
