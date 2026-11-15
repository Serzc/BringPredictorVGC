> **BORRADOR asistido por IA.** Registro para la *Declaración de uso de IA* de la Etapa 2. Los estudiantes deben completarlo, corregirlo y firmar qué verificaron a mano.

# Uso de IA en la Etapa 2

Herramienta: Claude Code (Anthropic), modelo Claude Opus 5.5, ejecutado localmente por Sergio a través de un "Worker Agent". Sesiones: 07-11-2026 (implementación con torch) y 11-11-2026 (reestructuración: numpy + pandas, layout ``src/``).

## Qué se hizo con IA

| Pieza (rutas actuales) | Qué hizo la IA | Qué deben verificar a mano los estudiantes | Estado |
|---|---|---|---|
| `Research/ETAPA2_PLAN.md`, `ETAPA2_PLAN_v2.md` | Borrador del plan | Cada decisión (A1–A9) y que la puedan defender. **Desde el 11-11 el plan no coincide con el código** (dice DeepSets/torch) | revisión pendiente |
| Reestructuración del 11-11-2026 | Layout `src/` (`main.py`, `config/`, `dataPrep/`, `utils/`, `models/`, `baselines/`, `sim/`, `training/`, `tests/`), port de torch a numpy, `main.py` que arranca y apaga Showdown | Que el layout y los comandos les sirvan; reporte en `src/output/RESTRUCTURE_REPORT.md` | pendiente |
| `src/scripts/fase0/` | Spikes de la Fase 0; hallazgo de que `SimpleHeuristicsPlayer` nunca megaevoluciona (`auto_mega`) | Decidir si se adopta `auto_mega` | pendiente |
| `src/dataPrep/parser.py` | Parser del protocolo de Showdown; normalización de formas, apodos e Illusion | **Anotar a mano los 20 replays** de `src/tests/fixtures/annotated_replays.json` (todos en TODO; el test se salta). Hasta entonces el parser **no está verificado** | **pendiente** |
| `src/dataPrep/particion.py`, `construir.py`, `src/config/data.yaml` | Partición por serie, cortes temporales, selección de pares (congelados), filtro de equipos vistos, validador | Revisar los cortes y el criterio de "equipo visto" (conjunto de especies) | pendiente |
| `src/utils/usage.py` | Usage stats con cuentas fraccionales y LOO por batalla | Revisar el suavizado (5 pseudo-cuentas hacia 2/6) | pendiente |
| `src/utils/spreads.py` (A8) | Regla `role_spread` | Decidir la regla (BORRADOR, decisión a defender) | pendiente |
| `src/utils/features.py` | Features hechas a mano (log-tasas de uso, ventaja de tipos, one-hot de especie) | Que cada feature tenga sentido y no filtre el resultado | pendiente |
| `src/models/bc.py`, `src/training/entrenar_bc.py` | Política softmax lineal autorregresiva, pérdida (7) y su gradiente a mano, Adam, early stopping por $\|\mathcal C\|=1$ | Revisar la derivación del gradiente (`gradiente_ponderado`) y los hiperparámetros | pendiente |
| `src/models/vhat.py`, `src/training/entrenar_vhat.py` | $\hat v$ bilineal antisimétrico (eqs. 8-11 del survey, port de la versión torch), verosimilitud (11) y su gradiente; parche del 12-11-2026 | **José**: revisar y hacer commit | pendiente |
| `src/models/base.py`, `src/baselines/` | Clase padre `MetodoBring`; Random, Usage, Tipos, Planificador-$\hat v$ | **José** revisa el Planificador; Tipos: revisar la regla | pendiente |
| `src/sim/` (`showdown.py`, `jugador.py`, `arena.py`), `src/scripts/validate_teams.js` | Arranque/apagado de Showdown, `BringPlayer` con `auto_mega`, tope de turnos, timeouts | Que el bring se respeta (`src/tests/test_arena.py`, pasa) | pendiente |
| `src/models/rl_agent.py`, `src/training/entrenar_rl.py` | REINFORCE + baseline de media + entropía + KL, con gradiente a mano | **José**: revisar el bucle y hacer commit | pendiente |
| Arreglo de la arena colgada (13-11-2026): `src/sim/arena.py`, `src/sim/showdown.py`, `src/training/entrenar_rl.py`, `src/training/eval/evaluar_arena.py`, `src/config/config.py`, `src/config/arena.yaml`, `src/main.py`, tests `test_arena_robusta.py` y `test_resume_y_salto.py` | Diagnóstico del cuelgue de la semilla 1; limpieza de jugadores colgados (/forfeit, /leave, cierre del websocket), vigilante que reinicia Showdown tras 2 fallos seguidos, log de Showdown con fecha por arranque, poda de batallas terminadas, `--resume` (ckpt_resume.npz), `--seed` para `rl`/`val`, salto de trabajo completo y `--force`. Reporte: `src/output/FIX_ARENA_REPORT.md` | Que el diagnóstico les convenza (la causa del **primer** cuelgue no se pudo confirmar); **José** revisa `entrenar_rl.py` (reanudación); con Showdown real la reanudación sigue la misma distribución, no la misma secuencia de batallas | pendiente |
| `src/utils/estadistica.py`, `src/training/eval/` | Wilson, intervalo t, diferencia pareada, regla del 55 %, tablas, diagnósticos | Comprobar los cuantiles t y la regla | pendiente |
| `src/tests/` | Tests unitarios, de leakage, de gradientes (diferencias finitas), de antisimetría y SHA-256 de los pares | Que cubren lo que dicen cubrir | pendiente |
| `docs/*.md`, `README.md` | Borradores de documentación | Reescribir con sus palabras lo que vaya al artículo | pendiente |
| Corridas (`src/output/`) | Piloto torch del 07-11 (`legacy_torch/`, no comparable) y smoke del 11-11 (`smoke_*`) | Son smoke tests; **no citar como resultados** | — |

## Qué NO hizo la IA

- No corrió nada sobre test (ni modelos ni batallas). Solo se seleccionó la lista de pares el 07-11. El paso `data` calcula estadísticas descriptivas de las tres particiones (conteos, fechas, $v$), igual que antes.
- No hizo commits ni editó el artículo (`.tex`) ni los planes de `Research/`.
- No generó los valores esperados del test de 20 replays (se dejan en TODO a propósito).
- No ajustó hiperparámetros mirando resultados: los valores de `src/config/*.yaml` son iniciales.
