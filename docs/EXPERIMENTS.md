> **BORRADOR asistido por IA.** Preregistro informal (plan v2 §9). Revisarlo, ajustarlo y **hacer commit antes de correr test**. El commit con fecha es la evidencia del preregistro.

# Experimentos y regla de decisión

## 1. Pregunta

¿Una política de RL que ve los dos equipos completos elige un bring que gana **≥ 55 %** contra el baseline Usage (estocástico), en **1000 batallas por par** (500 por orientación, con lados intercambiados) y con la misma $\pi_b$ en ambos lados?

## 2. Datos y particiones (fijados el 07-11-2026)

- Corte temporal por serie Bo3 (`src/config/data.yaml`): train < 2026-05-25 16:51 UTC ≤ val < 2026-06-05 01:50 UTC ≤ test.
- Pares congelados en `src/dataset/splits/` (SHA-256 comprobado contra `d3b1d7d` en `src/tests/test_data.py`; `setup` y `data` nunca los regeneran): 200 de train (RL), **10 de val**, **20 de test**. En val y test, ningún equipo (conjunto de especies) aparece en train, y test tampoco comparte equipos con val. Solo se eligen pares cuyos dos equipos pasan el validador de Showdown.
- **Test no se toca hasta la Fase 5.** El código lo impide salvo en `python src/main.py test` (`allow_test=True` en `src/dataPrep/cargar.py` y `particion.py`). `main.py test` se niega a correr dos veces (marca `src/output/.test_run_done`) salvo con `--force`, que queda registrado.
- Nota: la lista de test se regeneró una vez el 07-11-2026 (al añadir el validador). Sobre test no se había corrido nada.

## 3. Métodos

| Método | Semillas | Notas |
|---|---|---|
| Agente RL (softmax lineal, REINFORCE + KL a $\rho_{BC}$, $\beta_H=0.01$) | 3 | checkpoint elegido por win rate en val |
| BC ($\rho_{BC}$ lineal, greedy) | 3 | ablación: ¿aportó el RL? |
| Random | 3 (muestreo) | piso |
| Usage-argmax | determinista | baseline de literatura del survey |
| Tipos | determinista | heurística de dominio |
| Planificador-$\hat v$ | determinista ($\hat v$ = modelo bilineal antisimétrico del survey, eqs. 8-11, 1 semilla) | búsqueda con fitness aprendida (A3) |

Rival en todos los casos: Usage estocástico. Evaluación del agente y de BC: greedy (argmax autorregresivo).

## 4. Estadística

- **Por par:** win rate con IC de Wilson al 95 % (≈ ±0,03 con $n=1000$). Los empates cuentan como no victoria y se reportan.
- **Agregado:** el win rate de cada par se promedia sobre las semillas; sobre los **20 pares**: media e **intervalo t al 95 %** (unidad = par). Al lado: desviación estándar entre semillas.
- **Comparaciones:** diferencia pareada por par (agente − BC, agente − Usage-argmax, agente − Tipos, agente − Random): media ± error estándar sobre los 20 pares. Cada una se guarda en `src/output/tabla_test_vs_<referencia>.md`.
- **Agente − Tipos y agente − Random (añadidas el 14-11-2026, antes de correr test):** en val, Random (0,553) quedó por encima de Usage-argmax (0,467) y Tipos (0,594) por encima del agente (0,580). Con la misma $\pi_b$ fija, ganarle a Usage no basta para decir que el agente elige buenos brings. Estas dos comparaciones se reportan como secundarias y descriptivas: el signo de la diferencia y si su intervalo (media ± 1,96·EE) incluye el 0. No cambian la regla del §5. Si el agente no supera a Tipos, se declara como resultado y como posible explotación de $\pi_b$ (riesgo R4).
- **Mínimo de pares:** la regla del §5 solo se aplica con los 20 pares completos. Si un par falla en la arena, se reporta, se corre otra vez y no se reemplaza. Si la desviación estándar entre pares es 0 (intervalo degenerado) y la media supera 0,55, se reporta "no concluyente" en vez de "supera" (`decision()` en `tablas.py`).
- Código: `src/utils/estadistica.py`, `src/training/eval/tablas.py` (la regla del 55 % está en `decision()`).

## 5. Regla de decisión (preregistrada)

1. **"Supera el 55 %"** solo si el límite inferior del intervalo t sobre los 20 pares es **> 0,55**.
2. Si la media supera 0,55 pero el intervalo no: **"compatible, no concluyente"**.
3. Si no: **se rechaza**.

La regla se aplica una vez, a la corrida de test de la Fase 5, sin cambiar checkpoints ni hiperparámetros después de verla. Si se mira test antes, se reporta.

## 6. Diagnósticos sobre logs de test (una vez, Fase 5)

Lead-pair exact match (piso 6,67 %) y full-bring exact match permisivo ($\hat b \in \mathcal C(o)$, piso 1,11 %), media ± desviación estándar sobre las 3 semillas de $\rho_{BC}$ (`src/training/eval/diagnosticos.py`).

## 7. Presupuesto y $T$ medido

$T$ medido en el piloto de RL con $\rho_{BC}$ en el bucle (07-11-2026, i5-10400F, 6 núcleos, 16 GB, sin GPU, un proceso): **47,9 episodios/s** de entrenamiento y 51,7 batallas/s en la arena. Con eso, el presupuesto del plan §9.3 (≈ 533 000 batallas) son ≈ 3,1 h. Detalle en `src/output/legacy_torch/PILOTO_ETAPA2.md`. **Ese `T` es de la era torch**: con los modelos lineales hay que volver a medirlo (el smoke del 11-11-2026 dio unas 25 batallas/s con bloques muy cortos, que no sirve como medida).

## 8. Qué NO cuenta como resultado

Las corridas piloto (1 semilla, val, 07-11-2026, torch, `src/output/legacy_torch/`) y las corridas `smoke_*` (`python src/main.py all --smoke`) son **smoke tests**: sirven para comprobar que el pipeline funciona y para medir $T$. No se citan en el artículo como resultados.
