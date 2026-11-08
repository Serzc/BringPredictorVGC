> **BORRADOR asistido por IA (07-11-2026).** Explicación en lenguaje simple de lo que se implementó para la Etapa 2, siguiendo `Research/ETAPA2_PLAN_v2.md`. No es texto para el artículo: es para que Sergio y José entiendan, revisen y defiendan cada pieza. Lo marcado **"decisión a defender"** todavía no lo han decidido ustedes.

# Etapa 2 explicada: qué hace cada parte, por qué y cómo correrla

## 0. La idea en un párrafo

En VGC, antes de cada batalla los dos jugadores ven los 6 Pokémon del rival (Team Preview) y eligen **4 de sus 6**: dos que empiezan en el campo (los **líderes**) y dos que esperan atrás (las **reservas**). A esa elección le llamamos **bring**. Hay exactamente **90** brings posibles. El proyecto pregunta: *¿puede un agente de aprendizaje por refuerzo aprender a elegir el bring de forma que gane al menos el 55 % de las veces contra alguien que elige "como elige la mayoría de jugadores" (el baseline Usage)?* Para medirlo de forma justa, las batallas las juega **el mismo bot en los dos lados** (la política de batalla $\pi_b$); lo único que cambia es quién eligió el bring.

## 1. El recorrido completo

```
logs humanos (6 905 batallas)
   │  1. parser: quién lideró, quién entró, quién ganó
   ▼
registros ──► 2. partición por tiempo (train / val / test), por serie Bo3
   │
   ├─► estadísticas de uso (solo train) ──► baseline Usage = el rival
   ├─► 3a. ρ_BC: imita a los humanos (pérdida 7)
   ├─► 3b. v̂: predice quién gana dado un bring contra otro (pérdida 11)
   │
   ▼
4. arena: Showdown local + el mismo bot en ambos lados
   │     cada método solo cambia el Team Preview
   ▼
5. agente RL: parte de ρ_BC y aprende jugando batallas reales (REINFORCE)
   ▼
6. evaluación: win rate por par, intervalo t sobre pares, regla del 55 %
```

## 2. Cada parte, en simple

### 2.1 El parser (`src/bringvgc/parsing/showdown.py`)

**Qué hace.** Lee el texto de cada batalla (el "protocolo" de Showdown, líneas como `|switch|p1a: Rotom|Rotom-Wash, L50|100/100`) y saca:

- las dos hojas de equipo (`|showteam|`), es decir, las OTS;
- los **líderes**: los dos primeros Pokémon que entran de cada lado después de `|start`;
- las **reservas reveladas**: cualquier otro Pokémon que entró al campo después;
- el ganador, el número de turnos y la serie Bo3 a la que pertenece.

**El problema de los nombres.** En la batalla, un Pokémon puede aparecer con otro nombre: un apodo ("Rotom" en vez de "Rotom-Wash"), una forma Mega ("Floette-Mega" cuando en la hoja dice "Floette-Eternal"), o disfrazado con **Illusion** (Zoroark entra con la apariencia de otro). La solución: comparar por **número de Pokédex**. Como la *Species Clause* prohíbe repetir número en un equipo, el número identifica sin ambigüedad la posición en la hoja. Con Illusion, la línea `|replace|` dice quién era en realidad, y el parser corrige la entrada anterior de ese slot.

**Por qué importa.** Si el parser se equivoca, todo lo demás aprende de datos falsos.

**Cómo se comprobó.** Los porcentajes que salen coinciden con los que midieron en la Etapa 1: 84,1 % de brings completos ($v=2$), 12,0 % con una reserva revelada ($v=1$), 3,9 % sin ninguna ($v=0$), y 1 245 series que suman 2 476 batallas. **Esto no es una verificación**: la verificación real es el test de 20 replays anotados a mano (`tests/fixtures/annotated_replays.json`). Los 20 `battle_id` ya están elegidos (de train), pero los valores esperados están en "TODO". **Ustedes tienen que llenarlos mirando cada log.** Hasta entonces, el test se salta y el parser no se considera verificado.

**Exclusiones.** Se excluyen 3 batallas (0,04 %) que se jugaron con 5 Pokémon en vez de 4 (`|teamsize|p1|5`). El survey hablaba de "2 registros inconsistentes": el número cambió porque ahora la regla es explícita. Hay que corregirlo en el artículo.

### 2.2 El conjunto de brings consistentes $\mathcal C(o)$ (`src/bringvgc/brings.py`)

Si un jugador ganó rápido, puede que una o sus dos reservas nunca entraran. Entonces no sabemos su bring exacto, pero sí sabemos **cuáles son posibles**: los que tienen esos líderes y contienen las reservas que sí se vieron. Son 1, 3 o 6 brings según cuántas reservas se vieron (2, 1 o 0). En vez de tirar esas batallas (que no sería neutral: **los ganadores revelan menos**, 73,6 % contra 94,7 % de los perdedores), los modelos aprenden "repartiendo" la probabilidad entre los brings posibles.

### 2.3 La partición train / val / test (`src/bringvgc/data/splits.py`, `configs/data.yaml`)

- **Por tiempo:** train son las batallas más viejas (hasta el 25-05-2026), val las del medio (hasta el 05-06-2026) y test las más nuevas. Así se simula "entrenar con el pasado y probar en el futuro".
- **Por serie Bo3:** las tres partidas de una serie se quedan juntas en la misma partición. Si una partida de la serie estuviera en train y otra en test, el modelo habría visto ese mismo enfrentamiento (**leakage**, −15 puntos).
- **Pares para simular:** 200 de train (para el RL), 10 de val y 20 de test. En val y test **ningún equipo aparece en train** (comparando el conjunto de 6 especies), y test tampoco comparte equipos con val. Los 10 y 20 pares salieron sin tener que relajar el criterio (había 386 y 462 candidatos con 0 equipos vistos).
- **Validador:** 80 batallas tienen algún equipo que el validador de Showdown rechaza (por ejemplo, "Delphox can't learn Eruption"). Si uno de esos llega a la arena, el reto se rechaza y el bloque se cuelga. Por eso los pares solo se eligen entre equipos legales. Esas batallas sí se usan para entrenar BC y $\hat v$; hay que decidir si eso está bien (**decisión a defender**).
- **Barrera de test:** el código no deja cargar test sin pedirlo explícitamente (`allow_test=True`). Hay un test que lo comprueba.

### 2.4 Estadísticas de uso y leave-one-out (`src/bringvgc/data/usage.py`)

Para cada especie se cuenta, en train, qué fracción de las veces fue líder, reserva o se quedó fuera. Con brings incompletos se usan **cuentas fraccionales**: si una reserva pudo ser cualquiera de 3, cada una suma 1/3. Las tasas se suavizan hacia 2/6 con 5 "pseudo-batallas" para que una especie vista una sola vez no tenga tasas extremas.

Estas tasas son features del estado y la base del baseline Usage. **Leave-one-out (LOO):** cuando se calculan las features de una batalla de train, se resta lo que aportó esa misma batalla. Si no, la feature "esta especie suele liderar" contendría la respuesta de esa batalla.

### 2.5 Stats imputados: A8 (`src/bringvgc/features/spreads.py`), **BORRADOR, decisión a defender**

Los logs no traen naturaleza ni stat points, pero Showdown los necesita. La regla tiene que ser **fija e igual para los dos lados** (si no, un lado tendría ventaja por cómo se inventaron sus stats y no por su bring).

**Regla implementada (`role_spread`, la que pidió el dueño):** el rol de cada Pokémon se deduce de sus movimientos.

| Rol | Cómo se decide | Naturaleza | Stat points (HP/Atk/Def/SpA/SpD/Spe) |
|---|---|---|---|
| Físico | más movimientos de daño físicos que especiales | Adamant | 2 / 32 / 0 / 0 / 0 / 32 |
| Especial | más especiales que físicos | Modest | 2 / 0 / 0 / 32 / 0 / 32 |
| Mixto | empate, con al menos uno de daño | Hardy (neutra) | 2 / 16 / 0 / 16 / 0 / 32 |
| Soporte | ningún movimiento de daño | Hardy (neutra) | 32 / 0 / 17 / 0 / 17 / 0 |

Todos suman 66 y ninguno pasa de 32 por stat (los límites del Showdown fijado). Hay un test que lo comprueba.

**Alternativas descartadas:**

| Alternativa | Por qué no |
|---|---|
| Reparto aleatorio (lo de la Fase 0, `a8_rule: random`) | Mete ruido que no tiene nada que ver con el bring; se deja solo para reproducir la Fase 0 |
| Reparto neutro igual para todos | Un atacante físico real nunca invierte en SpA; el bot jugaría con Pokémon peores de lo que son |
| Stats "reales" de uso | No hay una fuente abierta y verificable para Reg M-A |
| Regla que mira Trick Room (0 de velocidad) | Una regla más que justificar; queda como trabajo futuro |

**Qué pueden preguntar:** "¿no favorece esta regla a algún tipo de equipo?" Sí puede: los equipos de Trick Room quedan con velocidad alta, lo que los perjudica. Es una limitación que hay que declarar.

### 2.6 El codificador DeepSets (`src/bringvgc/models/encoder.py`)

Cada Pokémon se convierte en un vector de 64 números a partir de su especie, objeto, habilidad, movimientos y sus tasas de uso. Después, a cada Pokémon propio se le pega el **promedio** de los vectores rivales (y el del propio equipo) y pasa por otra red pequeña. Así cada Pokémon "ve" contra qué se enfrenta. Usar promedios hace que el orden de los Pokémon en la hoja no importe.

**Por qué DeepSets y no un Transformer (A9):** hay unos 9 700 ejemplos de train; un modelo más simple se entrena más rápido, se depura mejor y es más fácil de defender. El agregado del promedio del propio equipo es un añadido al plan (las reservas dependen de los compañeros).

### 2.7 La política autorregresiva (`src/bringvgc/models/policy.py`)

Primero elige el par líder (15 opciones) y después, sabiendo cuál eligió, las reservas (6 opciones entre los 4 que quedan). Matemáticamente, $\pi(b|s) = \pi(\text{líderes}|s) \cdot \pi(\text{reservas}|s, \text{líderes})$. Hay un test que comprueba que salen exactamente 90 brings distintos, todos legales, y que la máscara elimina los brings con un Pokémon no disponible.

### 2.8 $\rho_{BC}$: imitar a los humanos (`src/bringvgc/train/bc.py`)

Es la política anterior entrenada para hacer lo que hicieron los humanos en train. La pérdida es la (7) del survey, $-\log \sum_{b\in\mathcal C(o)} \pi(b|s)$: si el bring se vio completo es la entropía cruzada normal; si no, basta con que la política ponga probabilidad en *alguno* de los brings posibles.

**Early stopping estratificado:** la pérdida es más fácil cuando $|\mathcal C|$ es grande (acertar a "alguno de 6" es más fácil que a "este"), así que se elige la época con la pérdida de los ejemplos con $|\mathcal C|=1$.

**Para qué sirve:** (1) punto de partida del agente RL; (2) baseline "BC": ¿hacía falta RL o bastaba imitar?; (3) predictor del bring rival (llamándolo con los equipos al revés).

### 2.9 $\hat v$: el evaluador de matchup (`src/bringvgc/models/vhat.py`), revisión de José

Dado mi bring y el del rival, da la probabilidad de que yo gane: $\hat v(b_1,b_2) = \sigma(f(b_1)^\top M f(b_2))$ con $M$ antisimétrica. Eso garantiza por construcción que $\hat v(b_1,b_2) + \hat v(b_2,b_1) = 1$ (hay un test). Se entrena con la verosimilitud marginalizada (11): como no siempre se sabe el bring exacto, se promedia sobre todos los consistentes.

**Detalle importante para la defensa:** medir $\hat v$ usando los $\mathcal C(o)$ observados está **contaminado**, porque cuántas reservas se revelaron depende de quién ganó. Por eso se reportan dos métricas en val:

- **team** (la principal): $q$ uniforme sobre los 90 brings de cada lado, sin usar nada de lo que pasó en la batalla. Es la comparable con Oblander (equipo contra equipo) y la que decide el early stopping;
- **obs**: con los $\mathcal C(o)$ observados, solo como diagnóstico.

**Para qué sirve:** solo para el Planificador-$\hat v$ y como plan B. **Nunca es la recompensa del agente (A2).**

### 2.10 La arena (`src/bringvgc/sim/arena.py`)

Conecta con el Showdown local y juega batallas con dos jugadores `MethodPlayer`. Los dos usan **la misma política de batalla** (SimpleHeuristicsPlayer + megaevolucionar en cuanto se pueda, `auto_mega`). Lo único que cambia es el método que decide el Team Preview. Ese es el control experimental central.

- La recompensa es +1 si gana, −1 si pierde y 0 si empata o pasa de 50 turnos.
- Si un bloque de batallas se cuelga más de 10 minutos, se descarta, se cuenta y se recrean los jugadores.
- Los nombres de usuario incluyen el número de proceso: con los nombres por defecto, dos procesos a la vez chocaban en el login (pasó en el piloto).

### 2.11 Los baselines (`src/bringvgc/baselines/methods.py`)

| Baseline | Qué hace | Para qué está |
|---|---|---|
| Random | Uno de los 90 al azar | ¿Aprendió algo el agente? |
| Usage (estocástico) | Puntúa cada bring con las tasas de uso de train y sortea según esa puntuación | Es **el rival** de la RQ |
| Usage-argmax | El bring con mejor puntuación de uso | Baseline de literatura del survey |
| Tipos | Para cada Pokémon: qué tan bien pega a los rivales menos qué tan bien le pegan a él; trae los 4 mejores y lideran los 2 mejores | Regla de dominio escrita a mano |
| BC | $\rho_{BC}$ greedy | ¿Aportó algo el RL? |
| Planificador-$\hat v$ | Prueba los 90 brings y elige el que maximiza $\hat v$ promedio contra la distribución de Usage del rival | "Búsqueda con fitness aprendida" del survey, sin GA (A3) |

### 2.12 El agente RL (`src/bringvgc/train/rl.py`), revisión de José

**Algoritmo:** REINFORCE con baseline. En simple:

1. Toma un par de equipos de train y un lado al azar.
2. El agente **sortea** un bring con su política actual; el rival sortea el suyo con Usage.
3. Se juega la batalla en Showdown y se obtiene $R = +1$ o $-1$.
4. Con 128 batallas, se ajusta la política: los brings que ganaron más que el promedio del lote se vuelven más probables y los que ganaron menos, menos probables.

La función que se minimiza:

$$\mathcal L = -\overline{(R - \bar R)\log\pi(b|s)} \;-\; \beta_H\,\mathcal H[\pi] \;+\; \beta_{KL}\,\mathrm{KL}(\pi\,\|\,\rho_{BC})$$

- $\bar R$ (el promedio del lote) es el **baseline**: reduce el ruido sin sesgar el gradiente.
- $\mathcal H$ es la **entropía**: un bono por no volverse determinista demasiado pronto (exploración). $\beta_H = 0.01$.
- $\mathrm{KL}$ mide cuánto se aleja la política de $\rho_{BC}$; penalizarla ($\beta_{KL}=0.1$) evita que el agente se vaya a brings rarísimos que solo funcionan por debilidades del bot de batalla.

**Por qué este algoritmo:** el episodio dura 2 pasos y la recompensa llega al final, así que el retorno de cada batalla es exacto (no hay que estimar valores futuros). Por eso DQN o A2C no aportan nada. **No hay replay buffer ni red objetivo**: el método usa solo los datos de la política actual y no tiene targets que estabilizar. Hay que decirlo explícitamente porque el enunciado menciona esos hiperparámetros.

**Detalles de implementación:**

- La política se usa sin dropout (modo eval) tanto para sortear como para el gradiente, para que la distribución que se optimiza sea la misma con que se sortea.
- Las batallas van en bloques de 8 del mismo par y lado (el estado es el mismo dentro del bloque).
- Cada 2 000 episodios: evaluación greedy en los 10 pares de val (200 batallas por par) y checkpoint. El mejor checkpoint se elige por val, **nunca por test**.
- Se puede cortar y reanudar (`--resume results/rl_<id>`), con el optimizador y los generadores aleatorios incluidos (probado en el smoke test).

### 2.13 La estadística (`src/bringvgc/eval/stats.py`, `aggregate.py`)

- **Wilson** por par: el intervalo de confianza de una proporción. Con 1000 batallas da ±0,03.
- **Intervalo t sobre pares:** el resultado final se promedia sobre los 20 pares de test y se calcula un intervalo t al 95 %. La unidad es el par porque **los pares varían mucho más entre sí que el ruido de cada par** (en el piloto, la desviación entre pares fue de 0,09 a 0,17).
- **Regla preregistrada (`docs/EXPERIMENTS.md`):** "supera el 55 %" solo si el límite inferior del intervalo t es mayor que 0,55. Hay que hacer commit de ese archivo **antes** de correr test.

## 3. Cómo correr cada paso

Desde la raíz del repo, con Showdown corriendo (`node pokemon-showdown start --no-security` en `..\vgc-bench\pokemon-showdown`) y `$env:PYTHONPATH="src"`:

| Paso | Comando | Tiempo en el piloto |
|---|---|---|
| Datos | `.\.venv\Scripts\python.exe scripts\build_dataset.py` | ~1 min |
| Tests | `.\.venv\Scripts\python.exe -m pytest -q` | ~4 s |
| $\rho_{BC}$ | `.\.venv\Scripts\python.exe -m bringvgc.train.bc --seed 0` | ~2 min |
| $\hat v$ | `.\.venv\Scripts\python.exe -m bringvgc.train.vhat --seed 0` | ~1 min |
| Un baseline en val | `.\.venv\Scripts\python.exe -m bringvgc.eval.run_arena --method tipos --pairs val` | ~40 s (2 000 batallas) |
| RL | `.\.venv\Scripts\python.exe -m bringvgc.train.rl --seed 0` | ver §4 |
| Tabla | `.\.venv\Scripts\python.exe -m bringvgc.eval.aggregate --pairs val` | segundos |
| Todo | `.\scripts\run_all.ps1 -Stage <data/offline/val/rl/test>` | — |

## 4. Resultados del piloto

**Todo esto es un smoke test: 1 semilla, solo val, no citable.** El detalle está en `results/PILOTO_ETAPA2.md`.

| Pieza | Resultado en val | Cómo leerlo |
|---|---|---|
| Parser | 99,96 % de batallas válidas | Pasa el umbral del 98 % |
| $\rho_{BC}$ | NLL 4,11 con $|\mathcal C|=1$ (azar 4,50); acierta el par líder 20,8 % (azar 6,67 %) | Aprende algo de los humanos, pero lejos de perfecto |
| $\hat v$ | 55,1 % de accuracy, AUC 0,570 | Por encima del azar; muy por debajo de Oblander (66,4 %, que usó 11,7 M partidas) |
| Usage vs Usage | 0,472 [0,382, 0,562] | Compatible con 50 %: el arena es simétrica |
| Mismo equipo y bring en ambos lados | 0,490 [0,468, 0,512] | Compatible con 50 % |
| Random / Tipos / BC / Planificador / Usage-argmax vs Usage | 0,545 / 0,591 / 0,520 / 0,500 / 0,453 | **Sorpresa:** Random y Tipos le ganan a Usage |
| Agente RL | 0,526 (inicio = $\rho_{BC}$) → 0,614 tras 10k episodios | Sube, pero es el mismo val que elige el checkpoint (optimista) |
| $T$ | 47,9 episodios/s entrenando; 51,7 batallas/s en la arena | El presupuesto completo cabe en unas 3 h en una PC |

**La sorpresa, en simple:** el plan esperaba "Random < 50 % contra Usage". Pasó lo contrario. Una explicación posible es que los humanos eligen su bring pensando en *cómo ellos juegan*, y el bot de batalla juega distinto, así que copiar a la población no ayuda al bot. Otra es que las batallas del bot son muy cortas (mediana de 3–5 turnos contra 7 de los humanos), así que pesa más quién lidera que la estrategia completa. Hay que investigarlo antes de las corridas largas. Cambia la lectura de la RQ: superar el 55 % contra un rival débil es menos impresionante, y eso hay que discutirlo en el artículo.

## 5. Qué falta (para ustedes)

Ordenado por urgencia:

1. **Anotar a mano los 20 replays** (`tests/fixtures/annotated_replays.json`): líderes y reservas reveladas de cada lado, con su nombre en `checked_by`. Hasta entonces el parser no está verificado.
2. **Decidir A8, `auto_mega` y el uso de las 80 batallas con equipos ilegales** (§6), y escribir el porqué con sus palabras.
3. **Revisar y hacer commit** de lo que les toca (ver la tabla de responsables en `docs/AI_USAGE.md`): José revisa $\hat v$, el Planificador y el bucle de RL.
4. **Investigar el hallazgo del piloto** (Random y Tipos por encima de Usage, §4) antes de las corridas largas: si Usage es un rival débil para este bot, la RQ del 55 % es más fácil de lo que parece y hay que discutirlo.
5. **Confirmar el umbral de parseo** (98 %, `min_parse_success` en `configs/data.yaml`).
6. **Corridas largas** (no se hicieron): $\rho_{BC}$ con semillas 1 y 2; RL con 3 semillas × 50k ($\beta_H=0.01$) y 3 × 15k ($\beta_H=0$); baselines en val con las 3 semillas de BC y Random (`run_all.ps1 -Stage offline/val/rl`). Repartir semillas entre las dos PCs y anotar el hardware (sale solo en `meta.json`).
7. **Commit de `docs/EXPERIMENTS.md` antes de test**, y después la corrida de test **una sola vez** (`run_all.ps1 -Stage test`).
8. Curvas con banda de ±1 desviación estándar entre semillas (los datos están en `results/rl_*/curves.csv` y `episodes.csv`; falta el notebook que las grafique leyendo `results/`).
9. Análisis cualitativo y casos de falla (§10 del plan), con los CSV por par.
10. Artículo: metodología definitiva, experimentos y resultados; corregir "2 registros excluidos" por 3; declarar A9, `auto_mega`, el validador y la métrica de $\hat v$ a nivel de equipo.

## 6. Decisiones que tienen que poder defender

Cada fila: qué se decidió, por qué, y qué se descartó. Lo marcado ⚠ no lo han decidido ustedes todavía.

| Decisión | Por qué | Alternativa descartada |
|---|---|---|
| MDP de 2 pasos (líderes, luego reservas), $\gamma=1$ | Separa líderes de reservas; $\gamma$ no cambia la política óptima | Bandido de 90 brazos; solo líderes; Team Preview dentro del MDP de batalla; juego de Markov; recompensa $=\hat v$ (ver `docs/MDP.md`) |
| REINFORCE + baseline de media + KL | Retorno exacto en 2 pasos; sin red extra | DQN/A2C (bootstrapping sin necesidad); $V_\psi$ aprendido (una red más) |
| Partición por serie Bo3 y por tiempo | Partir una serie es leakage | Partición aleatoria por batalla |
| "Equipo visto" = mismo conjunto de 6 especies | Más estricto que comparar la hoja completa | Comparar la hoja completa (un movimiento distinto bastaría para "no visto") |
| Supervisión parcial (7) y (11), sin descartar registros | Descartar sesga hacia perdedores | Usar solo brings completos |
| Features de uso solo de train + LOO | Evita leakage temporal y de etiqueta | Tasas con todo el dataset |
| ⚠ A8 `role_spread` | Ver §2.5 | Aleatorio, neutro, stats reales, regla con Trick Room |
| ⚠ `auto_mega` en $\pi_b$ | Sin megas, Reg M-A no es representativo | El bot original (nunca megaevoluciona) |
| ⚠ Usar en BC/$\hat v$ las 80 batallas con equipos ilegales | Siguen siendo decisiones humanas reales | Excluirlas también del entrenamiento |
| $\hat v$ evaluado a nivel de equipo ($q$ uniforme) | La versión con $\mathcal C(o)$ observados filtra el resultado por la revelación | Reportar solo la métrica (11) |
| Unidad estadística = par | La varianza entre pares domina | Tratar las 20 000 batallas como independientes |
| 3 semillas (A5) | Mínimo del enunciado; se reporta también la dispersión sobre pares | 5 o 10 semillas |

**Preguntas que va a hacer el profesor (y dónde está la respuesta):**

1. *¿Por qué ese MDP y qué alternativas descartaron?* → `docs/MDP.md`.
2. *¿Qué muestra el análisis multi-semilla sobre la estabilidad?* → Cuando tengan las 3 semillas: comparar la desviación entre semillas con el ancho de Wilson por par; si la de las semillas es mucho mayor, el entrenamiento es inestable.
3. *¿Qué comportamiento indeseado podría premiar la recompensa?* → (a) explotar debilidades del bot de batalla (brings que solo ganan porque SimpleHeuristics juega mal contra ellos); (b) sobreajustarse al rival Usage; (c) con empate = 0 al llegar a 50 turnos, preferir alargar batallas perdidas. La curva de KL y la tasa de empates son la evidencia.
4. *¿Por qué RL y no solo BC?* → La comparación agente vs BC por par.
5. *¿Dónde podía haber leakage?* → §2.3, §2.4, la lista de información negada en `docs/MDP.md` y los tests de `tests/test_data.py`.
6. *¿Por qué Random le gana a Usage?* → Ver §4: es un hallazgo del piloto que hay que investigar antes de defender Usage como rival.

## 7. Glosario

| Término | Qué significa aquí |
|---|---|
| **A1–A9** | Los ajustes respecto a la Etapa 1 listados en el plan v2 §3 (p. ej. A7 = partición por serie, A8 = stats imputados) |
| **Adam** | Optimizador estándar de redes neuronales; ajusta cada peso con un paso adaptativo |
| **Arena** | El código que conecta con Showdown y juega las batallas entre dos métodos |
| **AUC** | *Area Under the ROC Curve*: probabilidad de que el modelo le dé más probabilidad de ganar al ganador que al perdedor en un par al azar. 0,5 = azar, 1 = perfecto |
| **auto_mega** | Cambio a $\pi_b$: megaevolucionar en cuanto se pueda (el bot original nunca lo hacía) |
| **Baseline** | (1) Un método de comparación (Random, Usage…). (2) En REINFORCE, el valor que se resta al retorno para bajar la varianza ($\bar R$) |
| **BC** | *Behaviour Cloning*: aprender a imitar las decisiones de humanos a partir de logs |
| **Bo3** | *Best of 3*: serie de hasta tres partidas entre los mismos dos jugadores |
| **Bring** | Los 4 Pokémon que se llevan a la batalla: 2 líderes y 2 reservas |
| **Checkpoint** | Copia guardada del modelo (y del optimizador) en un momento del entrenamiento |
| **$\mathcal C(o)$** | Conjunto de brings consistentes con lo observado en el log: 1, 3 o 6 |
| **DeepSets** | Red que procesa un conjunto (aquí, un equipo) aplicando la misma red a cada elemento y promediando; el orden no importa |
| **Dropout** | Apagar neuronas al azar durante el entrenamiento supervisado para no sobreajustar |
| **Early stopping** | Parar de entrenar cuando la métrica en val deja de mejorar y quedarse con la mejor época |
| **Entropía ($\mathcal H$)** | Qué tan "repartida" está la probabilidad de la política. Alta = explora; baja = casi siempre elige lo mismo |
| **Episodio** | Aquí: una batalla. Dos decisiones (líderes, reservas) y una recompensa al final |
| **Exploración–explotación** | El dilema entre probar brings nuevos (explorar) y usar el que ya parece mejor (explotar) |
| **$\gamma$ (descuento)** | Cuánto valen las recompensas futuras frente a las inmediatas. Aquí $\gamma=1$ y no cambia nada porque solo hay recompensa al final |
| **Greedy** | Elegir siempre la opción más probable, sin sortear |
| **Intervalo t** | Intervalo de confianza para una media con pocas unidades (aquí, 20 pares), usando la distribución t de Student |
| **KL** | *Divergencia de Kullback-Leibler*: cuánto se diferencia una distribución de otra. Aquí, cuánto se aleja el agente de $\rho_{BC}$ |
| **Leakage** | Que el modelo vea, directa o indirectamente, información que no tendría al decidir (p. ej. la respuesta). Penalización de −15 |
| **Líderes / reservas** | Los 2 Pokémon que empiezan en el campo / los 2 que esperan atrás |
| **LOO** | *Leave-one-out*: al calcular una feature para una batalla de train, se excluye esa misma batalla |
| **Log set** | Los registros de batallas humanas de VGC-Bench (Reg M-A) |
| **MDP** | *Markov Decision Process*: estados, acciones, transiciones, recompensa y descuento. Es la formulación formal del problema |
| **Mega / megaevolución** | Transformación de un Pokémon con su megapiedra durante la batalla (una por batalla) |
| **NLL** | *Negative log-likelihood*: la pérdida; cuanto más baja, más probabilidad le da el modelo a lo que pasó. Con 90 opciones al azar sería $\ln 90 \approx 4{,}50$ |
| **On-policy** | El algoritmo aprende solo con datos generados por la política actual (no reutiliza datos viejos) |
| **OTS** | *Open Team Sheets*: las hojas de equipo abiertas (especie, objeto, habilidad, movimientos), visibles para los dos jugadores |
| **Par de equipos** | Dos equipos concretos que se enfrentan; la unidad de evaluación |
| **Partición temporal** | Dividir los datos por fecha: train (pasado), val (medio), test (futuro) |
| **$\pi_b$** | La política de batalla fija (SimpleHeuristicsPlayer + auto_mega), la misma en los dos lados |
| **Piloto / smoke test** | Corrida corta para comprobar que todo funciona y medir tiempos. No es un resultado citable |
| **Planificador-$\hat v$** | Baseline que prueba los 90 brings con $\hat v$ y elige el mejor |
| **poke-env** | Librería de Python para conectar bots con Showdown (aquí, el fork de vgc-bench) |
| **Preregistro** | Escribir la regla de decisión antes de ver los resultados de test |
| **REINFORCE** | Algoritmo de gradiente de política: sube la probabilidad de las acciones que dieron más recompensa de lo esperado |
| **Reg M-A** | El reglamento de VGC del log set (`gen9championsvgc2026regma`) |
| **RL** | *Reinforcement Learning*, aprendizaje por refuerzo: aprender a decidir probando y recibiendo recompensas |
| **RQ** | *Research question*, la pregunta de investigación |
| **$\rho_{BC}$** | La política entrenada por imitación (BC) |
| **Semilla** | Número que fija el azar de una corrida para poder repetirla |
| **Showdown** | El simulador de batallas Pokémon; aquí corre en local (commit `913da36`) |
| **Species Clause** | Regla que prohíbe dos Pokémon con el mismo número de Pokédex en un equipo |
| **$T$** | *Throughput*: batallas (o episodios) por segundo |
| **Team Preview** | El momento antes de la batalla en que se ven los dos equipos y se elige el bring |
| **Train / val / test** | Datos para entrenar / para elegir hiperparámetros y checkpoints / para la evaluación final, que se usa una sola vez |
| **Usage** | El baseline que elige como la población: según qué tan seguido cada especie lidera, va de reserva o se queda fuera en train |
| **$v$** | Número de reservas reveladas en un log (0, 1 o 2) |
| **$\hat v$** | Evaluador de matchup: probabilidad de ganar de un bring contra otro |
| **Validador** | El `TeamValidator` de Showdown, que dice si un equipo es legal en el formato |
| **Wilson** | Un intervalo de confianza para una proporción (win rate) que funciona bien también cerca de 0 o 1 |
| **Win rate** | Fracción de batallas ganadas |
