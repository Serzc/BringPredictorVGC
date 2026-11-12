> **BORRADOR asistido por IA (07-11-2026, actualizado el 11-11-2026 tras la reestructuración numpy).** Explicación en lenguaje simple de lo que se implementó para la Etapa 2, siguiendo `Research/ETAPA2_PLAN_v2.md`. No es texto para el artículo: es para que Sergio y José entiendan, revisen y defiendan cada pieza. Lo marcado **"decisión a defender"** todavía no lo han decidido ustedes. **Aviso:** desde el 11-11-2026 el código usa modelos lineales en numpy (§3). Los planes de `Research/` todavía hablan de DeepSets y torch: ahí el código y el plan no coinciden.

# Etapa 2 explicada: qué hace cada parte, por qué y cómo correrla

## 0. La idea en un párrafo

En VGC, antes de cada batalla los dos jugadores ven los 6 Pokémon del rival (Team Preview) y eligen **4 de sus 6**: dos que empiezan en el campo (los **líderes**) y dos que esperan atrás (las **reservas**). A esa elección le llamamos **bring**. Hay exactamente **90** brings posibles. El proyecto pregunta: *¿puede un agente de aprendizaje por refuerzo aprender a elegir el bring de forma que gane al menos el 55 % de las veces contra alguien que elige "como elige la mayoría de jugadores" (el baseline Usage)?* Para que la comparación sea justa, las batallas las juega **el mismo bot en los dos lados** (la política de batalla $\pi_b$). Lo único que cambia es quién eligió el bring.

## 1. Cómo correrlo (4 comandos)

```powershell
pip install -r requirements.txt     # numpy, pandas, PyYAML, pytest y el fork de poke-env (sin torch)
python src/main.py setup            # baja el log set (o reutiliza la copia local) y prepara Showdown
python src/main.py all              # data -> bc -> vhat -> val -> rl -> tabla (arranca y apaga Showdown solo)
python src/main.py test             # corrida FINAL en test, una sola vez
```

- `python src/main.py all --smoke` hace lo mismo con presupuestos mínimos (unos 70 s). Sirve para ver que todo corre. **No es un resultado.**
- Pasos sueltos: `python src/main.py data | bc | vhat | val | rl | tablas`.
- Tests: `python -m pytest`.

## 2. El recorrido completo y dónde está cada parte

```
logs humanos (6 905 batallas)          src/dataset/raw/          (setup lo descarga)
   │  1. dataPrep/: leer, limpiar, particionar          -> src/dataset/clean/, src/dataset/splits/
   ▼
registros ──► estadísticas de uso (solo train) ──► baseline Usage = el rival
   ├─► 2a. ρ_BC: imita a los humanos (pérdida 7)            models/bc.py, training/entrenar_bc.py
   ├─► 2b. v̂: quién gana con un bring contra otro (11)     models/vhat.py, training/entrenar_vhat.py
   ▼
3. arena: Showdown local + el mismo bot en ambos lados      sim/
   ▼
4. agente RL: parte de ρ_BC y aprende jugando (REINFORCE)   models/rl_agent.py, training/entrenar_rl.py
   ▼
5. evaluación: win rate por par, intervalo t, regla 55 %    training/eval/   -> src/output/
```

| Carpeta (`src/`) | Qué tiene |
|---|---|
| `main.py` | El único punto de entrada |
| `config/` | `config.py` (rutas, semillas, `meta.json`) y un `.yaml` por paso; `smoke.yaml` pisa los presupuestos |
| `scripts/` | `setup.py`, `validate_teams.js` y, en `fase0/`, los scripts históricos de la Fase 0 |
| `dataPrep/` | Parser de logs, validador, partición por serie y la construcción del dataset |
| `dataset/` | `raw/` y `clean/` (no se versionan), `splits/` (**sí** se versiona: pares congelados) |
| `utils/` | Espacio de 90 brings y $\mathcal C(o)$, features, usage stats, regla A8, estadística |
| `models/` | `base.py` (clase padre `MetodoBring`), `bc.py`, `rl_agent.py`, `vhat.py` |
| `baselines/` | `random.py`, `usage.py`, `tipos.py`, `planner.py` (cada uno corto, con su porqué) |
| `sim/` | Arrancar/apagar Showdown, el jugador `BringPlayer` y la arena |
| `training/` | Entrenamiento de BC, v̂ y RL; `eval/` = evaluación en la arena, tablas, regla 55 %, diagnósticos |
| `output/` | Corridas (`<tipo>_<fecha>_s<semilla>/` con `meta.json`, CSV y `.npz`). Solo se versionan los `.md` |
| `tests/` | pytest: espacio de acciones, parser, leakage, gradientes, antisimetría, arena |

**Todos los métodos heredan de `MetodoBring`** (`models/base.py`): dan una distribución sobre los 90 brings con `probs(mi_equipo, equipo_rival)`. Por eso la arena los trata igual.

## 3. Decisión: de torch a modelos lineales en numpy (11-11-2026)

**Qué se decidió.** Se quitaron torch y el DeepSets. $\rho_{BC}$ y el agente RL son ahora una **softmax lineal** sobre features hechas a mano. $\hat v$ es una **regresión logística antisimétrica**. Los gradientes están escritos a mano y se comprueban contra diferencias finitas (`src/tests/test_gradients.py`).

**Por qué.**
- **Se puede defender línea por línea.** Cada peso tiene nombre (p. ej. "especie=incineroar como líder"). La rúbrica castiga lo que no se puede justificar, y con una red eso cuesta mucho más.
- **Hay pocos datos:** unos 9 700 ejemplos de train. En el piloto (1 semilla), BC con DeepSets quedó por debajo de Random en la arena (0,520 contra 0,545), y $\hat v$ sobreajustaba desde la época 3. La capacidad extra no se tradujo en nada medible.
- **La instalación es más simple:** sin torch, `pip install` pesa poco y no depende de la versión de CUDA ni de la CPU.
- **Es rápido:** entrenar BC o $\hat v$ tarda segundos. El cuello de botella son las batallas.

**Alternativas descartadas.**

| Alternativa | Por qué no |
|---|---|
| Mantener el DeepSets en torch | Más difícil de explicar y de instalar; en el piloto no mostró una ventaja medible (1 semilla, no concluyente) |
| MLP en numpy | Habría que escribir y verificar el backprop de varias capas; más superficie de error para poca ganancia esperada |
| Bandido tabular por par | No generaliza a pares nuevos de val/test (todos tienen equipos no vistos) |
| scikit-learn | No trae la pérdida de etiquetas parciales (7), la verosimilitud marginalizada (11) ni el KL; habría que escribirlas igual |

**Qué se pierde (hay que declararlo).**
- **Interacciones entre Pokémon.** El puntaje de un par es la suma de los puntajes de sus dos Pokémon. El modelo no puede aprender "estos dos juntos son mejores que la suma de cada uno" (sinergia). El paso 2 depende del paso 1 solo por cuáles 4 quedan disponibles (la máscara).
- **Objeto, habilidad y movimientos** solo entran a través de los tipos de los movimientos y de la especie. No hay embedding de objeto ni de habilidad.
- **Capacidad.** Si el agente lineal no llega al 55 %, puede ser por el modelo y no por el problema. Es un hallazgo que hay que reportar, **no algo que se ajuste hasta pasar**.
- **Los números del piloto con torch** (`src/output/legacy_torch/`) no son comparables con los nuevos.

## 4. Cada parte, en simple

### 4.1 El parser (`dataPrep/parser.py`)

**Qué hace.** Lee el texto de cada batalla (líneas como `|switch|p1a: Rotom|Rotom-Wash, L50|100/100`) y saca las dos hojas de equipo (OTS), los **líderes** (los dos primeros que entran de cada lado después de `|start`), las **reservas reveladas** (cualquier otro que entró después), el ganador, los turnos y la serie Bo3.

**Nombres.** Apodos, formas Mega o **Illusion** cambian el nombre en batalla. La solución es comparar por **número de Pokédex**: la *Species Clause* prohíbe repetir número, así que identifica la posición en la hoja. Con Illusion, `|replace|` dice quién era y se corrige la entrada anterior.

**Verificación pendiente.** Hay 20 replays elegidos en `src/tests/fixtures/annotated_replays.json` con los valores esperados en "TODO". **Ustedes tienen que llenarlos mirando cada log.** Nunca se generan con el parser. Hasta entonces, esos 20 tests se saltan y el parser no está verificado. Se excluyen 3 batallas jugadas con 5 Pokémon (`|teamsize|p1|5`). El survey dice "2": hay que corregirlo.

### 4.2 Brings consistentes $\mathcal C(o)$ (`utils/brings.py`)

Si un jugador ganó rápido, puede que alguna reserva nunca entrara. No sabemos su bring exacto, pero sí **cuáles son posibles**: 1, 3 o 6 según cuántas reservas se vieron (2, 1 o 0). No se tiran esas batallas, porque **los ganadores revelan menos** (73,6 % contra 94,7 %). Los modelos reparten la probabilidad entre los brings posibles.

### 4.3 Partición train / val / test (`dataPrep/particion.py`, `config/data.yaml`)

- **Por tiempo:** train hasta el 25-05-2026, val hasta el 05-06-2026, test después.
- **Por serie Bo3:** una serie nunca se parte entre particiones (si se partiera, sería **leakage**).
- **Pares congelados** (`src/dataset/splits/`): 200 de train, 10 de val, 20 de test. En val y test **ningún equipo aparece en train**, y test no comparte equipos con val. **No se regeneran nunca.** Un test compara su SHA-256 con el commit `d3b1d7d`.
- **Barrera de test:** el código no carga test sin `allow_test=True`, y eso solo lo hace `main.py test`.

### 4.4 Estadísticas de uso y leave-one-out (`utils/usage.py`)

Para cada especie se mide, en train, la fracción de veces que fue líder, reserva o se quedó fuera. Con brings incompletos se usan **cuentas fraccionales**, y las tasas se suavizan hacia 2/6 con 5 pseudo-batallas. **LOO:** al calcular las features de una batalla de train se resta lo que aportó esa misma batalla. Si no, la feature contendría la respuesta.

### 4.5 Stats imputados: A8 (`utils/spreads.py`), **BORRADOR, decisión a defender**

Los logs no traen naturaleza ni stat points. La regla es **fija e igual para los dos lados**. Se deduce un rol de los movimientos:

| Rol | Cómo se decide | Naturaleza | Stat points (HP/Atk/Def/SpA/SpD/Spe) |
|---|---|---|---|
| Físico | más movimientos de daño físicos que especiales | Adamant | 2 / 32 / 0 / 0 / 0 / 32 |
| Especial | más especiales que físicos | Modest | 2 / 0 / 0 / 32 / 0 / 32 |
| Mixto | empate, con al menos uno de daño | Hardy | 2 / 16 / 0 / 16 / 0 / 32 |
| Soporte | ningún movimiento de daño | Hardy | 32 / 0 / 17 / 0 / 17 / 0 |

Alternativas descartadas: un reparto aleatorio (mete ruido ajeno al bring), uno neutro para todos (un atacante físico nunca invierte en SpA), stats reales de uso (no hay una fuente abierta para Reg M-A) y una regla para Trick Room (queda como trabajo futuro). **Limitación:** los equipos de Trick Room quedan con velocidad alta, lo que los perjudica.

### 4.6 Las features (`utils/features.py`)

Por cada Pokémon propio, frente a los 6 rivales:

| Feature | Qué mide |
|---|---|
| `log_p_lider`, `log_p_reserva`, `log_p_fuera` | Qué hace la población con esa especie (log de las tasas de train) |
| `uso_norm` | Qué tan común es la especie |
| `ofensiva_tipos` | Qué tan bien pegan sus movimientos a los rivales (promedio de la mejor efectividad) |
| `defensa_tipos` | Qué tan bien le pegan los tipos rivales (más alto = más débil) |
| `especie=<nombre>` | One-hot de la especie (vocabulario de train; `<UNK>` si no se vio) |

Las tasas están en log para que el modelo pueda **imitar exactamente** al baseline Usage si eso fuera lo mejor. Nunca entran el número de revelados, los turnos, el resultado, el rating, los nombres ni el historial Bo3.

### 4.7 La política lineal y $\rho_{BC}$ (`models/bc.py`, `training/entrenar_bc.py`)

Cada Pokémon recibe dos puntajes: uno como líder ($\theta_L \cdot \phi_x$) y otro como reserva ($\theta_R \cdot \phi_x$).

- **Paso 1:** el logit de cada uno de los 15 pares líderes es la suma de los puntajes de líder de sus dos Pokémon. Se aplica softmax.
- **Paso 2:** entre los 4 que quedan, el logit de cada uno de los 6 pares de reservas es la suma de sus puntajes de reserva. Se aplica softmax.
- $\pi(b|s) = \pi(\text{líderes}|s) \cdot \pi(\text{reservas}|s,\text{líderes})$.

$\rho_{BC}$ es esa política entrenada con la pérdida (7), $-\log \sum_{b\in\mathcal C(o)} \pi(b|s)$, y Adam. El early stopping se hace con la pérdida de los ejemplos con $|\mathcal C|=1$, porque con $|\mathcal C|$ grande la pérdida baja sola. $\rho_{BC}$ sirve de punto de partida del RL, de baseline ("¿hacía falta RL?") y de predictor del bring rival.

### 4.8 $\hat v$ (`models/vhat.py`), revisión de José

$\hat v(b_1,b_2) = \sigma(w \cdot x(b_1,b_2))$, donde $x$ es *"features de $b_1$ menos features de $b_2$"* (líderes y reservas por separado) más una ventaja de tipos entre los 4 de cada lado. Al no tener sesgo, al cambiar los lados el logit cambia de signo, y por eso $\hat v(b_1,b_2) + \hat v(b_2,b_1) = 1$ **exacto** (hay un test). Se entrena con la verosimilitud marginalizada (11).

- La métrica principal es **team** ($q$ uniforme sobre los 90 brings): no usa nada de lo que pasó en la batalla.
- **obs** (con los $\mathcal C(o)$ observados) está contaminada por la revelación y se reporta solo como diagnóstico.
- **Nunca es la recompensa del agente (A2).**

### 4.9 La arena (`sim/`)

`sim/showdown.py` arranca Showdown (falla si el puerto 8000 está ocupado o si no abre en 60 s) y lo apaga siempre al terminar (`taskkill /T`). `sim/jugador.py` define `BringPlayer`: el mismo bot de batalla en los dos lados (SimpleHeuristicsPlayer + `auto_mega`). Solo cambia el Team Preview.

- La recompensa es +1, −1, o 0 si hay empate o se pasa de 50 turnos.
- Un bloque colgado más de 10 min se descarta y se cuenta.

### 4.10 Los baselines (`baselines/`)

| Baseline | Qué hace | Para qué está |
|---|---|---|
| Random | Uno de los 90 al azar | Piso: ¿aprendió algo el agente? |
| Usage (estocástico) | Sortea según las tasas de uso de train | **El rival** de la RQ |
| Usage-argmax | El bring más típico | Baseline de literatura del survey |
| Tipos | Trae los 4 con mejor ventaja de tipos; lideran los 2 mejores | Regla de dominio simple |
| BC | $\rho_{BC}$ greedy | ¿Aportó algo el RL? |
| Planificador-$\hat v$ | Prueba los 90 y elige el mejor según $\hat v$ contra Usage | Búsqueda con fitness aprendida, sin GA (A3) |

### 4.11 El agente RL (`models/rl_agent.py`, `training/entrenar_rl.py`), revisión de José

**REINFORCE con baseline:**

1. Se toma un par de train y un lado al azar.
2. El agente **sortea** un bring y el rival sortea el suyo con Usage.
3. Se juega la batalla real y se obtiene $R=\pm1$.
4. Con un lote de batallas, los brings que ganaron más que el promedio se vuelven más probables.

$$\mathcal L = -\overline{(R - \bar R)\log\pi(b|s)} \;-\; \beta_H\,\mathcal H[\pi] \;+\; \beta_{KL}\,\mathrm{KL}(\pi\,\|\,\rho_{BC})$$

- $\bar R$ (`retorno_medio` en el código) baja el ruido sin sesgar el gradiente.
- $\mathcal H$ es un bono por explorar.
- El KL evita que el agente se vaya a brings rarísimos que solo ganan por debilidades del bot.

El episodio dura 2 pasos y el retorno es exacto, así que DQN o A2C no aportan nada. **No hay replay buffer ni red objetivo**: hay que decirlo, porque el enunciado menciona esos hiperparámetros. El mejor checkpoint se elige por val, **nunca por test**.

### 4.12 La estadística (`utils/estadistica.py`, `training/eval/tablas.py`)

- **Wilson** por par (±0,03 con 1000 batallas).
- **Intervalo t sobre pares:** la unidad es el par, porque los pares varían mucho más entre sí que el ruido de cada uno.
- **Regla preregistrada** (`docs/EXPERIMENTS.md`): "supera el 55 %" solo si el límite inferior del intervalo t es mayor que 0,55.

## 5. Resultados

- **Los del piloto de torch** (07-11-2026) están en `src/output/legacy_torch/PILOTO_ETAPA2.md`. Son de la era torch y **no son comparables** con el código actual. El hallazgo que dejaron sigue abierto: Random y Tipos le ganaron a Usage, y eso cambia la lectura del 55 %.
- **El smoke test del 11-11-2026** solo comprueba que el pipeline corre de punta a punta. Usa 2 pares y 8 batallas por par: **no es un resultado**.

## 6. Qué falta (para ustedes)

1. **Anotar a mano los 20 replays** (§4.1).
2. **Decidir A8, `auto_mega` y el uso de las 80 batallas con equipos ilegales**, con el porqué en sus palabras.
3. **Revisar y hacer commit** de lo que les toca (`docs/AI_USAGE.md`). José revisa $\hat v$, el Planificador y el RL.
4. **Investigar el hallazgo del piloto** (Usage como rival débil) con el código nuevo.
5. **Corridas largas** (`python src/main.py all`): 3 semillas de BC y de RL, con baselines en val. El barrido $\beta_H=0$ se corre con `python src/main.py rl --beta-H 0`. Repartan las semillas entre las dos PCs (el hardware queda en `meta.json`).
6. **Commit de `docs/EXPERIMENTS.md` antes de test**, y después `python src/main.py test` **una sola vez**.
7. Curvas con ±1 desviación estándar entre semillas (`src/output/rl_*/curves.csv`), análisis cualitativo y casos de falla.
8. **Actualizar los planes de `Research/`** (DeepSets → lineal) y el artículo.

## 7. Decisiones que tienen que poder defender

Lo marcado ⚠ no lo han decidido ustedes todavía.

| Decisión | Por qué | Alternativa descartada |
|---|---|---|
| MDP de 2 pasos, $\gamma=1$ | Separa líderes de reservas; $\gamma$ no cambia la política óptima | Bandido de 90 brazos; solo líderes; Team Preview dentro del MDP de batalla (ver `docs/MDP.md`) |
| Modelos lineales en numpy | Defendibles, rápidos, sin torch (§3) | DeepSets en torch, MLP, scikit-learn |
| REINFORCE + baseline de media + KL | Retorno exacto en 2 pasos | DQN/A2C; $V_\psi$ aprendido |
| Partición por serie Bo3 y por tiempo | Partir una serie es leakage | Partición aleatoria por batalla |
| "Equipo visto" = mismo conjunto de 6 especies | Más estricto que comparar la hoja completa | Comparar la hoja completa |
| Supervisión parcial (7) y (11), sin descartar registros | Descartar sesga hacia perdedores | Usar solo brings completos |
| Features de uso solo de train + LOO | Evita leakage temporal y de etiqueta | Tasas con todo el dataset |
| ⚠ A8 `role_spread` | §4.5 | Aleatorio, neutro, stats reales, Trick Room |
| ⚠ `auto_mega` en $\pi_b$ | Sin megas, Reg M-A no es representativo | El bot original |
| $\hat v$ evaluado a nivel de equipo | La métrica con $\mathcal C(o)$ filtra el resultado | Reportar solo (11) |
| Unidad estadística = par | La varianza entre pares domina | 20 000 batallas como independientes |
| 3 semillas | Mínimo del enunciado | 5 o 10 semillas |

**Preguntas que va a hacer el profesor:**

1. *¿Por qué ese MDP?* → `docs/MDP.md`.
2. *¿Qué muestra el multi-semilla sobre la estabilidad?* → Comparar la desviación entre semillas con el ancho de Wilson por par.
3. *¿Qué comportamiento indeseado podría premiar la recompensa?* →
   - explotar debilidades del bot de batalla;
   - sobreajustarse al rival Usage;
   - con empate = 0 a los 50 turnos, alargar batallas perdidas.

   La curva de KL y la tasa de empates son la evidencia.
4. *¿Por qué RL y no solo BC?* → La diferencia pareada agente − BC.
5. *¿Por qué un modelo lineal?* → §3, con lo que se pierde.
6. *¿Dónde podía haber leakage?* → §4.3, §4.4, `docs/MDP.md` y `src/tests/test_data.py`.

## 8. Glosario

| Término | Qué significa aquí |
|---|---|
| **A1–A9** | Los ajustes respecto a la Etapa 1 del plan v2 §3 (p. ej. A7 = partición por serie, A8 = stats imputados) |
| **Adam** | Optimizador que ajusta cada peso con un paso adaptativo (aquí escrito a mano en `models/bc.py`) |
| **Arena** | El código que conecta con Showdown y juega las batallas entre dos métodos |
| **AUC** | Probabilidad de que el modelo le dé más probabilidad de ganar al ganador que al perdedor. 0,5 = azar |
| **auto_mega** | Cambio a $\pi_b$: megaevolucionar en cuanto se pueda (el bot original nunca lo hacía) |
| **Baseline** | (1) Un método de comparación (Random, Usage…). (2) En REINFORCE, el valor que se resta al retorno ($\bar R$, `retorno_medio`) |
| **BC** | *Behaviour Cloning*: imitar las decisiones de humanos a partir de logs |
| **Bo3** | *Best of 3*: serie de hasta tres partidas entre los mismos dos jugadores |
| **Bring** | Los 4 Pokémon que se llevan: 2 líderes y 2 reservas |
| **Checkpoint** | Copia guardada de los pesos (aquí un `.npz`) |
| **$\mathcal C(o)$** | Brings consistentes con lo observado en el log: 1, 3 o 6 |
| **DeepSets** | Red para conjuntos que usaba la versión torch; **ya no se usa** (§3) |
| **Diferencias finitas** | Comprobar un gradiente moviendo cada peso un poquito y viendo cuánto cambia la pérdida |
| **Early stopping** | Parar cuando la métrica en val deja de mejorar y quedarse con la mejor época |
| **Entropía ($\mathcal H$)** | Qué tan "repartida" está la probabilidad. Alta = explora; baja = casi siempre lo mismo |
| **Episodio** | Aquí: una batalla. Dos decisiones (líderes, reservas) y una recompensa al final |
| **Exploración–explotación** | Probar brings nuevos frente a usar el que ya parece mejor |
| **Feature** | Un número que describe algo del estado (p. ej. la tasa de uso de una especie) |
| **$\gamma$ (descuento)** | Cuánto valen las recompensas futuras. Aquí $\gamma=1$ y no cambia nada |
| **Greedy** | Elegir siempre la opción más probable, sin sortear |
| **Intervalo t** | Intervalo de confianza para una media con pocas unidades (aquí, pares) |
| **KL** | Divergencia de Kullback-Leibler: cuánto se aleja el agente de $\rho_{BC}$ |
| **L2** | Penalización al tamaño de los pesos para no sobreajustar |
| **Leakage** | Que el modelo vea información que no tendría al decidir. Penalización de −15 |
| **Líderes / reservas** | Los 2 que empiezan en el campo / los 2 que esperan atrás |
| **LOO** | *Leave-one-out*: al calcular una feature de una batalla de train se excluye esa batalla |
| **Log set** | Los registros de batallas humanas de VGC-Bench (Reg M-A) |
| **MDP** | *Markov Decision Process*: estados, acciones, transiciones, recompensa y descuento |
| **MetodoBring** | La clase padre de todos los métodos (`models/base.py`) |
| **NLL** | Pérdida: cuanto más baja, más probabilidad le da el modelo a lo que pasó. Al azar sería $\ln 90 \approx 4{,}50$ |
| **On-policy** | Se aprende solo con datos de la política actual |
| **One-hot** | Vector con un 1 en la posición de la especie y 0 en las demás |
| **OTS** | *Open Team Sheets*: las hojas de equipo abiertas, visibles para los dos jugadores |
| **Par de equipos** | Dos equipos concretos que se enfrentan; la unidad de evaluación |
| **$\pi_b$** | La política de batalla fija (SimpleHeuristicsPlayer + auto_mega), igual en los dos lados |
| **Planificador-$\hat v$** | Baseline que prueba los 90 brings con $\hat v$ y elige el mejor |
| **poke-env** | Librería de Python para conectar bots con Showdown (el fork de vgc-bench) |
| **Preregistro** | Escribir la regla de decisión antes de ver test |
| **Regresión logística** | Modelo lineal que da una probabilidad con $\sigma(w\cdot x)$ (aquí, $\hat v$) |
| **REINFORCE** | Gradiente de política: sube la probabilidad de las acciones que dieron más recompensa de lo esperado |
| **Reg M-A** | El reglamento de VGC del log set (`gen9championsvgc2026regma`) |
| **RL** | Aprendizaje por refuerzo: aprender a decidir probando y recibiendo recompensas |
| **RQ** | La pregunta de investigación |
| **$\rho_{BC}$** | La política entrenada por imitación |
| **Semilla** | Número que fija el azar de una corrida |
| **Showdown** | El simulador de batallas; aquí corre en local (commit `913da36`) |
| **Smoke test** | Corrida mínima para ver que todo funciona. No es un resultado |
| **Softmax lineal** | Puntajes lineales en las features convertidos en probabilidades con softmax (aquí, $\rho_{BC}$ y el agente) |
| **Species Clause** | Prohíbe dos Pokémon con el mismo número de Pokédex en un equipo |
| **$T$** | *Throughput*: batallas (o episodios) por segundo |
| **Team Preview** | Antes de la batalla: se ven los dos equipos y se elige el bring |
| **Train / val / test** | Entrenar / elegir hiperparámetros y checkpoints / evaluación final, una sola vez |
| **Usage** | El baseline que elige como la población |
| **$v$** | Número de reservas reveladas en un log (0, 1 o 2) |
| **$\hat v$** | Evaluador de matchup: probabilidad de ganar de un bring contra otro |
| **Validador** | El `TeamValidator` de Showdown: dice si un equipo es legal |
| **Wilson** | Intervalo de confianza para una proporción, bueno también cerca de 0 o 1 |
| **Win rate** | Fracción de batallas ganadas |
