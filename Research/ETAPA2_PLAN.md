# Etapa 2: plan de implementación (BringPredictorVGC)

> **Nota 14-11-2026:** esta v1 es histórica. Manda la v2 (`ETAPA2_PLAN_v2.md`); el código usa una política lineal en numpy, no Transformer, DeepSets ni torch.

**Proyecto:** *The state of ML in Pokémon: the benefits of a bring predictor*. Track C (Aprendizaje por refuerzo)
**Equipo:** Sergio Zúñiga Castillo · José Ignacio Madriz Ortiz
**Entrega:** Etapa 2, 45 % del proyecto. Fecha: la que esté publicada en Discord.
**Fuentes:** `_IAR_2026S2_PY01_TrackRL.pdf` (enunciado), `Pokemon_AI_Survey (1).pdf` (Etapa 1) y el log set `logs_gen9championsvgc2026regma.json` (medido otra vez para este documento).

> Este plan es un borrador hecho con asistencia de IA. Cada decisión que aparece aquí la tienen que poder defender los dos (Principio de Defensa Técnica). Cuando no estén de acuerdo con algo, cámbienlo y anoten el porqué. Ese razonamiento es lo que se evalúa.

---

## 0. Resumen ejecutivo

La Etapa 2 pide cinco cosas: (1) un MDP declarado y justificado, (2) al menos dos baselines (una aleatoria y una heurística o un algoritmo más simple), (3) un entrenamiento documentado, (4) al menos 3 semillas con media ± dispersión en todas las curvas y tablas, y (5) un análisis del comportamiento aprendido que incluya exploración–explotación.

El survey de la Etapa 1 ya resuelve casi todo el diseño, pero tiene **un hueco que el profesor va a señalar**. La política se entrena con la pérdida de etiquetas parciales (7), y eso es **imitación supervisada (behaviour cloning), no RL**. No hay recompensa en el gradiente, no hay exploración y no hay nada que analizar en el trade-off exploración–explotación. El plan resuelve ese hueco sin cambiar la pregunta de investigación:

1. **Fase offline (supervisada, a partir de logs humanos):**
   - un predictor de bring humano $\rho_{BC}$, entrenado con (7), que hace dos trabajos: predecir el bring del rival e inicializar la política del agente;
   - el evaluador de matchup $\hat v$, entrenado con (11).
2. **Fase online (RL de verdad):** la política inicializada con BC se afina con **REINFORCE con baseline**, regularizado con KL hacia $\rho_{BC}$, sobre el MDP de 2 pasos. Cada episodio es **una batalla real** en un Showdown local, con una política de batalla fija para ambos lados y el baseline de *usage* como rival. Es la receta BC→RL que Angliss et al. [4] reportan como dominante.
3. **Evaluación:** win rate en batallas jugadas, nunca con $\hat v$. Se usan pares de equipos de test que no se vieron en el entrenamiento, 1000 batallas por par con los lados intercambiados, 5 semillas de entrenamiento y el umbral del 55 %.

---

## 1. Qué exige el enunciado (y dónde se cubre en este plan)

| Rubro (puntos) | Exigencia concreta | Sección |
|---|---|---|
| Formulación del MDP (10) | Estados, **qué información se le oculta al agente**, acciones (cardinalidad), recompensa exacta, reward shaping justificado si lo hay, terminación y horizonte, $\gamma$ justificado | §4 |
| Baselines (7) | Política aleatoria uniforme, más una heurística o un algoritmo más simple, comparadas con el agente **y** con el baseline de literatura | §8 |
| Entrenamiento (8) | Algoritmo justificado para el tipo de espacio de estados y acciones; hiperparámetros (lr, buffer, exploración y su decaimiento, red objetivo); episodios o pasos; **hardware y tiempo** | §7, §9.4 |
| Evaluación multi-semilla (7) | ≥ 3 semillas; media y dispersión **en todas** las curvas y tablas | §9 |
| Análisis del comportamiento (7) | Recompensa por episodio, tasa de éxito, longitud de episodio con curvas; descripción cualitativa; casos de falla; exploración–explotación | §10 |
| Profesionalidad (6) | Repo limpio, semillas, configs, README, **commits de ambos**, docs en Markdown, declaración de uso de IA | §11 |

**Penalizaciones transversales que aplican aquí:**

- *Leakage* (−15). En este track, un MDP mal especificado cuenta como leakage. Los riesgos concretos están en §4.2 y §6.
- Ausencia de baselines (−10).
- Decisiones sin justificar (−5 cada una, hasta −15). Las decisiones críticas son la recompensa, la partición, la métrica y la arquitectura.
- No reproducible (hasta −10).
- Citas que no se leyeron (−10).

**Módulo adicional:** no aplica, porque el grupo es de dos.

---

## 2. Qué se hereda de la Etapa 1 (no se toca)

- **RQ:** ¿Puede una política de RL, viendo los dos equipos completos en Team Preview, elegir un bring (4 de 6, con el par líder designado) que logre **≥ 55 %** de victorias contra un **baseline de population usage**, en **1000 batallas por par de equipos** jugadas en ambas direcciones y con **la misma política de batalla fija** en los dos lados?
- **Espacio de acciones:** $|B| = \binom{6}{2}\binom{4}{2} = 15 \times 6 = 90$, factorizado en dos pasos (líderes y luego reservas).
- **Recompensa:** dispersa y terminal, $\{-1,+1\}$, sin shaping.
- **Datos:** el log set de Reg M-A de VGC-Bench: 6 905 batallas y 13 810 player-brings. El par líder se observa en el 100 % de los casos. Del bring completo, el 84,1 % tiene $v=2$, el 12,0 % $v=1$ y el 3,9 % $v=0$. **La revelación está correlacionada con el resultado** (92,8 % de los perdedores revelan completo, contra 73,2 % de los ganadores).
- **Supervisión parcial:** la pérdida (7) sobre el conjunto consistente $C(o)$ y la verosimilitud marginalizada (11) para $\hat v$. **No se descarta ningún registro.**
- **$\hat v$:** el modelo bilineal antisimétrico (8)–(10), adaptado de Oblander [15] al nivel del bring (proyecciones distintas para líderes y reservas).
- **Protocolo:** partición temporal, pares de equipos de validación y test no vistos, sides swap [5], 1000 batallas por matchup [4], diagnósticos (lead-pair exact match, per-slot recall, full-bring exact match estricto y permisivo).
- **Fuera de alcance:** team building, entrenar una política de batalla, LLMs y juego superhumano.

---

## 3. Ajustes necesarios respecto a la Etapa 1

Hay que declarar estos cambios en el artículo como *refinamientos de la metodología*. Esconderlos sería peor que explicarlos.

| # | Lo que dice la Etapa 1 | Problema | Ajuste |
|---|---|---|---|
| A1 | La política "se entrena sobre el MDP bajo (7)" | (7) es una pérdida de imitación: no usa $R_B$, no explora y no hay trade-off que analizar. El rubro de entrenamiento y el de exploración–explotación quedarían vacíos | La política BC pasa a ser la **inicialización**. Se añade una **fase de RL online** con REINFORCE, usando como recompensa batallas reales (§7.5). BC queda como baseline-ablación: "¿aportó algo el RL sobre la imitación?" |
| A2 | Sección III-B: "ningún proxy aprendido sustituye a $R_B$". Sección III-F: "$\hat v$ … usado para rankear candidatos durante el entrenamiento" | Las dos frases se contradicen | Se fija esta regla: **$\hat v$ nunca es la recompensa del agente principal**. $\hat v$ se usa (i) en el baseline *Planificador-$\hat v$* (§8), (ii) como diagnóstico de distribution shift y (iii) solo como plan B de recompensa si el throughput del simulador no alcanza (§13, R2). Si se activa el plan B, se reporta |
| A3 | Baseline GA sobre los 90 candidatos con fitness $\hat v$ | Con 90 acciones, **enumerar** las 90 (o los 90×90 = 8 100 pares) es exacto y más barato que un GA. Un GA solo puede igualar ese óptimo | Se sustituye por el **argmax exhaustivo** de la misma fitness (Planificador-$\hat v$). Se justifica en el paper: es el techo de cualquier búsqueda sobre esa fitness |
| A4 | "Search builders" (Tabla IV) adaptados al bring | Costo alto y poca relación directa con el problema | Se eliminan con esta justificación. Se mantienen Random, Usage, Heurística de tipos, Q-tabular, BC y Planificador-$\hat v$ |
| A5 | "Al menos diez semillas" (III-H.4) | Con el presupuesto de batallas, 10 semillas × RL es caro (§9.4) | **5 semillas de entrenamiento** (el enunciado pide ≥ 3). El argumento de Li & Wellman [14] se respeta reportando distribuciones sobre semillas **y** sobre pares de equipos |
| A6 | Longitud de episodio como métrica (la pide el enunciado) | En este MDP la longitud es **siempre 2**, así que la métrica es trivial | Se declara así. Como diagnóstico se reporta la **duración de la batalla en turnos**, que nunca es feature ni recompensa |
| A7 | Partición temporal | 2 476 batallas pertenecen a **1 245 series Bo3**. Los juegos de una misma serie comparten equipos y jugadores, así que una serie partida entre train y test es **leakage** | La unidad de partición es la **serie** (o la batalla, si está suelta), asignada por el timestamp de su primer juego |
| A8 | "Totalmente observado salvo la asignación de stats oculta" | Para **simular** hacen falta stats. En el log, `showteam` trae vacío el campo de stats y de naturaleza | Se **imputa** una asignación fija y documentada, idéntica para los dos lados (no introduce sesgo entre métodos). Va como limitación |

---

## 4. Formulación definitiva del MDP (rubro de 10 pts)

### 4.1 Definición

Se fijan la política de bring del rival $\rho_{opp}$ (el baseline Usage, en modo estocástico) y una política de batalla común $\pi_b$. Con eso, el juego $G(c_1,c_2)$ se reduce a un MDP de un solo agente con horizonte finito $H=2$:

$$\mathcal M = \langle S_B, A_B, T_B, R_B, \gamma \rangle$$

**Estado** $s_t = \langle c_{self}, c_{opp}, m, \ell_t \rangle$:

- $c_{self}, c_{opp}$: las dos hojas de equipo OTS. Cada Pokémon trae especie (con forma normalizada), 4 movimientos, objeto (las megapiedras entran aquí) y habilidad.
- $m$: features de metajuego **calculadas solo en la ventana de train**. Son tasas de uso, de líder y de bring por especie, y se añaden como features por Pokémon, no como un vector global, para que generalicen a equipos nuevos.
- (opcional, ablación) $\hat q_{opp}$: la distribución predicha del bring rival por $\rho_{BC}$, con $P(\text{líder})$ y $P(\text{traído})$ para cada uno de los 6 rivales. Esta feature hace explícito el objetivo 2 ("estima el bring rival y responde").
- $\ell_t$: el par líder ya comprometido ($\varnothing$ en $t=0$).

**Acciones** (discretas, con máscara de legalidad):

- $t=0$: $a_0 \in \binom{c_{self}}{2}$, el par líder (15 opciones).
- $t=1$: $a_1 \in \binom{c_{self}\setminus a_0}{2}$, las reservas (6 opciones).
- Bring $b=(a_0,a_1)$, con $|A_B| = 90$. La política es **autorregresiva**: $\pi(b|s)=\pi(a_0|s)\,\pi(a_1|s,a_0)$.

**Transiciones:**

- $t=0 \to 1$ es determinista: $\ell \leftarrow a_0$.
- $t=1 \to$ terminal es estocástica: los dos brings se mandan a Showdown y la batalla se juega completa con $\pi_b$ en ambos lados. La estocasticidad viene de las tiradas de daño, la precisión, los efectos secundarios, los speed ties, la política estocástica del rival y $\pi_b$ si no es determinista.

**Recompensa:**

- $R_B(s_0,a_0)=0$.
- $R_B(s_1,a_1) = +1$ si se gana, $-1$ si se pierde y $0$ si hay empate.
- El empate incluye tanto el empate real como llegar al tope de turnos (**fijar el tope, p. ej. 50 turnos**; la mediana real es 7 y el p90 es 11). Hay que reportar cuántas veces pasa.
- Si una batalla **falla por error del simulador**, se descarta y se registra. No cuenta como derrota. Hay que reportar la tasa de fallos.

**Terminación:** después de $a_1$, siempre. El horizonte es $H=2$.

**Descuento:** $\gamma=1$. Justificación: el horizonte es finito y la recompensa es solo terminal. Cualquier $\gamma\in(0,1]$ escala el retorno en $t=0$ por $\gamma$ y **no cambia la política óptima**. Con $\gamma=1$, el valor se interpreta directamente: $V(s)=2P(\text{win})-1$, y eso enlaza con (12): $\hat v \approx (E[R_B]+1)/2$.

**Sin reward shaping.** Se mantienen los argumentos del survey:

- el horizonte de 2 pasos no deja un hueco de asignación de crédito que haya que puentear;
- el shaping fue explotado en [3] y [10];
- la recompensa dispersa mide exactamente lo que pregunta la RQ.

Si se prueba algún término de shaping, va como ablación y no se adopta sin evidencia.

### 4.2 Información que se le **niega** al agente (y por qué)

El enunciado pide esta lista explícitamente. Es el equivalente a la sección de "prevención de leakage".

| Información | Por qué se excluye |
|---|---|
| El bring del rival en *este* episodio | Es simultáneo y oculto en el juego real. Dárselo sería target leakage |
| Número de Pokémon revelados, duración de la batalla, turnos | Son proxies del resultado (por la correlación revelación–victoria). Prohibidos como features en **todos** los modelos |
| Resultado, rating, nombres de usuario | Leakage o información ajena a la decisión |
| Historial Bo3 (número de juego, brings de juegos anteriores) | Mantiene el MDP markoviano y comparable entre batallas. Es una limitación: los humanos sí se adaptan entre juegos. Puede ser trabajo futuro |
| Stats y naturaleza reales | OTS no los revela. Se imputan igual para ambos lados (A8) |
| Estadísticas de uso calculadas con datos de validación o test | Serían leakage temporal |

### 4.3 Alternativas descartadas (pregunta garantizada en la defensa)

1. **Bandido contextual de un paso con 90 brazos.** Es equivalente en el conjunto de políticas alcanzables. Se prefiere la factorización 15×6 porque baja el branching por paso, da una política autorregresiva más fácil de parametrizar y separa la decisión de líderes de la de reservas, que es donde opera el *counter-selection* [16].
2. **Solo el par líder (15 acciones).** Se descarta por los cinco argumentos de III-D. La decisión real es el bring completo.
3. **Meter el Team Preview en el MDP de batalla, como VGC-Bench.** Obliga a entrenar una política de batalla (fuera de alcance) y no aísla la contribución del bring, que es justamente el gap del proyecto.
4. **Juego de Markov de dos jugadores con solución de equilibrio.** Exige resolver un equilibrio sobre 90×90 por par con payoffs ruidosos. Se fija al rival en el entorno y la explotabilidad se mide aparte (Etapa 3).
5. **Recompensa = $\hat v$.** Sería barato, pero el agente optimizaría un proxy aprendido y podría explotar sus errores. Además, evaluar con lo mismo que se optimiza es circular. Queda solo como plan B (§13).

---

## 5. Arquitectura del sistema

```
 logs Reg M-A ──► parser ──► registros (c1, c2, o1, o2, w, t, serie)
                                │
                partición temporal por serie (§6)
                                │
     ┌──────────────┬───────────┴───────────┬───────────────────┐
     ▼              ▼                       ▼                   ▼
  usage stats    ρ_BC (pérdida 7)        v̂ (pérdida 11)      pares de equipos
  (solo train)   predictor humano        evaluador            train / val / test
     │              │  ├─► predice bring rival (diagnósticos)
     │              │  └─► inicializa política del agente
     ▼              ▼
  Baseline Usage   Agente RL (REINFORCE + KL a ρ_BC)
  = rival en el MDP          │  episodios = batallas reales
     │                       ▼
     └──────► ARENA: Showdown local + poke-env (fork VGC-Bench)
              BringPlayer.teampreview() = política de bring
              choose_move() = π_b fija (igual ambos lados)
                             │
                             ▼
              resultados CSV ► métricas, IC, curvas, tablas
```

**Interfaz clave.** En poke-env, la política de bring se implementa sobrescribiendo `teampreview(battle)`, que devuelve `/team abcd`: las dos primeras posiciones son los líderes y las dos siguientes las reservas. La política de batalla es la heredada de `choose_move`. Así, **cada método (agente o baseline) es una subclase que solo cambia `teampreview`**, y $\pi_b$ es idéntica por construcción. Ese es el control experimental central.

**Política de batalla $\pi_b$:** el `SimpleHeuristicsPlayer` de poke-env extendido a dobles por VGC-Bench [4]. Se elige porque:

- es barato (CPU), así que permite muchas batallas;
- es determinista o casi determinista;
- está publicado, lo que lo hace reproducible y citable.

Para la Etapa 3 queda un $\pi_b$ alternativo (MaxBasePower o el agente BC de VGC-Bench), que sirve como "entorno perturbado".

---

## 6. Datos y particiones

**Cifras medidas en el log set** (re-medidas para este plan):

- Rango temporal: 2026-04-11 → 2026-06-20.
- 1 245 series Bo3, que suman 2 476 batallas. El resto son batallas sueltas.
- 5 630 equipos distintos (por conjunto de especies). Solo 495 aparecen 5 veces o más.
- 5 451 pares de equipos distintos (el más repetido aparece 14 veces).
- 276 especies. Las más usadas: Sneasler, Garchomp, Basculegion, Incineroar y Kingambit.
- Duración: mediana de 7 turnos, p90 de 11.

**Partición temporal por serie:**

| Conjunto | Ventana (aprox., por cuantiles) | Uso |
|---|---|---|
| Train | 2026-04-11 → ~2026-05-25 (≈ 70 %) | $\rho_{BC}$, $\hat v$, usage stats, Q-tabular, pares de equipos para el RL |
| Val | ~2026-05-25 → ~2026-06-05 (≈ 15 %) | Early stopping, hiperparámetros, elección del checkpoint de RL |
| Test | ~2026-06-05 → 2026-06-20 (≈ 15 %) | **Solo al final**: diagnósticos sobre logs y pares de equipos para las batallas de la RQ |

Los cortes exactos se fijan en `configs/data.yaml` **antes** de mirar resultados y no se mueven.

**Pares de equipos para simular:**

- **Train-RL:** ~200 pares muestreados de la ventana de train (con lados aleatorios en cada episodio).
- **Val:** 10 pares.
- **Test:** **20 pares**, tomados de la ventana de test y con **ninguno de sus dos equipos presente en train** (comparando el conjunto de especies y los sets). La generalización a equipos no vistos sigue [4], [8]. Muestrear estratificado por "arquetipo" (p. ej. Trick Room, lluvia, sol, Tailwind) para que haya variedad. **Congelar la lista** en un archivo versionado.

**Normalización del parser:** formas (Mega, Floette-Eternal, Rotom-Wash, etc.), apodos e Illusion. El survey encontró 2 registros inconsistentes (0,014 %), que se excluyen. Hay que **verificar el parser contra una muestra anotada a mano** (p. ej. 50 batallas, revisadas en el replay) y guardar esa muestra como test unitario.

**Imputación de stats (A8):** decidir una regla y documentarla. Opciones: (a) una asignación "estándar" según el rol (atacante físico, especial, soporte), inferida de los movimientos; (b) una asignación neutra fija. Se aplica igual a los dos lados y a todos los métodos. Antes, verificar qué hace el simulador cuando el campo viene vacío.

---

## 7. Componentes y entrenamiento

### 7.1 Codificador compartido

- **Embedding por Pokémon** $\phi_x$:
  - especie (d=32), objeto (16), habilidad (16) y movimientos (16 cada uno, promediados);
  - las features de metajuego $m_x$;
  - un MLP que lleva todo eso a d=64.
- **Codificador de matchup:** un Transformer pequeño (2 capas, 4 cabezas) sobre 12 tokens (6 propios y 6 rivales) con embedding de lado. Así cada Pokémon propio "ve" al equipo rival. Alternativa más simple: DeepSets.

### 7.2 Política autorregresiva

- **Cabeza de líderes:** un score para cada uno de los 15 pares, $\text{MLP}(h_i + h_j)$, con softmax.
- **Cabeza de reservas:** condicionada en el par elegido. Da un score a cada uno de los 6 pares restantes, con softmax.
- **Máscara de legalidad** en las dos cabezas. Test unitario: hay exactamente 90 brings alcanzables y todos son legales.

### 7.3 $\rho_{BC}$: imitación con etiquetas parciales (pérdida 7)

- Datos: todos los player-brings de train (13 810 × ~70 %). Cada registro aporta $(c_{self}, c_{opp}, o)$, y se usan **los dos lados** de cada batalla.
- Pérdida: $-\log \sum_{b\in C(o)} \pi(b|s)$. Es CE exacta cuando $|C(o)|=1$.
- Usos: (i) **predictor del bring rival**, llamando $\rho_{BC}(\cdot \mid c_{opp}, c_{self})$ con los argumentos invertidos; (ii) **inicialización** del agente RL; (iii) baseline-ablación "BC".
- Hiperparámetros iniciales: Adam, lr 1e-3, batch 256, weight decay 1e-4, dropout 0.1, ≤ 30 épocas con early stopping sobre la pérdida parcial en val.
- Se entrena con las **5 semillas**.

### 7.4 $\hat v$: evaluador de matchup (pérdida 11)

- Forma: (8)–(10), con $K=16$, $M=A-A^\top$ (antisimetría por construcción) y proyecciones distintas para líderes y reservas.
- Pérdida: log-verosimilitud marginalizada sobre $C(o_1)\times C(o_2)$ (≤ 36 términos), con $q$ uniforme.
- Tests: $\hat v(b_1,b_2)+\hat v(b_2,b_1)=1$ (exacto); invariancia al orden dentro de cada par.
- Métrica: accuracy y AUC en test. Es la **única comparación numérica con la literatura**: Oblander, 66,43 % y 0,730 [15]. Hay que dejar claras las diferencias: brings frente a equipos, otro dataset y sin término de habilidad.

### 7.5 Agente: REINFORCE con baseline y KL a $\rho_{BC}$ (el RL)

**Por qué este algoritmo** (justificación para el rubro):

- Los episodios duran 2 pasos con recompensa terminal y $\gamma=1$, así que el retorno de Monte Carlo es **exacto, insesgado y sin bootstrapping**. Los métodos de TD (DQN, A2C con bootstrap) no aportan nada aquí y suman sesgo e inestabilidad.
- Las acciones son discretas y pocas (15 y 6) y la política es estocástica con máscara. El policy gradient sobre un softmax es lo natural.
- El estado es **de alta dimensión y combinatorio** (pares de equipos casi nunca repetidos), así que hace falta aproximación de funciones. El Q tabular no generaliza, y por eso es el baseline "más simple" (§8).
- La inicialización con BC sigue la evidencia de [4]: las variantes inicializadas con BC dominan a las entrenadas desde cero.
- **No hay replay buffer ni red objetivo**: el método es on-policy con retornos de MC y no tiene targets que estabilizar. Hay que decirlo explícitamente, porque el enunciado los menciona como hiperparámetros esperados.

**Objetivo:**

$$\mathcal L(\theta) = -\mathbb E\big[(R - V_\psi(s_0))\,\log\pi_\theta(b|s)\big] \;-\; \beta_H\, \mathcal H[\pi_\theta(\cdot|s)] \;+\; \beta_{KL}\, \mathrm{KL}\big(\pi_\theta(\cdot|s)\,\|\,\rho_{BC}(\cdot|s)\big)$$

- $V_\psi$ es un baseline de valor aprendido (cabeza lineal sobre el codificador, MSE contra $R$). Reduce la varianza sin introducir sesgo.
- El término KL a $\rho_{BC}$ cumple dos funciones: estabiliza el entrenamiento y **reduce que el agente explote las debilidades de $\pi_b$** (§10.4).

**Bucle de entrenamiento:**

1. Muestrear un par de train y un lado.
2. El agente muestrea $b\sim\pi_\theta$ y el rival muestrea su bring con Usage (estocástico).
3. Jugar la batalla con $\pi_b$ en los dos lados.
4. Calcular $R$, acumular un batch y hacer un paso de gradiente.
5. Cada $N$ episodios, evaluar en val (200 batallas por par, de forma greedy) y guardar checkpoint.
6. El checkpoint final se elige por **win rate en val**, nunca en test.

**Hiperparámetros iniciales** (ajustar solo con val):

| Parámetro | Valor inicial | Notas |
|---|---|---|
| Optimizador / lr | Adam, 1e-4 (política), 1e-3 ($V_\psi$) | Clip de gradiente 1.0 |
| Batch | 128 episodios por actualización | Las batallas corren en paralelo (concurrencia de poke-env) |
| $\beta_H$ (entropía) | 0.01 | Barrido {0, 0.01, 0.05} para el análisis de exploración |
| $\beta_{KL}$ | 0.1 | Ablación {0, 0.1} si el presupuesto alcanza |
| Presupuesto | **50 000 episodios por semilla** (provisional) | Se ajusta con el throughput medido (§9.4) |
| Semillas | 5 | Controlan la init de la cabeza de valor, el muestreo de pares y lados, y el muestreo de acciones |
| Evaluación | Greedy (argmax autorregresivo) | También se reporta con muestreo |

### 7.6 Baseline "algoritmo más simple": Q tabular (bandido ε-greedy por par)

- Para cada par de train y lado, una tabla $Q(b)$ de 90 entradas, con $\alpha = 1/n$ (promedio muestral) y ε lineal de 1.0 a 0.05.
- Se entrena en pocos pares (p. ej. 5 pares × 2 lados × 2 000 episodios × 3 semillas).
- **No generaliza a pares nuevos**: en los pares de test cae en Random (o en Usage si se usa como prior). Eso es justamente lo que demuestra: el valor de la aproximación de funciones.
- Da una curva de **regret** y un análisis directo del decaimiento de ε (exploración–explotación).

---

## 8. Baselines (rubro de 7 pts)

| Baseline | Definición operativa | Papel |
|---|---|---|
| **Random** (obligatorio) | Uniforme sobre los 90 brings legales | "¿Aprendió algo?" Piso de azar: 1,11 % exact match, 6,67 % lead pair. Equivale al toggle de bring aleatorio de VGC-Bench [4] |
| **Usage** (heurística, rival de la RQ) | $\text{score}(b)=\sum_{x\in L}\log P(\text{lead}\mid x)+\sum_{x\in R}\log P(\text{back}\mid x)+\sum_{x\notin b}\log P(\text{fuera}\mid x)$, con frecuencias de train y **cuentas fraccionales** sobre $C(o)$. Modo estocástico (softmax del score) como rival; modo argmax como baseline | "¿Bastaba copiar a la población?" Humanos: 59,3 % de peso en el uso poblacional [15]. Si no se definía bien la "condición por arquetipo" del survey, esto la concreta y generaliza a equipos nuevos |
| **Heurística de tipos** | Para cada Pokémon propio: (mejor efectividad de sus movimientos contra el equipo rival) − (mejor efectividad STAB rival contra él). Se traen los 4 mejores y lideran los 2 mejores | Una regla de dominio escrita a mano. Simões [6] la reporta como subóptima |
| **Q tabular** (§7.6) | Bandido ε-greedy por par | "Algoritmo más simple" en el sentido del enunciado |
| **BC** ($\rho_{BC}$ greedy) | La imitación sin RL | **Ablación clave:** ¿el RL aportó algo sobre imitar? |
| **Planificador-$\hat v$** | $\arg\max_b \sum_{b_2} q_{opp}(b_2)\,\hat v(b,b_2)$, con la distribución del rival Usage. Enumeración exacta, que sustituye al GA (A3) | ¿Basta planificar con el modelo aprendido o hace falta aprender con batallas reales? |

**Baseline de literatura** (la tercera referencia que pide el enunciado):

- No hay un número publicado para esta cantidad. Se reporta la **Tabla III** del survey como contexto: el umbral de dominancia del 55 % [5], el rango 0,33–0,74 de VGC-Bench [4] y los pisos de azar.
- Se compara directamente lo único comparable: **$\hat v$ contra Oblander** (accuracy y AUC), con sus salvedades.

**Sanity checks obligatorios** (van en el apéndice o en un párrafo):

- Usage contra Usage ≈ 50 % (valida el sides swap y la simetría del arena).
- Random contra Usage < 50 %.
- El mismo bring en los dos lados del mismo par da ≈ 50 %.

---

## 9. Protocolo de evaluación (rubro de 7 pts)

### 9.1 Batallas de la RQ

- Por cada par de test $(c_A,c_B)$, se juegan **500 batallas con el método en $c_A$ contra Usage en $c_B$, y 500 con los lados invertidos**. Son **1000 por par** [4], con el swap de [5].
- Se aplica al **agente RL** y a **BC** en sus 5 semillas, y una vez a los baselines deterministas (Random, Tipos, Planificador-$\hat v$). Q tabular solo se reporta en sus pares de entrenamiento.

### 9.2 Estadística y regla de decisión (fijarla antes de ver test)

- Win rate por par con IC de Wilson al 95 % (con $n=1000$, el ±IC ronda 0,03, coherente con [4]).
- **Agregado:** media sobre pares por semilla, y después **media ± desviación estándar entre las 5 semillas**. Además, un IC por bootstrap jerárquico (remuestreando semillas y luego pares).
- **Regla para la hipótesis:** se afirma "supera el 55 %" solo si el **límite inferior del IC agregado es > 0,55**. Si el punto supera 0,55 pero el IC lo cruza, el resultado es "compatible, no concluyente". Si no, se rechaza. Además, contraste pareado agente-vs-BC por par (test de signos o de Wilcoxon sobre los 20 pares).
- La regla se escribe en `docs/EXPERIMENTS.md` **antes** de correr test, como preregistro informal. Cuenta como evidencia de rigor en la defensa.

### 9.3 Diagnósticos sobre logs de test (predictor del rival, $\rho_{BC}$)

- **Lead-pair exact match** (15 alternativas, piso 6,67 %), sobre el 100 % de las batallas.
- **Per-slot recall** de los Pokémon rivales revelados, sobre el 100 % de las batallas.
- **Full-bring exact match** (piso 1,11 %), en dos versiones: *estricto* en el subconjunto completo (avisando que está sesgado hacia perdedores) y *permisivo* (acierto si $\hat b\in C(o)$).
- Top-3 de los anteriores (opcional).
- Todo como media ± desviación estándar sobre las 5 semillas de $\rho_{BC}$.

### 9.4 Presupuesto de cómputo (medirlo en la Fase 0)

Con un throughput de $T$ batallas/s (concurrencia local, $\pi_b$ heurística):

| Bloque | Batallas | Horas con T = 10 | Horas con T = 3 |
|---|---|---|---|
| RL: 5 semillas × 50k | 250 000 | 6,9 | 23 |
| Eval de val durante el RL (5 × 10 checkpoints × 10 pares × 200) | 100 000 | 2,8 | 9,3 |
| Q tabular: 3 × 5 pares × 2 lados × 2k | 60 000 | 1,7 | 5,6 |
| Test de la RQ: (5 RL + 5 BC) × 20 pares × 1000 | 200 000 | 5,6 | 18,5 |
| Test de los baselines deterministas: 3 × 20 × 1000 | 60 000 | 1,7 | 5,6 |
| Barrido de entropía (3 valores × 3 semillas × 15k, corto) | 135 000 | 3,8 | 12,5 |
| **Total** | **≈ 805 000** | **≈ 22 h** | **≈ 75 h** |

- **Go/no-go:** si $T < 3$, recortar el barrido, el test de BC a 3 semillas y la eval de val a 5 pares, o activar el plan B (§13, R2).
- Repartir las corridas entre las **dos computadoras** del equipo (las semillas son independientes).
- **Checkpoints cada N episodios y reanudación** obligatoria. Una de las PCs ha tenido pantallazos azules recientes y no conviene perder una corrida de 7 horas.
- En el artículo hay que **registrar el hardware (CPU, RAM, GPU si se usó) y el tiempo de pared** de cada bloque. Lo pide el rubro.

---

## 10. Análisis del comportamiento (rubro de 7 pts)

### 10.1 Métricas y curvas (todas con banda de ±1 desviación estándar entre semillas)

- **Recompensa media por episodio** contra los episodios de entrenamiento (= $2\cdot WR - 1$ menos el efecto de los empates).
- **Tasa de éxito** (win rate) en val contra los episodios.
- **Longitud de episodio:** constante (= 2). Se dice así, y en su lugar se grafica la **duración en turnos** de las batallas ganadas y perdidas como diagnóstico.
- La **entropía de la política** contra los episodios.
- **Calibración de $\hat v$:** el $\hat v$ predicho contra el win rate observado por par. Cuantifica el distribution shift entre humanos y $\pi_b$ (riesgo 6 del survey). Si no coinciden, eso también es un resultado.

### 10.2 Descripción cualitativa de la política

- Para cada par de test, una tabla con el bring del agente, el de Usage, el humano más frecuente (si existe) y el win rate de cada uno.
- Patrones a buscar y nombrar:
  - ¿Evita liderar contra Fake Out + Intimidate?
  - ¿Trae respuestas a Trick Room (Taunt, Imprison, atacantes lentos)?
  - ¿Responde a la lluvia o al sol con su propio clima?
  - ¿Reserva el counter del líder rival más probable? Es el *counter-selection* de [16].
- **Comparar con $\hat q_{opp}$:** ¿cambia el agente su bring cuando cambia el bring rival predicho? Una prueba concreta es una perturbación controlada: cambiar un Pokémon del rival y observar la respuesta.

### 10.3 Casos de falla

- Los pares donde el agente queda **por debajo** de Usage o de BC: ¿qué tienen en común (arquetipo, especies raras con poco soporte en train)?
- Los brings con baja frecuencia en train (la cola de los 90): ¿se equivoca más ahí?

### 10.4 Exploración–explotación (lo pide el enunciado explícitamente)

- **Cómo se configuró:** exploración intrínseca de la política estocástica, con la bonificación de entropía $\beta_H$ y el ancla KL hacia $\rho_{BC}$, que limita cuánto explora lejos del prior humano. En el Q tabular, ε-greedy con decaimiento lineal.
- **Qué se observó:**
  1. la curva de entropía y su colapso;
  2. el barrido $\beta_H\in\{0, 0.01, 0.05\}$ (win rate final, varianza entre semillas, velocidad);
  3. el número de brings distintos probados por par a lo largo del entrenamiento;
  4. el regret del Q tabular según el schedule de ε;
  5. (opcional) BC-init frente a init aleatoria.
- **Riesgo que hay que discutir** (enlaza con la Etapa 3): con $\beta_{KL}=0$, el agente puede encontrar brings que **explotan las debilidades de $\pi_b$** en lugar de ser buenos en general. Esa es la especificación defectuosa típica de este problema. Se detecta en la Etapa 3 evaluando con otro $\pi_b$.

---

## 11. Reproducibilidad y repositorio (rubro de 6 pts)

```
BringPredictorVGC/
├── README.md                 # qué es, cómo instalar, cómo reproducir cada tabla/figura
├── requirements.txt          # versiones fijadas; poke-env del fork VGC-Bench por commit hash
├── configs/                  # data.yaml, bc.yaml, vhat.yaml, rl.yaml, tabular.yaml, eval.yaml
├── data/                     # .gitignore (el JSON pesa ~50 MB) + script de descarga desde HF
├── splits/                   # cortes temporales y lista congelada de pares val/test (versionado)
├── src/bringvgc/
│   ├── parsing/              # parser del protocolo Showdown, normalización de formas
│   ├── data/                 # registros, C(o), particiones por serie
│   ├── features/             # encoders, usage stats (solo train)
│   ├── models/               # política autorregresiva, v̂
│   ├── baselines/            # random, usage, tipos, tabular, planificador
│   ├── sim/                  # BringPlayer (teampreview), arena, servidor local
│   ├── train/                # bc.py, vhat.py, rl.py, tabular.py
│   └── eval/                 # arena_eval.py, metrics.py, stats.py (Wilson, bootstrap)
├── tests/                    # parser vs muestra anotada, 90 acciones legales, antisimetría v̂,
│                             # |C(o)|∈{1,3,6}, "features no dependen de nº revelados"
├── scripts/                  # run_all.ps1 / .sh: una línea por tabla del paper
├── results/                  # CSV por corrida (semilla, config hash, commit); resúmenes versionados
├── notebooks/                # figuras finales (leen results/, no recalculan)
└── docs/
    ├── ETAPA2_PLAN.md        # este documento
    ├── MDP.md                # §4 en limpio
    ├── EXPERIMENTS.md        # protocolo + regla de decisión preregistrada + log de cambios
    └── AI_USAGE.md           # declaración de uso de IA (obligatoria)
```

- **Semillas:** `torch`, `numpy`, `random` y el muestreo de pares, fijadas desde la config. Showdown tiene su propio PRNG: verificar si se puede sembrar a través de poke-env. Si no se puede, declarar que la reproducibilidad es **a nivel de distribución** (por eso se usan 1000 batallas e IC).
- **Trazabilidad:** cada corrida guarda su config, el commit, la semilla, el tiempo de pared y el hardware en `results/<run_id>/meta.json`.
- **Servidor local:** Showdown con `node pokemon-showdown start --no-security`, fijado a un commit que soporte `gen9championsvgc2026regma`.
- **Commits de los dos.** La autoría se determina por los commits. Lo mejor es trabajar con ramas y PRs revisados por el otro, porque la revisión cruzada también prepara la defensa individual.
- **AI_USAGE.md:** en qué partes se usó IA, con qué propósito y qué se verificó a mano (p. ej. "el parser se verificó contra 50 replays anotados", "las ecuaciones se derivaron y comprobaron a mano").

---

## 12. Plan de trabajo

Las semanas son relativas, porque la fecha está en Discord. Es mejor reservar la última semana completa para escribir.

| Fase | Contenido | Entregable / criterio de salida | Responsable sugerido |
|---|---|---|---|
| **0. Spikes (2–3 días)** | (a) Showdown local con el formato Reg M-A; (b) `BringPlayer.teampreview` funciona y se respeta el orden; (c) `SimpleHeuristicsPlayer` dobles corre con megas; (d) **throughput $T$**; (e) regla de imputación de stats | Script `scripts/smoke_test` que juega 100 batallas Usage vs Usage ≈ 50 %; $T$ medido → presupuesto final (§9.4) | Ambos (S: a, e · J: b, c, d) |
| **1. Datos** | Parser re-verificado, registros, particiones por serie, lista congelada de pares, usage stats | Tests en verde; tabla de estadísticas del dataset; `splits/` versionado | Sergio |
| **2. Modelos offline** | Codificador, $\rho_{BC}$ (5 semillas), $\hat v$; diagnósticos en val | NLL parcial en val; accuracy/AUC de $\hat v$ en val | José ($\hat v$) · Sergio ($\rho_{BC}$) |
| **3. Arena y baselines** | Random, Usage, Tipos, Planificador-$\hat v$, Q tabular; sanity checks | Sanity checks aprobados; baselines en val | Sergio (arena/estadística) · José (tabular) |
| **4. RL** | REINFORCE + KL; 1 semilla piloto en val, ajuste mínimo; luego 5 semillas; barrido de entropía | Curvas con bandas; checkpoints elegidos por val | José (entrenamiento) · Sergio (pipeline de eval) |
| **5. Test y análisis** | Correr test **una sola vez** con la regla preregistrada; diagnósticos; cualitativo; casos de falla | Todas las tablas y figuras de §10 | Ambos |
| **6. Escritura** | Metodología definitiva + experimentos y resultados (§14); README; AI_USAGE | PDF actualizado; repo limpio | Ambos (cada uno redacta lo que implementó y revisa lo del otro) |

> Regla de oro: **test no se toca hasta la Fase 5.** Todo ajuste (hiperparámetros, checkpoint, umbrales) se hace con val. Si se mira test antes, se reporta.

---

## 13. Riesgos y planes B

| # | Riesgo | Señal temprana | Plan B |
|---|---|---|---|
| R1 | El Showdown local o el `SimpleHeuristicsPlayer` no soportan bien el formato Champions (megas, stat points) | Falla el spike 0a/0c | Actualizar el commit de Showdown o del fork. Si no se resuelve, un formato custom de dobles con Team Preview y nivel 50, con la limitación declarada. Último recurso: `MaxBasePowerPlayer` como $\pi_b$ |
| R2 | Throughput bajo ($T<3$) | Spike 0d | Recortes de §9.4. Si no basta: **RL con $\hat v$ como simulador aprendido** (model-based, estilo Dyna) para el grueso del entrenamiento y un fine-tuning corto con batallas reales. La evaluación **sigue siendo con batallas reales** (no hay circularidad). Declarar el cambio respecto a A2 |
| R3 | El RL no mejora sobre BC | Las curvas de val se quedan planas | El enunciado lo permite: "un agente que no aprende no es automáticamente un fracaso si el diagnóstico es riguroso". Se diagnostica con varianza del gradiente, entropía, calibración de $\hat v$, sensibilidad a $\beta_{KL}$ y ruido del entorno (1 batalla = 1 bit de información) |
| R4 | El agente explota a $\pi_b$ (reward hacking) | Gran mejora con $\pi_b$ y nada con otro $\pi_b$ | Subir $\beta_{KL}$. Reportarlo en la Etapa 3 como especificación defectuosa |
| R5 | Pocos pares de test con equipos totalmente nuevos | Al construir `splits/` | Relajar a "par no visto, con un equipo posiblemente visto" y reportar por separado los pares con 0 o 1 equipos vistos |
| R6 | El sesgo de revelación contamina $\rho_{BC}$ y $\hat v$ | Diferencias grandes entre el exact match estricto y el permisivo | Ya está mitigado (sin features de revelación, objetivos marginalizados). La RQ se mide por simulación y no depende de etiquetas |
| R7 | Corridas largas interrumpidas | — | Checkpoints y reanudación; repartir las semillas entre las dos PCs |

---

## 14. Qué escribir en el artículo (secciones nuevas de la Etapa 2)

**III. Metodología (versión definitiva):** reescribir III-B a III-I con los ajustes A1–A8.

- MDP final con la tabla de información excluida (§4.2) y las alternativas descartadas (§4.3).
- La política autorregresiva y su inicialización con BC.
- REINFORCE con baseline y KL: ecuación, justificación y por qué no hay replay ni red objetivo.
- Regla de uso de $\hat v$ (A2) y sustitución del GA (A3).
- Partición por series (A7) e imputación de stats (A8).

**IV. Experimentos:**

- Setup: hardware, versiones, $T$ medido, presupuesto y tiempos.
- Tabla de hiperparámetros.
- Tabla de baselines.
- Protocolo con la regla de decisión.

**V. Resultados:**

- **Tabla 1:** win rate contra Usage en test (media ± desviación estándar entre semillas, e IC) para todos los métodos y la línea del 55 %.
- **Tabla 2:** diagnósticos del predictor del rival frente a los pisos de azar.
- **Tabla 3:** $\hat v$ frente a Oblander (con salvedades).
- **Figura 1:** curvas de aprendizaje (recompensa y win rate en val) con bandas.
- **Figura 2:** entropía y barrido de $\beta_H$.
- **Figura 3:** calibración de $\hat v$ frente al win rate observado.
- **Figura 4:** win rate por par (agente vs BC vs Usage).
- **Tabla 4:** ejemplos cualitativos y casos de falla.

**Pendientes de forma que vienen de la Etapa 1:** citar las versiones publicadas y no arXiv; conseguir que el profesor apruebe Oblander (SSRN); ordenar la bibliografía por primera cita (IEEE) o pasar a BibTeX + IEEEtran.bst; declaración de uso de IA.

---

## 15. Preparación para la defensa (preguntas garantizadas)

1. **"¿Por qué formularon el MDP así y qué alternativas descartaron?"** → §4.3 (bandido de 1 paso, solo líderes, Team Preview dentro de la batalla, juego de Markov, $\hat v$ como recompensa). Saber explicar por qué $\gamma$ no afecta la política óptima.
2. **"¿Qué muestra su análisis multi-semilla sobre la estabilidad?"** → la dispersión de las curvas, la varianza entre semillas de RL frente a BC y el efecto de $\beta_H$. Saber distinguir la varianza *entre semillas* del ruido *de evaluación* (IC de Wilson).
3. **"¿Qué comportamiento indeseado podría premiar su recompensa?"** →
   - explotar las debilidades de $\pi_b$ en lugar de elegir buenos brings (R4);
   - especializarse contra el rival Usage (estocástico pero fijo), o sea, sobreajustar al oponente del entorno;
   - si hubiera tope de turnos con empate = 0, preferir brings que alargan las batallas cuando van perdiendo.
4. Otras previsibles:
   - ¿Por qué el RL y no solo BC? → la ablación BC, §8.
   - ¿Dónde está el leakage potencial y cómo lo evitaron? → §4.2, A7, usage solo de train, test intocado.
   - ¿Por qué 55 %? → el `MAX_WIN_RATE` de [5] con el rol invertido.
   - ¿Por qué 1000 batallas? → [4], ±0,03.

---

## 16. Checklist de entrega

- [ ] `docs/MDP.md` con estados, información excluida, acciones (90), recompensa exacta, terminación, $\gamma$ y alternativas
- [ ] Random + Usage + Tipos + Q tabular + BC + Planificador-$\hat v$ evaluados con el mismo arena
- [ ] Sanity checks (Usage vs Usage ≈ 50 %) documentados
- [ ] Hiperparámetros, presupuesto, **hardware y tiempo de pared** en una tabla
- [ ] 5 semillas; **todas** las curvas con banda y todas las tablas con ± desviación estándar
- [ ] Regla de decisión del 55 % escrita antes de correr test
- [ ] Análisis cualitativo, casos de falla y exploración–explotación
- [ ] Diagnósticos del predictor (lead pair, per-slot, full-bring estricto y permisivo)
- [ ] $\hat v$ frente a Oblander (accuracy y AUC) con salvedades
- [ ] README que reproduce cada tabla con un comando; `requirements.txt` fijado; datos fuera del repo
- [ ] Commits de **ambos**; PRs con revisión cruzada
- [ ] `AI_USAGE.md` y la sección de declaración de IA en el artículo
- [ ] Secciones de metodología definitiva y de experimentos y resultados en el PDF
- [ ] Sin `.venv`, `node_modules` ni datos pesados en la entrega
