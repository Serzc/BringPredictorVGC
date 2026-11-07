# Etapa 2: plan de implementación v2, versión condensada (BringPredictorVGC)

**Proyecto:** *The state of ML in Pokémon: the benefits of a bring predictor*. Track C (Aprendizaje por refuerzo)
**Equipo:** Sergio Zúñiga Castillo · José Ignacio Madriz Ortiz
**Entrega:** Etapa 2, 45 % del proyecto, fin de la semana 10 (fecha exacta en Discord).
**Fuentes:** `_IAR_2026S2_PY01_TrackRL.pdf` (enunciado), `Research/Pokemon AI Survey.tex` (Etapa 1), `ETAPA2_PLAN.md` (v1), resultados de la Fase 0 en `results/phase0/`.

> Borrador hecho con asistencia de IA. Sustituye a la v1 en alcance; la v1 queda como referencia de las justificaciones largas. Cada decisión la tienen que poder defender los dos. Si no están de acuerdo con algo, cámbienlo y anoten el porqué.

---

## 0. Qué cambia respecto a la v1 y por qué

La v1 cubría el rubro, pero añadía muchas piezas propias (no pedidas por el enunciado ni prometidas por el survey) que juntas casi duplicaban el código, las corridas y lo que hay que defender. La v2 conserva:

- todo lo que exige el rubro de la Etapa 2;
- todo lo que el survey promete y no se puede quitar sin declararlo (RQ, $\rho$ con (7), $\hat v$ con (11), el protocolo de 1000 batallas con swap, los diagnósticos);
- lo mínimo para responder las tres preguntas garantizadas de la defensa.

| Pieza de la v1 | v2 | Motivo |
|---|---|---|
| Q tabular (baseline) | **Se quita** | Random + Usage + Tipos ya cumplen "aleatoria + heurística". Era un segundo bucle de entrenamiento para un punto ya cubierto |
| Ablación $\beta_{KL}$, BC-init vs init aleatoria, feature opcional $\hat q_{opp}$ | **Se quitan** | Experimentos extra que ningún rubro pide |
| Bootstrap jerárquico + Wilcoxon | **Se quitan** | Wilson por par (descriptivo) + intervalo t sobre los 20 pares + desviación estándar entre semillas cubre el rubro (§9.2) |
| Calibración de $\hat v$, estratificación por arquetipo, perturbaciones controladas | **Se quitan** (calibración pasa a la Etapa 3 si sobra tiempo) | Análisis opcional |
| Transformer de 2 capas | **DeepSets** (MLP por Pokémon + pooling + interacción) | ~9,7k ejemplos de train; más fácil de entrenar, depurar y defender |
| $V_\psi$ aprendido | **Media del retorno del batch** como baseline | Sigue siendo REINFORCE con baseline, sin una red más |
| 5 semillas (BC y RL) | **3 semillas** | Mínimo del enunciado; se declara junto a A5 |
| Barrido $\beta_H \in \{0, 0.01, 0.05\}$ | $\{0, 0.01\}$ | Basta para el análisis de exploración–explotación |
| Diagnósticos: lead pair, per-slot, full-bring estricto y permisivo | **Lead pair + full-bring permisivo** | Los dos más informativos. Se declara (el survey promete los cuatro) |
| 50 replays anotados a mano | **20** | Suficiente para el test del parser |
| **Se mantienen** | KL a $\rho_{BC}$, $\hat v$ + Planificador-$\hat v$, preregistro del 55 %, 1000 batallas por par, 20 pares de test | Ver §8 |

Cambios frente al survey que hay que declarar en el artículo (todos en §3 y §14):

- A3: GA → enumeración exacta. A4: sin search builders. A5: "al menos diez" → 3 semillas.
- A9 (nuevo), cada punto con su porqué:
  - Usage **sin condicionar por arquetipo** (el survey, Tabla IV y III-G, lo condiciona): no hay una definición operativa de arquetipo y las tasas por especie generalizan a equipos nuevos.
  - Diagnósticos de 4 a 2: se quitan per-slot recall y full-bring estricto.
  - Un solo rival (Usage estocástico) en lugar de "una población de oponentes" (survey, Li & Wellman): la RQ fija a Usage como rival; la generalización a otros rivales y otra $\pi_b$ queda para la Etapa 3.
  - Exclusión de registros: el survey dice que "no se descarta ninguna batalla" (eq. marginal) y en Riesgos excluye 2 registros inconsistentes (0,014 %). Se aclara: se excluyen solo esos 2, y se reportan.
  - 20 replays anotados (no 50); sin ablaciones, bootstrap jerárquico ni calibración de $\hat v$.

---

## 1. Qué exige el enunciado y dónde se cubre

| Rubro (pts) | Exigencia | Sección |
|---|---|---|
| MDP (10) | Estados, información oculta, acciones, recompensa exacta, terminación, $\gamma$ | §4 |
| Baselines (7) | Aleatoria + heurística o algoritmo simple, comparadas con el agente **y** con el baseline de literatura | §8 |
| Entrenamiento (8) | Algoritmo justificado, hiperparámetros, episodios, **hardware y tiempo** | §7, §9.3 |
| Multi-semilla (7) | ≥ 3 semillas, media y dispersión en todas las curvas y tablas | §9 |
| Comportamiento (7) | Recompensa, éxito, longitud con curvas; cualitativo; fallas; exploración–explotación | §10 |
| Profesionalidad (6) | Repo limpio, semillas, configs, README, commits de ambos, Markdown, declaración de IA | §11 |

Penalizaciones que aplican: leakage (−15; en este track un MDP mal especificado cuenta), sin baselines (−10), decisiones sin justificar (−5 c/u, hasta −15), no reproducible (hasta −10), citas no leídas (−10). Módulo adicional: no aplica (grupo de dos).

---

## 2. Qué se hereda de la Etapa 1 (no se toca)

- **RQ e hipótesis:** una política de RL que ve los dos equipos completos elige un bring (4 de 6, par líder designado) que gana **≥ 55 %** contra el baseline de *population usage*, en **1000 batallas por par** con lados intercambiados y la misma política de batalla fija en ambos lados.
- **Acciones:** $|B| = \binom{6}{2}\binom{4}{2} = 90$, factorizadas en líderes (15) y reservas (6).
- **Recompensa:** terminal $\{-1, +1\}$, sin shaping.
- **Datos:** log set Reg M-A de VGC-Bench, 6 905 batallas, 13 810 player-brings; revelación correlacionada con el resultado.
- **Supervisión parcial:** pérdida (7) sobre $\mathcal C(o)$ y verosimilitud marginalizada (11) para $\hat v$, con $q$ uniforme sobre cada conjunto consistente (así lo define el survey). No se descarta ningún registro.
- **$\hat v$:** modelo bilineal antisimétrico (8)–(10) adaptado de Oblander al nivel del bring.
- **Protocolo:** partición temporal, pares de test no vistos, sides swap, 1000 batallas por par.
- **Fuera de alcance:** team building, entrenar una política de batalla, LLMs.

---

## 3. Ajustes respecto a la Etapa 1

Se mantienen A1–A8 de la v1 (resumidos) y se añade A9.

| # | Ajuste | Justificación corta |
|---|---|---|
| A1 | BC pasa a ser la inicialización; se añade una fase de RL online (REINFORCE) con batallas reales | (7) es imitación: sin ella no hay RL ni trade-off de exploración que analizar |
| A2 | $\hat v$ nunca es la recompensa del agente; se usa en el Planificador y como plan B | El survey se contradecía (III-B vs III-F) |
| A3 | El GA sobre 90 candidatos se sustituye por enumeración exacta (argmax) | Con 90 acciones enumerar es exacto y más barato; un GA solo puede igualarlo |
| A4 | Sin search builders | Costo alto y poca relación con el problema |
| A5 | **3 semillas** en lugar de "al menos diez" | Mínimo del enunciado; presupuesto de dos personas. Se reportan distribuciones sobre semillas **y** sobre pares de equipos |
| A6 | Longitud de episodio = 2 siempre; se reporta duración en turnos como diagnóstico | La métrica pedida es trivial en este MDP |
| A7 | La partición se hace por **serie Bo3** | 2 476 batallas en 1 245 series; partir una serie es leakage |
| A8 | Stats y naturaleza imputados con una regla fija, igual para ambos lados | `showteam` no los trae. **Pendiente: elegir la regla** (la Fase 0 usó un reparto aleatorio temporal) |
| A9 | **Simplificaciones de la v2:** sin Q tabular, DeepSets en vez de Transformer, baseline de media del batch, Usage sin arquetipo, 2 diagnósticos en vez de 4, un solo rival, 20 replays, sin ablaciones/bootstrap/calibración (lista completa en §0) | Alcance de un proyecto de curso de dos personas. Ninguna cambia la pregunta; las que reducen evidencia (diagnósticos, un solo rival) se declaran como limitación |

**Hallazgo de la Fase 0 que también se declara:** el `SimpleHeuristicsPlayer` del fork nunca megaevoluciona. $\pi_b$ = SimpleHeuristics **+ megaevolucionar en cuanto se pueda** (`auto_mega`). Decidir si se adopta (recomendado: sí, porque sin megas Reg M-A no es representativo).

---

## 4. MDP (rubro de 10 pts)

Igual que la v1 §4 (se copia en limpio a `docs/MDP.md`), sin la feature opcional $\hat q_{opp}$:

- **Estado** $s_t = \langle c_{self}, c_{opp}, m, \ell_t \rangle$: las dos hojas OTS (especie normalizada, 4 movimientos, objeto, habilidad), features de metajuego $m$ por Pokémon calculadas **solo en train** y **leave-one-out** (cada registro no cuenta en su propia tasa), y el par líder comprometido ($\varnothing$ en $t=0$).
- **Acciones:** $a_0$ = par líder (15), $a_1$ = reservas (6), con máscara de legalidad. Política autorregresiva $\pi(b|s) = \pi(a_0|s)\,\pi(a_1|s,a_0)$.
- **Transiciones:** $t=0\to1$ determinista; $t=1\to$ terminal, batalla completa en Showdown con $\pi_b$ en ambos lados y el rival eligiendo con Usage estocástico.
- **Recompensa:** $0$ en $t=0$; $+1$/$-1$/$0$ (victoria, derrota, empate o tope de **50 turnos**) al final. Fallos del simulador se descartan y se reporta su tasa.
- **Terminación:** siempre tras $a_1$, $H=2$. **$\gamma = 1$:** horizonte finito y recompensa terminal; cualquier $\gamma \in (0,1]$ deja la política óptima igual y con $\gamma=1$, $V(s) = 2P(\text{win})-1$.
- **Sin reward shaping** (argumentos del survey).
- **Información negada:** bring rival del episodio, número de revelados, duración, resultado, rating, nombres, historial Bo3, stats reales, estadísticas con datos de val/test (tabla de la v1 §4.2).
- **Alternativas descartadas** (defensa): bandido de 90 brazos, solo líderes, Team Preview dentro del MDP de batalla, juego de Markov con equilibrio, recompensa $=\hat v$ (v1 §4.3).

---

## 5. Arquitectura

```
logs Reg M-A ─► parser ─► registros (c1, c2, o1, o2, w, t, serie)
                              │ partición temporal por serie
        ┌──────────────┬──────┴─────────┬──────────────────┐
        ▼              ▼                ▼                  ▼
   usage stats     ρ_BC (pérdida 7)   v̂ (pérdida 11)    pares train/val/test
   (train, LOO)        │                │
        ▼              ▼                ▼
   Baseline Usage   Agente RL       Planificador-v̂
   = rival             │ (REINFORCE + KL a ρ_BC)
        └──────► ARENA: Showdown local (913da36) + poke-env (fork vgc-bench)
                 BringPlayer.teampreview() = método; choose_move() = π_b fija
                              ▼
                 results/*.csv ► métricas, curvas, tablas
```

Cada método es una subclase de `BringPlayer` (ya existe en `scripts/phase0_arena_spikes.py`) que solo cambia `choose_bring()`. $\pi_b$ es idéntica por construcción: ese es el control experimental central.

---

## 6. Datos y particiones

- **Partición temporal por serie:** train ≈ 70 % (2026-04-11 → ~05-25), val ≈ 15 %, test ≈ 15 % (~06-05 → 06-20). Cortes exactos en `configs/data.yaml`, fijados antes de mirar resultados.
- **Pares para simular:** ~200 de train para el RL, **10 de val**, **20 de test**. Test: ninguno de los dos equipos presente en train. **Val: mismo filtro** (si no, el early stopping es optimista). Si faltan pares, relajar a "par no visto" y reportar cuántos tienen 0 o 1 equipos vistos. Lista congelada en `splits/`.
- **Parser (bloqueante, especificar antes del modelo):** función `consistent_brings(o, team)` que devuelve $\mathcal C(o)$:
  - líderes = los dos primeros switch-in del lado tras el inicio;
  - reservas reveladas $V$ = cualquier otra entrada al campo;
  - normalizar formas (Mega, Floette-Eternal, Rotom-*), apodos e Illusion a la especie base antes de comparar;
  - tests: $|\mathcal C| \in \{1,3,6\}$ según $v$, siempre contiene el bring verdadero en los 20 replays anotados.
- **Vocabulario:** token `<UNK>` para especie, objeto, habilidad o movimiento no visto en train; reportar la cobertura de vocabulario en val y test.
- **Stats (A8):** regla fija y documentada, igual para ambos lados y todos los métodos.

---

## 7. Modelos y entrenamiento

### 7.1 Codificador (DeepSets)
- Embedding por Pokémon $\phi_x$: especie (32), objeto (16), habilidad (16), movimientos (16, promediados), features $m_x$ → MLP a d=64.
- Contexto del rival: pooling (media) de los 6 embeddings rivales, concatenado a cada Pokémon propio → MLP. Así cada Pokémon propio "ve" al equipo rival sin atención.

### 7.2 Política autorregresiva
- Cabeza de líderes: score por par $\text{MLP}(h_i + h_j)$, softmax sobre 15. Cabeza de reservas: condicionada al par elegido, softmax sobre 6. Máscara de legalidad.
- Test unitario: exactamente 90 brings alcanzables y legales.

### 7.3 $\rho_{BC}$ (pérdida 7)
- Datos: todos los player-brings de train, los dos lados de cada batalla.
- Pérdida $-\log \sum_{b \in \mathcal C(o)} \pi(b|s)$.
- Usos: (i) inicialización del agente; (ii) baseline-ablación BC; (iii) predictor del bring rival para los diagnósticos (llamándolo con los argumentos invertidos).
- Adam 1e-3, batch 256, weight decay 1e-4, dropout 0.1, ≤ 30 épocas. **Early stopping:** pérdida parcial en val **estratificada por $|\mathcal C|$** (la pérdida baja trivialmente cuando $|\mathcal C|$ es grande); se elige con la de $|\mathcal C|=1$. 3 semillas (depurar con 1).

### 7.4 $\hat v$ (pérdida 11)
- Forma (8)–(10), $K=16$, $M = A - A^\top$, proyecciones distintas para líderes y reservas.
- Pérdida: verosimilitud marginalizada sobre $\mathcal C(o_1)\times\mathcal C(o_2)$ (≤ 36 términos), $q$ uniforme (definición del survey). El sesgo de $q$ uniforme (los no revelados no son equiprobables) se menciona como limitación de la comparación con Oblander.
- Tests: $\hat v(b_1,b_2) + \hat v(b_2,b_1) = 1$; invariancia al orden dentro de cada par.
- Métrica: accuracy y AUC en **val** durante la Fase 2; en test **una sola vez** en la Fase 5. Comparación con Oblander (66,43 %, 0,730) con sus salvedades. 1 semilla basta (no es el modelo de la RQ).

### 7.5 Agente: REINFORCE con baseline y KL a $\rho_{BC}$

**Por qué este algoritmo:** episodios de 2 pasos con recompensa terminal y $\gamma=1$ → el retorno de Monte Carlo es exacto e insesgado, sin bootstrapping (DQN/A2C no aportan). Acciones discretas y pocas → policy gradient sobre softmax. Estado combinatorio (pares casi nunca repetidos) → hace falta aproximación de funciones. Init con BC siguiendo [4]. **Sin replay buffer ni red objetivo** (on-policy, sin targets): decirlo explícitamente.

$$\mathcal L(\theta) = -\mathbb E\big[(R - \bar R_{batch})\log\pi_\theta(b|s)\big] - \beta_H\,\mathcal H[\pi_\theta] + \beta_{KL}\,\mathrm{KL}(\pi_\theta \,\|\, \rho_{BC})$$

- $\bar R_{batch}$: media del retorno del batch (baseline sin red adicional).
- KL a $\rho_{BC}$: **se espera** que estabilice y limite que el agente explote a $\pi_b$ (pregunta de defensa 3). Se registra la curva de $\mathrm{KL}(\pi_\theta \| \rho_{BC})$ por semilla como evidencia.

**Bucle:** muestrear par de train y lado → el agente muestrea $b$, el rival elige con Usage estocástico → batalla con $\pi_b$ → $R$ → batch → paso de gradiente → cada $N$ episodios, eval greedy en val (200 batallas por par) y checkpoint. Checkpoint final por win rate en val.

| Parámetro | Valor inicial |
|---|---|
| Optimizador | Adam 1e-4, clip de gradiente 1.0 |
| Batch | 128 episodios (batallas en paralelo, concurrencia 8) |
| $\beta_H$ | 0.01 (barrido $\{0, 0.01\}$) |
| $\beta_{KL}$ | 0.1 (fijo) |
| Presupuesto | 50 000 episodios por semilla (provisional, ver §9.3) |
| Semillas | 3 |
| Evaluación | greedy (argmax autorregresivo) |

---

## 8. Baselines (rubro de 7 pts)

| Baseline | Definición | Papel |
|---|---|---|
| **Random** (obligatorio) | Uniforme sobre los 90 brings | ¿Aprendió algo? Pisos: 1,11 % exact match, 6,67 % lead pair |
| **Usage** (heurística, rival de la RQ y **baseline de literatura**) | Score con frecuencias por especie de líder/reserva/fuera de train (cuentas fraccionales sobre $\mathcal C(o)$), **sin condicionar por arquetipo** (A9); estocástico como rival, argmax como baseline evaluado en test | ¿Bastaba copiar a la población? Es el baseline de literatura del survey (Tabla IV; humanos ponen 59,3 % de peso en el uso poblacional según Oblander) |
| **Tipos** | Ventaja de tipos: mejores 4, lideran los 2 mejores | Regla de dominio escrita a mano |
| **BC** ($\rho_{BC}$ greedy) | Imitación sin RL | Ablación clave: ¿aportó el RL? |
| **Planificador-$\hat v$** | $\arg\max_b \sum_{b_2} q_{Usage}(b_2)\,\hat v(b, b_2)$, enumeración exacta | Cumple el baseline de "búsqueda con fitness aprendida" del survey (A3). Una vez que existe $\hat v$, son pocas líneas |

**Baseline de literatura:** Usage (arriba). No hay un win rate publicado para esta cantidad, así que se reporta la Tabla III del survey como contexto (umbral 55 %, rango de VGC-Bench, pisos). La comparación numérica $\hat v$ vs Oblander (Tabla 3 del artículo) **solo se incluye si el profesor aprueba la fuente** (SSRN); si no, se quita.

**Deterministas y semillas:** Tipos es determinista. Planificador-$\hat v$ es determinista dado $\hat v$, y $\hat v$ se entrena con **1 semilla** porque no es el modelo de la RQ (se declara). Random se evalúa con **3 semillas** de muestreo.

**Sanity checks:** Usage vs Usage ≈ 50 % en pares fijos con lados intercambiados; Random vs Usage < 50 %; mismo bring en ambos lados ≈ 50 %. (Nota de la Fase 0: con bring aleatorio vs aleatorio y swap, el 50 % sale casi por construcción; el check real es Usage vs Usage por par.)

---

## 9. Evaluación (rubro de 7 pts)

### 9.1 Batallas de la RQ
Por par de test: 500 con el método en $c_A$ contra Usage en $c_B$ y 500 con los lados invertidos (1000 por par). Agente RL, BC y Random en 3 semillas; Usage-argmax, Tipos y Planificador una vez (deterministas).

### 9.2 Estadística y regla de decisión (preregistrada)
- **Descriptivo:** win rate por par con IC de Wilson 95 % (±0,03 con $n=1000$).
- **Inferencia:** el win rate de cada par se promedia sobre las 3 semillas; sobre los **20 pares** se calcula la media y un **intervalo t al 95 %** (la unidad es el par, porque la varianza entre pares domina a la de Wilson). Al lado se reporta la desviación estándar entre semillas.
- **Regla:** "supera el 55 %" solo si el límite inferior del intervalo t sobre pares es > 0,55. Si la media supera 0,55 pero el intervalo no, "compatible, no concluyente". Si no, se rechaza.
- **Comparaciones entre métodos** (agente vs BC, agente vs Usage-argmax): diferencia pareada por par, media ± error estándar sobre los 20 pares.
- Escrita en `docs/EXPERIMENTS.md` **antes** de correr test.

### 9.3 Presupuesto de cómputo
Fase 0 midió $T \approx 17$–$24$ batallas/s **sin modelos** (`results/phase0/`). Con $\rho_{BC}$ en el bucle será menor: **volver a medir $T$ en el piloto de RL** y actualizar esta tabla con el CSV.

| Bloque | Batallas |
|---|---|
| Piloto RL (1 semilla, 10k, re-medir $T$) | 10 000 |
| RL: 3 semillas × 50k | 150 000 |
| Eval en val durante el RL (3 × 10 checkpoints × 10 pares × 200) | 60 000 |
| Barrido $\beta_H = 0$ (3 semillas × 15k) + su eval en val (3 × 3 checkpoints × 10 × 200) | 63 000 |
| Baselines en val (5 × 10 pares × 200) | 10 000 |
| Test RQ: (3 RL + 3 BC) × 20 pares × 1000 | 120 000 |
| Test baselines: (3 Random + Usage-argmax + Tipos + Planificador) × 20 × 1000 | 120 000 |
| **Total** | **≈ 533 000** |

A 10 batallas/s ≈ 15 h; a 3/s ≈ 49 h. Si $T < 3$: recortar episodios por semilla, la eval en val a 5 pares, o activar el plan B (R2). Checkpoints y reanudación obligatorios. Registrar hardware y tiempo de pared por bloque.

### 9.4 Diagnósticos sobre logs de test
- **Lead-pair exact match** (piso 6,67 %), 100 % de las batallas.
- **Full-bring exact match permisivo** (acierto si $\hat b \in \mathcal C(o)$, piso 1,11 %).
- Media ± desviación estándar sobre las 3 semillas de $\rho_{BC}$.

---

## 10. Análisis del comportamiento (rubro de 7 pts)

- **Curvas** (banda ±1 desviación estándar entre semillas): recompensa media por episodio, win rate en val, entropía de la política. Longitud de episodio: constante (= 2), se dice así y se grafica la duración en turnos de batallas ganadas y perdidas.
- **Cualitativo:** por par de test, bring del agente vs Usage vs el humano más frecuente, con su win rate. Nombrar patrones visibles (p. ej. respuesta a Trick Room, Fake Out + Intimidate, clima).
- **Casos de falla:** pares donde el agente queda por debajo de Usage o BC; qué tienen en común.
- **Exploración–explotación** (rubro obligatorio): exploración por política estocástica + bonificación de entropía + ancla KL. Observado: curva de entropía y su colapso; **comparación justa de $\beta_H$:** las corridas con $\beta_H=0$ (3 × 15k) contra los checkpoints de 15k de las corridas con $\beta_H=0.01$ (mismo presupuesto, sin costo extra), en win rate en val, varianza entre semillas y velocidad; el resultado a 50k de $\beta_H=0.01$ se reporta aparte; número de brings distintos probados por par. Riesgo: explotar a $\pi_b$ (se mide en la Etapa 3 con otro $\pi_b$).

---

## 11. Reproducibilidad y repositorio (rubro de 6 pts)

Estructura de la v1 §11, con estos cambios: sin `tabular.py` ni `tabular.yaml`; `eval/stats.py` solo con Wilson, intervalo t sobre pares y diferencia pareada; los planes viven en `Research/` (no en `docs/`); el ejemplo de `AI_USAGE.md` dice "20 replays anotados". Además:
- `requirements.txt` con poke-env del fork **por commit hash**; Showdown fijado a `913da36`.
- Semillas de `torch`, `numpy`, `random` y muestreo de pares desde la config. Si Showdown no se puede sembrar, declarar reproducibilidad a nivel de distribución.
- `results/<run_id>/meta.json` con config, commit, semilla, tiempo de pared y hardware (el script de Fase 0 ya lo hace).
- `.gitignore`: `.venv/`, `node_modules/`, `data/`.
- Commits de los dos, ramas y PRs con revisión cruzada.
- `docs/AI_USAGE.md`: qué se hizo con IA y qué se verificó a mano (incluye este plan, `extract_teams.py`, `phase0_arena_spikes.py` y el hallazgo de `auto_mega`).

---

## 12. Plan de trabajo

| Fase | Contenido | Criterio de salida | Responsable sugerido |
|---|---|---|---|
| **0. Spikes (hecha)** | Showdown local, bring respetado, $\pi_b$ en dobles con megas, $T$ | **Hecho:** bring_ok 1,0; `auto_mega`; $T$ 17–24/s sin modelos | — |
| **1. Datos** | Parser + `consistent_brings`, 20 replays anotados, partición por serie, pares congelados (val y test filtrados), usage stats LOO, regla A8 | Tests en verde; tabla de estadísticas del dataset; `splits/` versionado | Sergio |
| **2. Modelos offline** | DeepSets, $\rho_{BC}$ (3 semillas), $\hat v$ (1 semilla) | NLL estratificada por $|\mathcal C|$ en val; accuracy/AUC de $\hat v$ en val | José ($\hat v$) · Sergio ($\rho_{BC}$) |
| **3. Arena y baselines** | Random, Usage, Tipos, BC, Planificador; sanity checks | Usage vs Usage ≈ 50 % por par; baselines en val | Sergio (arena) · José (Planificador) |
| **4. RL** | Piloto 1 semilla (re-medir $T$), luego 3 semillas + $\beta_H=0$ | Curvas con banda; checkpoints por val | José (entrenamiento) · Sergio (eval) |
| **5. Test y análisis** | Test una sola vez con la regla preregistrada; diagnósticos; cualitativo; fallas | Tablas y figuras de §10 | Ambos |
| **6. Escritura** | Metodología definitiva + experimentos y resultados; README; AI_USAGE | PDF actualizado; repo limpio | Ambos |

Regla de oro: **test no se toca hasta la Fase 5.**

---

## 13. Riesgos y planes B

| # | Riesgo | Plan B |
|---|---|---|
| R1 | Formato no soportado | **Descartado en la Fase 0** (Showdown `913da36` incluye Reg M-A) |
| R2 | $T$ bajo con modelos en el bucle | Recortes de §9.3; si no basta, RL con $\hat v$ como simulador aprendido + fine-tuning corto con batallas reales (evaluación siempre con batallas reales). Declarar el cambio respecto a A2 |
| R3 | El RL no mejora sobre BC | Es un resultado válido si el diagnóstico es riguroso: entropía, varianza entre semillas, ruido del entorno |
| R4 | El agente explota a $\pi_b$ | Subir $\beta_{KL}$; reportarlo en la Etapa 3 |
| R5 | Pocos pares de val/test con equipos nuevos | Relajar a "par no visto" y reportar por separado |
| R6 | Sesgo de revelación en $\rho_{BC}$ y $\hat v$ | Ya mitigado (sin features de revelación, objetivos marginalizados); la RQ se mide por simulación |
| R7 | Corridas interrumpidas | Checkpoints y reanudación; repartir semillas entre las dos PCs |

---

## 14. Qué escribir en el artículo

- **III. Metodología:** MDP final, política autorregresiva con init BC, REINFORCE con baseline y KL (y por qué no hay replay ni red objetivo), regla de uso de $\hat v$ (A2), A3, A4, A5, A7, A8, `auto_mega` y **cada punto de A9** (lista de §0: Usage sin arquetipo, 2 diagnósticos, un solo rival, los 2 registros excluidos, 20 replays, sin ablaciones/bootstrap/calibración).
- **IV. Experimentos:** hardware, versiones, $T$ medido, presupuesto y tiempos; tabla de hiperparámetros; tabla de baselines; regla de decisión.
- **V. Resultados:** Tabla 1, win rate en test (media e intervalo t sobre pares, desviación estándar entre semillas) con la línea del 55 %; Tabla 2, diagnósticos del predictor vs pisos; Tabla 3, $\hat v$ vs Oblander (solo si se aprueba la fuente); Figura 1, curvas de aprendizaje con bandas; Figura 2, entropía y $\beta_H$; Figura 3, win rate por par (agente vs BC vs Usage); Tabla 4, ejemplos cualitativos y fallas.
- **Pendientes de la Etapa 1:** citas publicadas en vez de arXiv; aprobación de Oblander (SSRN); BibTeX + IEEEtran.bst; declaración de uso de IA.

---

## 15. Defensa (preguntas garantizadas)

1. **¿Por qué ese MDP y qué alternativas descartaron?** §4 (cinco alternativas) y por qué $\gamma$ no cambia la política óptima.
2. **¿Qué muestra el análisis multi-semilla?** Dispersión entre las 3 semillas de RL vs BC, efecto de $\beta_H$; distinguir varianza entre semillas de ruido de evaluación (Wilson).
3. **¿Qué comportamiento indeseado podría premiar la recompensa?** Explotar a $\pi_b$; sobreajustar al rival Usage; con empate = 0 al tope de turnos, alargar batallas perdidas.
4. Previsibles: ¿por qué RL y no solo BC? (ablación BC); ¿dónde podía haber leakage? (§4, A7, LOO, val/test filtrados, test intocado); ¿por qué 55 %? (`MAX_WIN_RATE` de [5]); ¿por qué 1000 batallas? ([4], ±0,03); **¿por qué 3 semillas y no 10?** (A5: presupuesto, mínimo del enunciado, distribución también sobre pares).

---

## 16. Checklist de entrega

- [ ] `docs/MDP.md` (estados, información excluida, 90 acciones, recompensa, terminación, $\gamma$, alternativas)
- [ ] Random + Usage + Tipos + BC + Planificador-$\hat v$ evaluados con el mismo arena
- [ ] Sanity checks documentados (Usage vs Usage ≈ 50 % por par)
- [ ] Tabla de hiperparámetros, presupuesto, hardware y tiempo de pared
- [ ] 3 semillas (RL, BC, Random); todas las curvas con banda; ± desviación estándar en las filas con semillas (Tipos, Usage-argmax y Planificador marcados como deterministas; $\hat v$ como 1 semilla)
- [ ] Regla del 55 % escrita antes de correr test
- [ ] Cualitativo, fallas y exploración–explotación
- [ ] Diagnósticos: lead pair y full-bring permisivo
- [ ] $\hat v$ vs Oblander con salvedades
- [ ] README que reproduce cada tabla; `requirements.txt` fijado; datos fuera del repo
- [ ] Commits de ambos; PRs con revisión cruzada
- [ ] `AI_USAGE.md` y declaración en el artículo
- [ ] Metodología definitiva y experimentos y resultados en el PDF
- [ ] Sin `.venv`, `node_modules` ni datos pesados
