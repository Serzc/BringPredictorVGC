> Formulación del MDP implementado en el proyecto, basada en `Research/ETAPA2_PLAN_v2.md` §4 y actualizada para los modelos lineales en NumPy.

# MDP del bring (Track C)

Se fijan dos cosas: la política de bring del rival $\rho_{opp}$ (baseline **Usage estocástico**) y una política de batalla común $\pi_b$ (**SimpleHeuristicsPlayer del fork vgc-bench + `auto_mega`**). Con eso el juego entre dos jugadores se reduce a un MDP de un solo agente:

$$\mathcal M = \langle S, A, T, R, \gamma \rangle, \qquad H = 2$$

## Estado

$s_t = \langle c_{self}, c_{opp}, m, \ell_t \rangle$

| Componente | Qué es | Dónde está en el código |
|---|---|---|
| $c_{self}, c_{opp}$ | Las dos hojas OTS: por Pokémon, especie (con forma), objeto, habilidad y 4 movimientos | `src/dataPrep/parser.py`; features por Pokémon en `src/utils/features.py` (tasas de uso, ventaja de tipos, one-hot de especie) |
| $m$ | Por Pokémon: $P(\text{líder})$, $P(\text{reserva})$, $P(\text{fuera})$ de su especie y $\log(1+n)/\log(1+n_{max})$. **Solo train**, cuentas fraccionales sobre $\mathcal C(o)$, suavizado hacia 2/6 con 5 pseudo-cuentas, **leave-one-out por batalla** para los registros de train | `src/utils/usage.py` (en las features entran como log-tasas) |
| $\ell_t$ | Par líder ya elegido ($\varnothing$ en $t=0$) | implícito en el paso 2: solo se puntúan pares de reservas entre los 4 que quedan (`src/models/bc.py`) |

Tokens no vistos en train → `<UNK>` (cobertura en val/test en `src/output/dataset/dataset_stats.json`).

## Información que se le niega al agente

| Información | Por qué |
|---|---|
| Bring del rival en este episodio | Es simultáneo y oculto; dárselo sería target leakage |
| Número de revelados, duración, turnos | Proxies del resultado: los ganadores revelan completo el 73,6 % de las veces y los perdedores el 94,7 % (medido en `src/output/dataset/`) |
| Resultado, rating, nombres | Leakage o ajeno a la decisión |
| Historial Bo3 | Mantiene el MDP markoviano; limitación declarada |
| Stats y naturaleza reales | OTS no los trae: se imputan con A8, igual para ambos lados |
| Estadísticas con datos de val/test | Leakage temporal |

## Acciones

- $t=0$: $a_0$ = par líder, 15 opciones. $t=1$: $a_1$ = reservas entre los 4 restantes, 6 opciones.
- $|B| = 15 \times 6 = 90$ (`src/utils/brings.py`; test: exactamente 90 brings distintos, legales y alcanzables).
- Política autorregresiva $\pi(b|s) = \pi(a_0|s)\,\pi(a_1|s,a_0)$ con máscara de legalidad por Pokémon. Desde el 11-11-2026 es una **softmax lineal**: logit de un par = suma de `theta_L · phi_x` (líderes) o `theta_R · phi_x` (reservas) de sus dos Pokémon. Limitación: no modela sinergias entre Pokémon (ver `docs/EXPLICACION_ETAPA2.md` §3).
- En Showdown: `/team abcd` (dos líderes y dos reservas en posiciones de la hoja).

## Transiciones

- $t=0\to1$: determinista ($\ell \leftarrow a_0$).
- $t=1\to$ terminal: se juega la batalla completa en Showdown local (`913da36`) con $\pi_b$ en ambos lados y el rival con su bring de Usage estocástico. Azar: daño, precisión, efectos secundarios, speed ties, bring del rival.

## Recompensa

- $R(s_0,a_0)=0$.
- $R(s_1,a_1) = +1$ si gana, $-1$ si pierde, $0$ si empata o pasa de **50 turnos** (`max_turns` en `src/config/arena.yaml`). Se reporta cuántas veces pasa (`draws`, `capped`).
- Un bloque de batallas colgado más de 10 min se descarta y se cuenta (`timeouts`). No cuenta como derrota.
- **Sin reward shaping.**

## Terminación y descuento

Siempre tras $a_1$ ($H=2$). $\gamma = 1$: con horizonte finito y recompensa solo terminal, cualquier $\gamma\in(0,1]$ escala el retorno por una constante positiva y no cambia la política óptima; con $\gamma=1$, $V(s)=2P(\text{win})-1$.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Bandido contextual de 90 brazos | Mismo conjunto de políticas; la factorización 15×6 baja el branching y separa líderes de reservas (donde está el counter-selection) |
| Solo par líder (15) | La decisión real es el bring completo; perdería la comparabilidad con VGC-Bench (90) |
| Team Preview dentro del MDP de batalla (como VGC-Bench) | Obliga a entrenar $\pi_b$ (fuera de alcance) y no aísla el bring |
| Juego de Markov con equilibrio | 90×90 por par con payoffs ruidosos; se fija el rival y la explotabilidad va a la Etapa 3 |
| Recompensa $=\hat v$ | Optimizaría un proxy aprendido y evaluar con lo mismo sería circular (A2). Solo plan B (R2) |

## Decisiones abiertas que afectan al MDP

- **A8** (stats imputados): `role_spread` por defecto, configurable (`a8_rule`). Ver `docs/EXPLICACION_ETAPA2.md`.
- **`auto_mega`**: cambio a $\pi_b$ respecto al SimpleHeuristicsPlayer original; se declara.
