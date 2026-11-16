> Los números se recopilaron de `src/output/`; el equipo formuló las interpretaciones y conclusiones a partir de esos resultados.

# Resultados de la Etapa 2

## 1. Qué se corrió

| Corrida | Cuándo | Salida |
|---|---|---|
| Preregistro (commit) | 15-11-2026, antes de test | `2e7afb3` |
| Barrido $\beta_H = 0$, semilla 0 | 15-11-2026 01:54 a 02:24 | `src/output/rl_20261115-015424_s0/` |
| Test final, una sola vez | 15-11-2026 02:24 a 03:28 | `src/output/tabla_test_vs_*.md`, `diagnosticos_test.json`, `arena_*_test_*` |

- **Test se corrió una vez** (`src/output/.test_run_done`: inicio 02:24:04, fin 03:28:55). No se cambió nada después de verlo.
- **Las 12 evaluaciones están completas:** 20 pares × 1000 batallas (500 por lado) = 20 000 batallas cada una, 0 empates, 0 timeouts.
- **El agente de test son las 3 corridas RL con la config de `rl.yaml`** ($\beta_H = 0{,}01$): `rl_20261113-141045_s0`, `rl_20261113-202239_s1`, `rl_20261114-130136_s2`, checkpoint elegido por win rate en val. La corrida del barrido no entró en test.
- Los errores de websocket al final de `salida_test.txt` (03:28:54) son el apagado de Showdown después de la última batalla. No afectan resultados.

## 2. Resultado principal (test, rival Usage estocástico)

| Método | Semillas | Win rate medio | Intervalo t 95 % (20 pares) | Desv. est. entre semillas | Regla del 55 % |
|---|---|---|---|---|---|
| **Agente RL** | 3 | **0,595** | **0,561 – 0,629** | 0,013 | **supera el 55 %** |
| BC | 3 | 0,561 | 0,520 – 0,603 | 0,012 | compatible, no concluyente |
| Tipos | 1 (determinista) | 0,561 | 0,517 – 0,606 | n/a | compatible, no concluyente |
| Usage-argmax | 1 (determinista) | 0,531 | 0,480 – 0,581 | n/a | se rechaza |
| Planificador-$\hat v$ | 1 | 0,506 | 0,463 – 0,550 | n/a | se rechaza |
| Random | 3 | 0,492 | 0,455 – 0,528 | 0,002 | se rechaza |

**Veredicto preregistrado:** el límite inferior del intervalo t sobre los 20 pares de test es 0,561 > 0,55. Según la regla del §5 de `docs/EXPERIMENTS.md`, el agente **supera el 55 %** contra Usage.

## 3. Comparaciones pareadas (agente − referencia, sobre los 20 pares)

Intervalo = media ± 1,96 · error estándar, como dice el preregistro. Al lado, con t de 19 grados de libertad (2,093 · EE), que es más estricto.

| Agente − … | Diferencia ± EE | Intervalo 1,96 | Intervalo t(19) | Lectura |
|---|---|---|---|---|
| Random | +0,103 ± 0,024 | +0,057 a +0,149 | +0,054 a +0,152 | mejor, claro |
| Usage-argmax | +0,065 ± 0,030 | +0,006 a +0,123 | +0,002 a +0,127 | mejor, por poco |
| BC | +0,034 ± 0,017 | +0,000 a +0,067 | −0,002 a +0,070 | **en el límite**: el RL aporta unos 3 puntos, pero el intervalo toca el 0 |
| Tipos | +0,034 ± 0,021 | −0,007 a +0,074 | −0,010 a +0,077 | **no distinguible** de la heurística de tipos |

Otras diferencias útiles (de las mismas tablas): BC − Tipos = +0,000 ± 0,031 (iguales); Planificador − BC = −0,055 ± 0,035.

## 4. Diagnósticos de $\rho_{BC}$ en test (predecir el bring humano)

2070 brings de jugadores en test, media ± desv. est. sobre 3 semillas:

| Métrica | Valor | Piso aleatorio |
|---|---|---|
| Par líder exacto | 14,4 % ± 0,06 % | 6,67 % |
| Bring completo permisivo ($\hat b \in \mathcal C(o)$) | 4,6 % ± 0,1 % | 1,11 % |
| NLL con $\lvert\mathcal C\rvert = 1$ | 4,289 | — |

BC predice el bring humano unas 2 a 4 veces mejor que el azar, pero lejos de acertar en la mayoría de los casos.

## 5. Exploración y explotación ($\beta_H \in \{0;\ 0{,}01\}$)

Win rate en val (10 pares, 200 batallas por par) durante el entrenamiento, `curves.csv`. La evaluación más cercana a 15 000 episodios es la de 14 080.

| Corrida | Val a 14 080 ep. | Mejor val (episodio) | Val final (50 048) | Entropía a 15k → final | KL a $\rho_{BC}$ a 15k → final |
|---|---|---|---|---|---|
| $\beta_H = 0{,}01$, s0 | 0,539 | 0,616 (42 112) | 0,605 | 4,14 → 4,04 | 0,018 → 0,092 |
| $\beta_H = 0{,}01$, s1 | 0,541 | 0,614 (50 048) | 0,614 | 4,16 → 4,04 | 0,017 → 0,100 |
| $\beta_H = 0{,}01$, s2 | 0,527 | 0,605 (24 064) | 0,578 | 4,21 → 4,02 | 0,018 → 0,114 |
| **$\beta_H = 0$, s0** | 0,525 | 0,623 (40 064) | 0,621 | 4,12 → 4,03 | 0,026 → 0,100 |

**Lectura:**
- Con la misma semilla, a 15k episodios $\beta_H = 0$ va 1,4 puntos debajo (0,525 contra 0,539); al final va 1,6 puntos arriba. El error estándar de una evaluación en val es de unos 0,04 (desv. est. entre pares ≈ 0,12, 10 pares), así que **ninguna diferencia es medible**.
- La entropía casi no cambia entre las dos configuraciones (4,12 contra 4,14 a 15k). Con $\beta_H = 0{,}01$ el término de entropía es muy chico frente al gradiente de REINFORCE y al KL ($\beta_{KL} = 0{,}1$).
- Ojo: el máximo posible es $\ln 90 \approx 4{,}50$. Una entropía de 4,0 significa que la política sigue muy repartida. El agente se evalúa en greedy (argmax), así que juega su mejor bring, pero entrenando explora mucho de todas formas.
- Limitación: el barrido es de **una sola semilla**.

## 6. Qué cambió entre val y test

| Método | Val (10 pares) | Test (20 pares) |
|---|---|---|
| Agente | 0,580 | 0,595 |
| BC | 0,484 | 0,561 |
| Tipos | 0,594 | 0,561 |
| Usage-argmax | 0,467 | 0,531 |
| Random | 0,553 | 0,492 |

- **El hallazgo de val "Usage-argmax queda debajo de Random" no se repitió en test** (0,531 contra 0,492). Con 10 pares era ruido o algo de esos pares en particular. La explicación de cambio de distribución que proponía el handoff queda **más débil**: hay que bajarle el tono en el artículo.
- En val el agente perdía contra Tipos (0,580 contra 0,594); en test le gana por 3,4 puntos, pero sin diferencia medible. **La conclusión estable es "el agente empata o supera levemente a Tipos", no "le gana".**
- Los niveles absolutos cambian entre particiones (BC sube 8 puntos). Por eso hay que comparar con **diferencias pareadas** en los mismos pares, no con los win rates sueltos.

## 7. Qué se puede afirmar y qué no (para las conclusiones)

**Se puede afirmar:**
1. Con la política de batalla fija $\pi_b$ en los dos lados, el agente RL elige brings que ganan 59,5 % contra Usage estocástico en 20 pares de equipos nunca vistos en train, y **cumple la regla preregistrada del 55 %** (límite inferior 0,561).
2. El agente es claramente mejor que Random (+10 puntos) y mejor que Usage-argmax (+6,5 puntos, intervalo justo por encima de 0).
3. El RL sobre BC aporta unos +3,4 puntos, **en el límite** de lo medible: con 1,96·EE el intervalo apenas excluye el 0; con t(19) lo incluye. Hay que decirlo así.
4. El resultado es estable entre semillas (desv. est. 0,013).

**No se puede afirmar:**
1. Que el agente es mejor que una heurística de tipos simple: la diferencia (+0,034 ± 0,021) es compatible con 0.
2. Que los brings son "buenos" en juego humano: todo se mide con la misma heurística de batalla $\pi_b$, y el agente pudo aprender a explotarla (riesgo R4). La Etapa 3 lo prueba con otra $\pi_b$ (MaxBasePower).
3. Nada sobre otros rivales: hay un solo rival (Usage estocástico).
4. Que la entropía ayuda o estorba: el barrido no muestra diferencia y es de 1 semilla.
5. Que $\hat v$ sirve para planear: el planificador queda debajo de BC (−5,5 puntos) y apenas sobre Random. Con accuracy 0,548 y AUC 0,571 en val, $\hat v$ es demasiado débil como función de fitness.

**Limitaciones a declarar:**
- Una sola $\pi_b$ y un solo rival.
- Política lineal (sin interacciones entre Pokémon más allá de la suma).
- 3 semillas y 20 pares de test.
- Tipos, Usage-argmax y el planificador con 1 semilla (deterministas).
- Checkpoint elegido en val (sesgo de selección solo en val; test quedó limpio).
- Barrido de 1 semilla.
- Los diagnósticos miden la imitación del humano (BC), no la calidad del agente.
- La comparación de $\hat v$ con Oblander no es directa (otros datos, formato y etiquetas).

## 8. Verificación del parser

Los 20 replays de train anotados a mano (`src/tests/fixtures/annotated_replays.json`, 10 por Sergio y 10 por José Ignacio) coinciden con el parser en líderes y reservas reveladas: `src/tests/test_parser_annotated.py`, 20 de 20. La revisión cruzada encontró un error de anotación (faltaba Garchomp en `2603150904`, lado p2) que se corrigió viendo el log. El parser estaba bien.
