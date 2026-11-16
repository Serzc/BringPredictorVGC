> **Smoke tests: 1 semilla, solo val, 07-11-2026. No citables como resultados.**

# Piloto de la Etapa 2

Hardware: Intel i5-10400F (6 núcleos / 12 hilos), 16 GB RAM, sin GPU, Windows 11. Python 3.14.6, torch 2.15.0+cpu, poke-env fork `379a046`, Showdown `913da36`, un solo proceso.

## Fase 1: datos (`results/dataset/dataset_stats.md`)

- Parseo: 6 902 / 6 905 válidas (99,96 %); excluidas 3 batallas con `teamsize` 5.
- $v$: 84,1 % / 12,0 % / 3,9 % (igual que la Etapa 1). Revelación completa: ganadores 73,6 %, perdedores 94,7 %.
- Particiones: train 4 831 batallas (882 series), val 1 036 (205), test 1 035 (158).
- Cobertura de vocabulario en val: especie 99,9 %, objeto 99,9 %, habilidad 100 %, movimiento 100 %. En test: 99,1 / 99,9 / 99,6 / 100 %.
- Pares: val 10 y test 20, todos con 0 equipos vistos (candidatos con 0/1/2 equipos vistos: val 386/335/70, test 462/305/49). R5 no hizo falta.
- Validador: 12 167 / 12 256 equipos legales; 80 batallas con algún equipo ilegal (no se usan como pares).

## Fase 2: modelos offline (val)

| Modelo | Métrica (val) | Valor | Referencia |
|---|---|---|---|
| $\rho_{BC}$ (semilla 0, mejor época 22) | NLL $|\mathcal C|=1$ / 3 / 6 | 4,109 / 2,999 / 2,395 | azar: ln 90 = 4,50 / ln 30 = 3,40 / ln 15 = 2,71 |
| | Lead-pair exact match | 20,8 % | piso 6,67 % |
| | Full-bring permisivo | 8,2 % | piso 1,11 % |
| $\hat v$ (semilla 0, mejor época 2) | Accuracy / AUC a nivel de equipo | 55,1 % / 0,570 | Oblander: 66,43 % / 0,730 con 11,7 M partidas (no comparable directamente) |
| | Accuracy / AUC con $\mathcal C(o)$ observados (diagnóstico) | 53,7 % / 0,552 | — |

$\hat v$ sobreajusta desde la época 3 (la pérdida de train sigue bajando y la de val sube): con ~4 800 batallas de train es esperable.

## Fase 3: sanity checks y baselines (val, 10 pares × 200 batallas, rival Usage estocástico)

| Método | Media sobre pares | Intervalo t 95 % | Dif. vs Usage-argmax |
|---|---|---|---|
| Usage vs Usage (sanity) | 0,472 | [0,382, 0,562] | — |
| Usage-argmax vs sí mismo, mismo equipo (sanity) | 0,490 | [0,468, 0,512] | — |
| Random | 0,545 | [0,471, 0,618] | +0,092 ± 0,036 |
| Usage-argmax | 0,453 | [0,366, 0,540] | — |
| Tipos | 0,591 | [0,500, 0,681] | +0,138 ± 0,067 |
| BC | 0,520 | [0,444, 0,596] | +0,067 ± 0,040 |
| Planificador-$\hat v$ | 0,500 | [0,400, 0,600] | +0,047 ± 0,076 |

Tabla generada: `results/tabla_val_piloto.md`. Sin empates, sin batallas en el tope, sin timeouts. $T$ ≈ 52–55 batallas/s.

**Hallazgo a investigar:** Random (0,545) y Tipos (0,591) quedan por encima de Usage, y Usage-argmax por debajo de 0,5. Para esta $\pi_b$, copiar a la población parece un rival débil. Antes de las corridas largas hay que entenderlo, porque afecta a cómo se lee el umbral del 55 %. Mediana de turnos en la arena: 3–5, frente a 7 en los logs humanos.

## Fase 4: RL (semilla 0, 10 112 episodios, $\beta_H=0.01$, $\beta_{KL}=0.1$)

| Episodio | Win rate val (greedy) | Desv. entre pares |
|---|---|---|
| 0 ($\rho_{BC}$) | 0,526 | 0,115 |
| 2 048 | 0,552 | 0,092 |
| 4 096 | 0,589 | 0,130 |
| 6 016 | 0,600 | 0,129 |
| 8 064 | 0,614 | 0,123 |
| 10 112 | 0,614 | 0,133 |

- Entropía: 3,97 → 3,65 (máximo ln 90 = 4,50). $\mathrm{KL}(\pi\|\rho_{BC})$: 0 → 0,11.
- El 0,614 es el mismo val que se usa para elegir el checkpoint, así que es optimista. Con 1 semilla no dice nada sobre estabilidad.

## $T$ re-medido con $\rho_{BC}$ en el bucle (plan §9.3)

| Medida | Valor |
|---|---|
| $T$ de entrenamiento (episodios/s, con actualizaciones) | **47,9** |
| $T$ de la arena (batallas/s) | 51,7 |
| Tiempo de pared del piloto | 440 s (211 s de entrenamiento + evaluaciones en val) |
| Timeouts / batallas colgadas | 0 |

Con $T \approx 48$/s, el presupuesto del plan (≈ 533 000 batallas) son unas **3,1 h** en esta PC con un proceso, frente a las 15 h estimadas a 10/s. El plan asumía $T$ de 17–24/s sin modelos: el $T$ más alto se debe a los bloques de 8 batallas concurrentes del mismo par. Queda holgura para 5 semillas o más batallas por par en val, si lo deciden.
