> **BORRADOR asistido por IA (Worker Agent, Claude Sonnet 5.5, 12-11-2026).** Reporte del parche de $\hat v$: de regresión logística a el modelo bilineal del survey (eq:bilinear, eq:marginal, eqs. 8-11), en numpy. No se hizo ningún commit ni push.

# Parche de $\hat v$ bilineal (numpy)

- Rama: `restructure-numpy`. Esta corrida **retomó** la primera (cortada por el límite de uso el 11-11 a las 21:04).
- Compuerta: la nota de reestructuración está en `status: done` y existe `src/output/RESTRUCTURE_REPORT.md`. Se continuó.
- **Estado al retomar:** el árbol estaba limpio. El `vhat.py` bilineal de la primera corrida, `entrenar_vhat.py` y los tests nuevos ya estaban dentro del commit `69f044a` ("reestructura otra vez"), que hizo Sergio con todo junto. Es decir, el código de $\hat v$ **no** quedó en un commit de José, como pedía el plan §12 (ver el comando de commit al final).

## Qué se hizo en esta corrida y qué ya estaba

| Condición del panel | Estado | Dónde |
|---|---|---|
| Port del spec torch (`d3b1d7d`): d=64, hidden=64, K=16, lr/épocas de `vhat.yaml` | Ya estaba | `src/models/vhat.py`, `src/config/vhat.yaml` |
| $\phi_x$ sin features de uso; `<UNK>`=1 desde vocabulario solo de train; `<PAD>`=0 | Ya estaba (revisado) | `VHat.tokens`, `iniciar_parametros` |
| L2 en lugar de dropout | Ya estaba (`l2: 0.001`; sesgos y fila `<PAD>` sin L2) | `vhat.yaml`, `SIN_L2` |
| Verosimilitud marginalizada (11), $q$ uniforme sobre cada $\mathcal C(o)$, ≤ 36 términos, early stopping en val | Ya estaba | `perdida_marginal`, `entrenar_vhat.py` |
| Interfaz del Planificador sin cambios | Ya estaba, con test | `test_vhat_planner_interface_unchanged` |
| Checkpoint `vhat_bilineal_*.npz`; el cargador falla claro con el formato logístico viejo | Ya estaba, con test | `VHat.guardar/cargar`, `test_vhat_checkpoint_roundtrip_and_rejects_old_logistic_format` |
| Tests (reemplazan los logísticos) y chequeo de gradientes | Ya estaba; **verificado** aquí | `src/tests/test_models.py`, `test_gradients.py` |
| Comentarios en español que dicen qué es cada parámetro | Ya estaba (revisado) | encabezado y líneas de `nombres_parametros` de `vhat.py` |
| Docs (solo las secciones de $\hat v$) y nota de decisión | **Hecho aquí** | `docs/EXPLICACION_ETAPA2.md` §3, §4.8 y glosario; `docs/EXPERIMENTS.md`; `docs/AI_USAGE.md` |
| Smoke: 1 semilla, bilineal vs logístico, piloto torch etiquetado, mejor época | **Hecho aquí** | sección "Smoke" abajo |
| Este reporte | **Hecho aquí** | |

Los archivos `docs/MDP.md` y `README.md` no se tocaron: solo mencionan $\hat v$ como "no es la recompensa" y como paso `vhat` del runner, y nada de eso cambió.

## Tests

`python -m pytest` (`.venv`, Python 3.14.6): **47 passed, 20 skipped** en 28.7 s.

| | Antes (reporte de la reestructura, $\hat v$ logístico) | Ahora (bilineal) |
|---|---|---|
| passed | 42 | **47** |
| skipped | 20 (`test_parser_annotated.py`, anotación a mano pendiente) | 20 (los mismos) |

Los tests de $\hat v$ que reemplazaron a los logísticos:

| Test | Qué comprueba |
|---|---|
| `test_vhat_antisymmetry_exact` | $v(b_1,b_2)+v(b_2,b_1)=1$ (atol 1e-12) con $A$ grande, para que no sea trivial, y lo mismo para (11) con $q$ uniforme |
| `test_vhat_order_invariance_within_pairs` | $f(b)$ no cambia al reordenar los 2 líderes ni las 2 reservas (permutando la hoja de equipo) |
| `test_vhat_leads_and_reserves_are_distinguished_only_by_WL_neq_WR` | con $W_L\neq W_R$, intercambiar líderes y reservas cambia $f$; con $W_R=W_L$ no |
| `test_vhat_unseen_tokens_map_to_unk_and_padding_to_zero` | especies no vistas → `<UNK>`=1; movimientos faltantes → `<PAD>`=0 |
| `test_vhat_planner_interface_unchanged` | `v_matrix` da (90, 90) en (0,1); complementaria sobre la matriz 90×90; `PlannerBring.expected/probs` con las mismas formas |
| `test_vhat_checkpoint_roundtrip_and_rejects_old_logistic_format` | ida y vuelta exacta; el formato viejo da `ValueError("...logistico antiguo...")` |
| `test_vhat_marginal_likelihood_gradient_all_params_small_dims` | gradiente de (11) contra diferencias finitas, **todas** las coordenadas de los 14 grupos (d=hidden=8, K=4) |
| `test_vhat_marginal_likelihood_gradient_default_dims_sampled` | lo mismo con las dimensiones reales (64/64/16), 25 coordenadas al azar por grupo |

**Chequeo de gradientes** (`test_gradients.py`): float64, semillas fijas, diferencias centrales con eps = 1e-6, error relativo $\lVert g_a-g_n\rVert/(\lVert g_a\rVert+\lVert g_n\rVert) < 10^{-5}$ **por grupo** (`emb_especie, emb_objeto, emb_habilidad, emb_movimiento, W1, b1, W2, b2, W_L, W_R, beta_h, W_z, beta_z, A`), a través de la pérdida (11) con L2. El lote tiene $|\mathcal C|=3, 1$ y $6$ (hay $|\mathcal C|>1$ en ambos lados), un Pokémon con 2 movimientos (`<PAD>`) y especies `<UNK>`. $A$ se pone aleatoria; con $A=0$ el gradiente del resto sería 0 y el test no probaría nada.

Limitación: el test de dimensiones reales muestrea 25 coordenadas por grupo (no todas), porque cada coordenada cuesta dos pasadas hacia adelante.

## Smoke y comparación

**SMOKE TEST / PILOTO DE 1 SEMILLA: no son resultados.** Misma semilla (0), misma partición (4 831 batallas de train, 1 036 de val; test no se tocó), métrica principal **team** ($q$ uniforme sobre los 90 brings; la comparable con Oblander, sin fuga por revelación). Con 1 036 batallas de val, el error estándar de un acierto cercano a 0,55 es de unos 1,5 puntos: las diferencias de abajo están dentro del ruido.

| Modelo | Corrida | Épocas | Mejor época | val team acc | val team AUC | val obs acc | val obs AUC |
|---|---|---|---|---|---|---|---|
| Logístico (numpy, **reemplazado**) | `smoke_vhat_20261111-204543_s0` (smoke) | 2 (smoke) | 1 | 0,552 | 0,571 | 0,547 | 0,560 |
| **Bilineal (numpy)** | `smoke_vhat_20261111-210301_s0` (smoke) | 2 (smoke) | 1 | 0,539 | 0,565 | 0,517 | 0,542 |
| **Bilineal (numpy)** | `vhat_20261112-215914_s0` (config completa de `vhat.yaml`, 1 semilla, 50 s) | 7 (parada temprana, paciencia 4) | **2** | 0,548 | 0,571 | 0,543 | 0,561 |
| Piloto torch (`d3b1d7d`), **no comparable** | `legacy_torch/vhat_20261107-142758_s0` | 8 (parada temprana) | 2 | 0,551 | 0,570 | 0,537 | 0,552 |

- **Por qué el piloto torch no es comparable:** es otro entrenamiento (dropout 0,1 en vez de L2, lr 0,001, batch 256, weight decay, inicialización y partición de minilotes de torch), otro código y otra semilla efectiva. Se pone solo como referencia.
- La fila logística tiene solo el smoke de 2 épocas: su código ya no está en el repositorio (nunca se hizo commit), así que no se puede repetir con la configuración completa.
- La corrida completa del bilineal es **determinista**: sus épocas 0 y 1 coinciden con las del smoke, número por número.
- En la corrida completa la pérdida de val deja de mejorar tras la época 2 y la de train sigue bajando (de 8,26 a 0,90; la de train incluye el término L2, por eso arranca alto). Es sobreajuste: con ~4 800 batallas y 39 280 parámetros, la parada temprana es la que acota el modelo.
- **No se afirma que un modelo sea mejor que otro.** Las tres filas de $\hat v$ dan acierto de equipo entre 0,54 y 0,55 y AUC entre 0,565 y 0,571. Para compararlas haría falta más de una semilla y, aun así, no se llega al piso de solo habilidad de Oblander (65,54 %), que además es otro conjunto de datos (11,7 M partidas).
- La corrida completa **no se usó para nada más**. El paso `val` del runner tomará el checkpoint `vhat_*` más reciente de `src/output/` (está en un `.gitignore`).

## Cosas que José debe verificar a mano

1. Que $\phi_x$ (concatenar 4 tablas de embeddings y un MLP de 2 capas) es lo que quieren defender. El survey solo dice "embedding sobre especie, movimientos, objeto y habilidad": el MLP es decisión del proyecto, y está declarado así en `docs/EXPLICACION_ETAPA2.md` §4.8.
2. Que `W_R` no tiene sesgo propio ($\beta_h$ es único, como en la ecuación del survey). No se comparó línea por línea contra el torch de `d3b1d7d` en esta corrida (se tomó como spec lo descrito en el encabezado de `vhat.py`); sí se comparó contra el survey (punto 4).
3. Que L2 = 0,001 (en vez del dropout) y lr = 0,01 son razonables: no se sintonizaron. Con la pérdida de val estancada desde la época 2, quizá un L2 mayor mejore algo; eso sería una decisión de ustedes, con val.
4. **Cotejo con el survey** (`Research/Pokemon AI Survey.tex`, eq:bilinear en las líneas 972-1012, $h$ y $f$ en 990-993, eq:marginal en 1031-1048): se releyó en esta corrida y el código coincide en $h$, $f$, $\hat v$, $W_L\neq W_R$, la suma dentro de cada par, sin features de jugador y $q$ uniforme con ≤ 36 términos. Una sola diferencia de forma, que no cambia el modelo: el survey escribe $M=-M^\top$ y el código lo parametriza como $M=A-A^\top$ (toda matriz antisimétrica se puede escribir así, y el gradiente $dA = dM - dM^\top$ está en el test). Un detalle del survey que `vhat.py` no menciona: el survey dice que la restricción de antisimetría no cuesta precisión (Oblander 66,43 % contra 66,40 %); eso es una afirmación de la literatura, no algo que se probó aquí.

## Fuera de alcance (no se hizo)

Cambios a la política o al RL, corridas largas (solo 1 semilla de $\hat v$, 50 s), corrida de test, commits.

## Comandos de commit (para José)

El código de $\hat v$ ya está en `69f044a` (commit de Sergio). Los archivos que cambiaron en esta corrida son solo los docs y este reporte. José los revisa y hace commit después de que el commit de la reestructuración esté en la rama:

```powershell
git switch restructure-numpy
git add docs/EXPLICACION_ETAPA2.md docs/EXPERIMENTS.md docs/AI_USAGE.md src/output/VHAT_PATCH_REPORT.md
git commit -m "Docs y reporte del v-hat bilineal (eqs. 8-11 del survey)"
```
