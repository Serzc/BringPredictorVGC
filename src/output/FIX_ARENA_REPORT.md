# Arreglo de la arena colgada, `--resume` y corridas por semilla (13-11-2026)

> **BORRADOR asistido por IA** (Claude Code, Worker Agent). Todo el código nuevo está marcado como BORRADOR.
> Sergio y José tienen que poder defender cada decisión; por eso cada cambio trae su razón y la alternativa descartada.

## 1. Qué pasó y causa

**Síntoma (13-11, rama `restructure-numpy`).** `python src/main.py all` se quedó parado en la semilla 1 de RL:
`[arena] bloque colgado > 600s` repetido y `timed out during opening handshake` del websocket.

**Línea de tiempo (de los archivos, no de la consola, que no quedó guardada).**

| Hora | Evento | Evidencia |
|---|---|---|
| 14:03:44 | Arranca Showdown (PID 18260) | `CreationDate` del proceso |
| 14:10 → 14:59 | RL semilla 0 completa, **0 timeouts** | `rl_20261113-141045_s0/meta.json` |
| 15:10:21 | Arranca RL semilla 1 | `rl_20261113-151003_s1/meta.json` |
| 15:45:16 | Última línea del log de Showdown: `[slow battle] 15017ms - >p1 team 1425` | `showdown_server_20261113_hang.log` (copia del original) |
| 16:09:24 | Último `ckpt_best_val.npz` de la semilla 1 | fecha del archivo |
| después | Bloques colgados de 600 s, uno tras otro | consola del dueño |

**Confirmado en el código: la bola de nieve.** En la versión anterior de `src/sim/arena.py`, ante
`asyncio.TimeoutError` se llamaba a `_crear_jugadores()` sin tocar a los jugadores viejos. En poke-env 0.15.0
(revisado en `.venv`, sin adivinar nombres):

- El listener de cada jugador (`PSClient.listen`) corre en el hilo compartido `POKE_LOOP`. Cancelar
  `battle_against` **no** lo detiene: los jugadores viejos seguían conectados y jugando sus batallas en ese mismo hilo.
- `start_timer_on_battle_start=False` por defecto, así que una batalla abandonada **nunca termina** en Showdown.
- Si el websocket nuevo falla al conectar (`timed out during opening handshake`), `listen()` registra la
  excepción y termina sin avisar. El jugador nunca entra, `_send_challenges` espera `logged_in` para siempre
  y el bloque siguiente vuelve a perder 600 s. Cada fallo dejaba hasta 8 batallas abiertas y 2 clientes vivos más.
  Eso carga a Showdown y al hilo de poke-env, y el siguiente fallo se vuelve más probable.
- El `>p1 team` de 15 s (un comando que normalmente tarda milisegundos) indica que el proceso de batallas
  de Showdown estuvo saturado.

**No confirmado: qué causó el *primer* bloque colgado.** No hay traceback de la consola, y Showdown solo registra
comandos lentos. Hay dos hipótesis compatibles con los datos:

- (a) **Memoria.** `pm.battles` y `po.battles` no se vaciaban nunca. A las 16:09 cada jugador tenía del orden de
  cientos de miles de batallas en memoria (semilla 0, evaluaciones y semilla 1, con la misma arena). Las pausas
  largas del recolector de basura en el hilo de poke-env pueden hacer que el ping del websocket venza (20 s)
  y que se corte la conexión con batallas a medio jugar.
- (b) **Una batalla trabada.** poke-env no reenvía la orden tras `[Unavailable choice]`, y una excepción en
  `choose_move` mata la tarea sin responder. Sin timer, esa batalla no termina nunca.

El arreglo cubre cualquier disparador: el bloque se descarta y se limpia, y si se repite se reinicia Showdown.
La poda de batallas (cambio 5) ataca la hipótesis (a), pero **no está demostrado** que esa haya sido la causa.

**Showdown huérfano todavía abierto.** El PID 18260 (`node pokemon-showdown start 8000`) sigue escuchando en el
puerto 8000 y su proceso padre ya no existe: Python murió y el `finally` no llegó a apagarlo. No lo toqué porque
no lo arranqué yo. Mientras siga abierto, `main.py` se niega a arrancar ("puerto 8000 ya está ocupado").

## 2. Cambios (cada uno: razón y alternativa descartada)

| # | Cambio (archivo) | Por qué | Alternativa descartada y por qué |
|---|---|---|---|
| 1 | **Limpieza al colgarse un bloque** (`sim/arena.py`, `_limpiar_jugadores`): `/forfeit` y `/leave` en cada batalla abierta de los jugadores viejos, `stop_listening()`, cancelar el listener y las tareas pendientes, y recién entonces crear jugadores nuevos. Cada mensaje tiene un tope de `limpieza_timeout_s` | Corta la bola de nieve: no quedan batallas ni clientes vivos acumulándose | Solo cerrar el websocket: Showdown mantiene abiertas las batallas de un usuario desconectado (no hay timer). Reiniciar Showdown en cada fallo: cuesta unos 8 s y es exagerado para un fallo aislado |
| 2 | **Chequeo de login** antes de cada bloque (`login_timeout_s: 30`) | Un handshake fallido se detecta en 30 s en vez de 600 s | Esperar el bloque completo (lo de antes): convertía cada handshake fallido en 10 minutos perdidos |
| 3 | **Vigilante**: tras `max_timeouts_seguidos: 2` fallos seguidos se llama a `reiniciar_showdown()` y la corrida sigue. Cada fallo y cada reinicio queda en `arena_eventos` del `meta.json` de la corrida | Si Showdown mismo está degradado, limpiar no alcanza | Abortar la corrida: se pierde todo lo hecho desde el último checkpoint. N=1: reinicia por cualquier fallo aislado |
| 4 | **`sim/showdown.py`**: log nuevo con fecha por cada arranque o reinicio (`showdown_server_<fecha>.log`). `reiniciar_showdown()` mata **solo** `PROCESO_ACTUAL` (el PID que este proceso arrancó), espera a que se cierre el puerto, arranca otro con las mismas reglas (puerto libre, 60 s, `taskkill /T` en `finally`) y actualiza `PROCESO_ACTUAL`. Si el puerto sigue ocupado, aborta | Un reinicio no borra la evidencia del cuelgue, y nunca se mata un proceso ajeno | Matar a quien tenga el puerto (por ejemplo con netstat): podría matar un Showdown u otra cosa del dueño |
| 5 | **Poda**: al empezar cada bloque se olvidan las batallas que terminaron **un bloque antes** | El diccionario de batallas ya no crece sin límite (smoke: 8 en memoria tras 20 bloques, contra 84 sin poda) | `reset_battles()` de poke-env: lanza error si queda alguna batalla abierta. Borrar enseguida: si llega tarde el `\|deinit\|` de una batalla borrada, poke-env deja una tarea esperando para siempre |
| 6 | **Reanudación** (`training/entrenar_rl.py`): `ckpt_resume.npz` escrito de forma atómica tras la evaluación inicial y en cada `eval_every`. Guarda parámetros, Adam (`m`, `v`, `t`), episodio, próxima evaluación, mejor val, tiempo entrenando, `rng.bit_generator.state` y los estados de `random` y `np.random`. `curves.csv` y `episodes.csv` se anexan en ese momento; al reanudar se recortan las filas posteriores al checkpoint. `ckpt_best_val.npz`, `ckpt_latest.npz` y `meta.json` también se escriben de forma atómica | Una caída cuesta como mucho `eval_every` episodios (2000, alrededor de 1 min de batallas) y no la semilla entera | Reanudar desde `ckpt_best_val.npz`: sin Adam, RNG ni contador de episodios ya no es el mismo experimento. Guardar en cada lote: más escritura a disco con poca ganancia |
| 7 | **Reanudar exige la misma config**: la config de `meta.json` (con `bc_ckpt`) y la semilla tienen que coincidir. Si no, se niega con la lista de claves distintas | Mezclar dos configs en una curva es un experimento mal especificado | Reanudar igual y anotar el cambio: la curva resultante no se puede defender |
| 8 | **`--seed N`** solo para `rl` y `val`. Con otros comandos da error de argparse | Permite repartir semillas entre los dos PCs | `--seed` en `bc`/`vhat`: el panel lo acotó; esos pasos tardan segundos |
| 9 | **Saltar trabajo hecho** en todos los pasos (`data`, `bc`, `vhat`, cada evaluación de `val`, cada semilla de `rl` y su evaluación). Se salta solo si hay una corrida completa **con la misma config**. `--force` (que ya existía para `test`) lo repite | `all` ya no repite BC, v̂, val ni la semilla 0. Un barrido `--beta-H 0` no se salta porque su config es otra | Un flag nuevo `--redo`: el panel pidió extender `--force` |
| 10 | **Qué es "completa"** (`config.corrida_completa`): `complete: true` o `train_wall_s` en `meta.json` (la regla del panel). **Además**, para corridas anteriores a la marca: bc/v̂ con `checkpoint` existente, dataset con `n_valid`, evaluación con `summary.battles > 0` y `per_pair.csv` | **Desviación declarada de la regla del panel**: sin estos marcadores, BC, v̂ y las evaluaciones de val ya hechas no tienen ni `complete` ni `train_wall_s`, y `all` las repetiría. Todos esos campos se escriben solo al final del paso | Reescribir los `meta.json` existentes: prohibido tocar resultados en `src/output/` |
| 11 | `ultimo_checkpoint` **ignora corridas incompletas** | La semilla 1 cortada (solo tiene `ckpt_best_val.npz`) ya no puede colarse en `test` | — |
| 12 | Los timeouts en `meta.json` ahora son **de esa corrida**. Antes se acumulaba `arena.n_timeouts` del proceso entero (la semilla 1 heredaba los de la 0) | Números correctos por corrida | — |

Configuración nueva en `src/config/arena.yaml`: `max_timeouts_seguidos: 2`, `login_timeout_s: 30`, `limpieza_timeout_s: 30`.
**Sugerencia, no aplicada (la decide el grupo):** bajar `timeout_s` de 600 a unos 120. Un bloque normal tarda
menos de 1 s, y con 600 el vigilante tarda hasta 20 min en reiniciar Showdown en el peor caso.

## 3. Tests

- **15 tests nuevos**, todos pasan:
  - `src/tests/test_arena_robusta.py` (7):
    - un bloque colgado rinde y abandona las batallas abiertas, cierra los dos websockets y crea jugadores nuevos;
    - el vigilante reinicia al 2.º fallo seguido y un bloque bueno pone el contador en 0;
    - un login fallido cuenta como fallo;
    - la poda espera un bloque;
    - `reiniciar_showdown` **nunca** mata un PID que no arrancó: sin proceso propio no mata nada; con el puerto en manos de otro solo mata el propio y aborta; en el caso normal mata el viejo y arranca otro.
  - `src/tests/test_resume_y_salto.py` (8):
    - **cortar y reanudar da parámetros idénticos bit a bit** (`ckpt_latest` y `ckpt_best_val`), las mismas curvas y los mismos episodios que una corrida sin cortes, contra una arena falsa determinista;
    - reanudar con otra config se niega;
    - las corridas incompletas se ignoran y las completas se reconocen;
    - `rl_reanudable` exige la misma config;
    - `online_rl` salta la semilla hecha, `--seed` corre solo esa y `--force` la repite;
    - 3 combinaciones de CLI no permitidas.
- **Suite completa: 61 pasan, 21 se saltan** (antes: 46 y 21). Los 21 saltados son los mismos de antes:
  `test_arena.py`, porque el 8000 está ocupado, y los 20 replays anotados pendientes.
- **Garantía más débil con Showdown real.** Los resultados de las batallas no se pueden sembrar. Una corrida
  reanudada sigue la misma distribución que una sin cortes, pero no la misma secuencia de batallas, y los
  episodios posteriores al último checkpoint se vuelven a jugar con resultados nuevos. Lo exacto (bit a bit)
  solo vale contra la arena falsa.

## 4. Smoke contra Showdown real

El puerto 8000 estuvo ocupado todo el trabajo por el Showdown huérfano (PID 18260). No lo maté. Para no esperar
en vano, corrí el smoke con **Showdown real en el puerto 8001**, con un script temporal fuera del repo. Usó solo
pares de **val** (y pares de train como estados de RL, nunca test) y escribió todo en una carpeta temporal, nada
en `src/output/`.

| Chequeo | Resultado |
|---|---|
| Bloque normal (4 batallas) | 4 filas |
| Cuelgue forzado (tope 0,05 s), con batallas reales a medio jugar | bloque descartado; se mandó `/forfeit` a las batallas abiertas (1 a 3 por fallo, en dos corridas del smoke); los 2 listeners viejos terminaron |
| Segundo fallo seguido | **reinicio real** de Showdown: PID nuevo, log nuevo con fecha, 7,7 a 8,2 s |
| Bloque después del reinicio | 4 filas (los jugadores nuevos entran al Showdown nuevo) |
| Poda tras 20 bloques | 8 batallas en memoria (sin poda serían 84: 1 + 20 bloques de 4 desde que se recrearon los jugadores) |
| RL corto cortado después del checkpoint del episodio 32 y reanudado con `reanudar=` | reanudó desde el episodio 32, terminó en 64, `complete: true`, `resumes` anotado |
| Procesos al terminar | ningún Showdown ni Python míos quedaron vivos |

**No verificado:**
- en el puerto 8000;
- a través de `python src/main.py` (que usa el 8000);
- `test_arena.py` (bring respetado), que necesita el 8000;
- que Showdown efectivamente cerró las batallas rendidas: el cliente cierra su websocket antes de recibir la
  respuesta, así que en el lado del cliente una batalla quedó "sin terminar". Se mandó el `/forfeit`; el cierre
  en el servidor no se observó directamente.

## 5. Semilla 1

`src/output/rl_20261113-151003_s1/` **no tiene estado de reanudación**: solo `ckpt_best_val.npz` y un `meta.json`
sin `train_wall_s`. No se finge una reanudación. **La semilla 1 empieza de cero.** La carpeta queda intacta;
`ultimo_checkpoint`, el salto de trabajo y `--resume` la ignoran.

## 6. Comandos para el dueño (en orden, desde la raíz del repo)

```powershell
# 0) Cerrar el Showdown huérfano del 8000 (confirmar antes que es el de las 14:03 y no otro)
Get-CimInstance Win32_Process -Filter "ProcessId=18260" | Select-Object ProcessId, CommandLine, CreationDate
taskkill /PID 18260 /T /F

# 1) Semilla 1 desde cero (si se corta: repetir el mismo comando con --resume)
.venv\Scripts\python.exe src/main.py rl --seed 1
.venv\Scripts\python.exe src/main.py rl --seed 1 --resume      # solo si la anterior se cortó

# 2) Semilla 2 (igual: --resume si se corta)
.venv\Scripts\python.exe src/main.py rl --seed 2

# (equivalente a 1+2: salta data, BC, v-hat, val y la semilla 0, y hace las semillas 1 y 2)
.venv\Scripts\python.exe src/main.py all --resume

# 3) Barrido de entropía beta_H = 0 (tres semillas nuevas: su config es otra, no se salta)
.venv\Scripts\python.exe src/main.py rl --beta-H 0

# 4) Tablas de val
.venv\Scripts\python.exe src/main.py tablas
```

**Ojo con el paso 3 (problema que ya existía; no lo arreglé porque está fuera del alcance).** Las dos variantes
de β_H se llaman `rl_*` y `agent` en las evaluaciones:
- `tablas` (`recolectar`) usa la evaluación **más reciente** por (método, semilla);
- `main.py test` usa el `rl` completo **más reciente** por semilla.

Después del barrido, la tabla de val y un futuro `test` usarían β_H = 0 sin avisar. Antes de `test`, el grupo
tiene que decidir en val qué β_H se queda y asegurar que esa sea la que se usa (o pedir un arreglo de esa
selección).

Nota aparte: `CLAUDE.md` dice 5 semillas (A5). El plan v2 fija 3 a propósito, así que `CLAUDE.md` está desactualizado.

## 7. Archivos tocados

- **Sergio:** `src/sim/arena.py`, `src/sim/showdown.py`, `src/main.py`, `src/config/config.py`,
  `src/config/arena.yaml`, `src/training/eval/evaluar_arena.py`, `src/tests/test_arena_robusta.py` (nuevo),
  `src/tests/test_resume_y_salto.py` (nuevo), `docs/AI_USAGE.md` (una fila nueva) y este reporte.
- **De José (revisar y hacer commit él):** `src/training/entrenar_rl.py` (reanudación). **No se tocaron**
  `src/models/rl_agent.py` ni `src/models/bc.py`: el estado de Adam se lee de sus atributos públicos `opt.m`,
  `opt.v` y `opt.t`.
- Copia de evidencia: `src/output/showdown_server_20261113_hang.log` (ignorada por git, como los demás logs).
  El original `showdown_server.log` sigue ahí; `showdown.py` ya no lo pisa.

## 8. Commits sugeridos (los hace el grupo; dependen entre sí: `main.py` importa `config_de_corrida` de `entrenar_rl.py`, así que hay que hacer los dos antes de correr o hacer push)

```powershell
# José, después de revisar la reanudación:
git add src/training/entrenar_rl.py
git commit -m "RL: checkpoint de reanudacion (ckpt_resume.npz), CSV por partes y escritura atomica"

# Sergio:
git add src/sim/arena.py src/sim/showdown.py src/main.py src/config/config.py src/config/arena.yaml src/training/eval/evaluar_arena.py src/tests/test_arena_robusta.py src/tests/test_resume_y_salto.py src/output/FIX_ARENA_REPORT.md docs/AI_USAGE.md
git commit -m "Arena: limpieza de bloques colgados, vigilante que reinicia Showdown, --seed, --resume y salto de trabajo hecho"
```

`docs/AI_USAGE.md` ya tenía cambios sin commitear de antes de este trabajo, y quedan incluidos en ese `git add`.
Revisar con `git diff docs/AI_USAGE.md` antes. `docs/EXPERIMENTS.md`, `docs/EXPLICACION_ETAPA2.md` y
`src/output/VHAT_PATCH_REPORT.md` también tenían cambios previos; no son de este trabajo y no están en los comandos.
