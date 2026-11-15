"""Informe temporal de resultados Etapa 2: figuras (matplotlib) + DOCX (python-docx)."""
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

OUT = Path(r"C:\Users\XPC\Documents\TEC\IA\BringPredictorVGC\Research\Etapa2_resultados")
SRC = Path(r"C:\Users\XPC\Documents\TEC\IA\BringPredictorVGC\src\output")
OUT.mkdir(parents=True, exist_ok=True)

AZUL, AZUL_OSC, NARANJA = "#2a78d6", "#184f95", "#eb6834"
TINTA, TINTA2, GRILLA, FONDO = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRILLA,
                     "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.facecolor": FONDO, "figure.facecolor": FONDO, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6})

NOMBRES = {"agent": "Agente RL", "bc": "BC", "tipos": "Tipos", "usage_argmax": "Usage-argmax",
           "planner": "Planificador-v-hat", "random": "Random"}


def tabla(ref):
    filas = [l for l in (SRC / f"tabla_test_vs_{ref}.md").read_text(encoding="utf-8").splitlines() if l.startswith("|")]
    cab = [c.strip() for c in filas[0].strip("|").split("|")]
    return [dict(zip(cab, [c.strip() for c in f.strip("|").split("|")])) for f in filas[2:]]


def por_par(metodo):
    """win rate por par promediado sobre semillas, en test."""
    acum = {}
    dirs = sorted(SRC.glob(f"arena_{metodo}_vs_usage_test_*"))
    for d in dirs:
        for r in csv.DictReader(open(d / "per_pair.csv", encoding="utf-8")):
            acum.setdefault(int(r["pair_id"]), []).append(float(r["win_rate"]))
    return {k: sum(v) / len(v) for k, v in acum.items()}


base = tabla("usage_argmax")
orden = ["agent", "bc", "tipos", "usage_argmax", "planner", "random"]
fila = {f["metodo"]: f for f in base}

# --- Figura 1: win rate en test con intervalo t ---------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 2.9))
for i, m in enumerate(orden[::-1]):
    f = fila[m]
    lo, hi, me = float(f["t95_bajo"]), float(f["t95_alto"]), float(f["media"])
    c = AZUL_OSC if m == "agent" else AZUL
    ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot(me, i, "o", ms=8, color=c, mec=FONDO, mew=2)
    ax.text(me, i + 0.22, f"{me:.3f}", ha="center", va="bottom", color=TINTA, fontsize=8.5)
ax.axvline(0.55, color=TINTA2, lw=1, ls=(0, (4, 3)))
ax.text(0.551, len(orden) - 0.42, "umbral 0,55", color=TINTA2, fontsize=8)
ax.axvline(0.5, color=GRILLA, lw=1)
ax.set_ylim(-0.5, len(orden) - 0.3)
ax.set_yticks(range(len(orden)), [NOMBRES[m] for m in orden[::-1]])
ax.set_xlabel("Win rate contra Usage estocástico (media e intervalo t 95 % sobre 20 pares)")
ax.set_xlim(0.44, 0.66)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig1_test_winrate.png", dpi=200); plt.close(fig)

# --- Figura 2: diferencias pareadas agente - referencia --------------------------------
refs = [("random", "Random"), ("usage_argmax", "Usage-argmax"), ("bc", "BC"), ("tipos", "Tipos")]
fig, ax = plt.subplots(figsize=(6.2, 2.3))
for i, (r, nom) in enumerate(refs[::-1]):
    ag = next(f for f in tabla(r) if f["metodo"] == "agent")
    d, se = [float(x) for x in ag[f"dif_vs_{r}"].replace("+", "").split("±")]
    ax.plot([d - 1.96 * se, d + 1.96 * se], [i, i], color=AZUL, lw=2, solid_capstyle="round")
    ax.plot(d, i, "o", ms=8, color=AZUL, mec=FONDO, mew=2)
    ax.text(d + 1.96 * se + 0.004, i, f"{d:+.3f}", va="center", color=TINTA, fontsize=8.5)
ax.axvline(0, color=TINTA2, lw=1)
ax.set_yticks(range(len(refs)), [f"Agente − {n}" for _, n in refs[::-1]])
ax.set_xlabel("Diferencia pareada de win rate (media ± 1,96 EE, 20 pares)")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig2_diferencias.png", dpi=200); plt.close(fig)

# --- Figura 3: agente vs Tipos por par ------------------------------------------------
ag, ti = por_par("agent"), por_par("tipos")
ids = sorted(set(ag) & set(ti), key=lambda k: ag[k] - ti[k])
fig, ax = plt.subplots(figsize=(6.2, 2.6))
difs = [ag[k] - ti[k] for k in ids]
ax.bar(range(len(ids)), difs, color=[AZUL if v >= 0 else NARANJA for v in difs], width=0.75)
ax.axhline(0, color=TINTA2, lw=1)
ax.set_xticks(range(len(ids)), [str(k) for k in ids], fontsize=7)
ax.set_xlabel("Par de test (ordenado por diferencia)")
ax.set_ylabel("Agente − Tipos")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig3_agente_vs_tipos_por_par.png", dpi=200); plt.close(fig)
n_mejor = sum(v > 0 for v in difs)

# --- Figura 4: curvas de aprendizaje (val), y entropia y KL en paneles aparte ----------
corridas = [("rl_20261113-141045_s0", "β_H = 0,01 · s0"), ("rl_20261113-202239_s1", "β_H = 0,01 · s1"),
            ("rl_20261114-130136_s2", "β_H = 0,01 · s2"), ("rl_20261115-015424_s0", "β_H = 0 · s0")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
for d, nom in corridas:
    filas = list(csv.DictReader(open(SRC / d / "curves.csv", encoding="utf-8")))
    c, lw = (NARANJA, 2) if "β_H = 0 " in nom else (AZUL, 1.2)
    v = [(int(r["episode"]), float(r["val_win_rate"])) for r in filas if r["val_win_rate"]]
    t = [(int(r["episode"]), float(r["entropy"]), float(r["kl_to_bc"])) for r in filas if r["entropy"]]
    # media movil de 20 actualizaciones: las curvas por batch son muy ruidosas
    w = 20
    t = [(t[i][0], sum(x[1] for x in t[i-w+1:i+1]) / w, sum(x[2] for x in t[i-w+1:i+1]) / w) for i in range(w - 1, len(t))]
    axs[0].plot([a for a, _ in v], [b for _, b in v], color=c, lw=lw, alpha=0.9 if lw > 1.5 else 0.7)
    axs[1].plot([a for a, _, _ in t], [b for _, b, _ in t], color=c, lw=lw, alpha=0.8)
    axs[2].plot([a for a, _, _ in t], [k for _, _, k in t], color=c, lw=lw, alpha=0.8)
for a, tit in zip(axs, ["Win rate en val", "Entropía (media móvil)", "KL(π ‖ ρ_BC) (media móvil)"]):
    a.set_title(tit, fontsize=9, color=TINTA, loc="left")
    a.axvline(15000, color=GRILLA, lw=1, ls=(0, (4, 3)))
    a.set_xlabel("Episodios")
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
axs[0].plot([], [], color=AZUL, lw=1.2, label="β_H = 0,01 (3 semillas)")
axs[0].plot([], [], color=NARANJA, lw=2, label="β_H = 0 (semilla 0)")
fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig4_curvas.png", dpi=200); plt.close(fig)

# --- Figura 5: val contra test por metodo ----------------------------------------------
val = {"agent": 0.580, "bc": 0.484, "tipos": 0.594, "usage_argmax": 0.467, "random": 0.553, "planner": 0.5415}
fig, ax = plt.subplots(figsize=(6.2, 2.7))
for i, m in enumerate(orden[::-1]):
    a, b = val[m], float(fila[m]["media"])
    ax.plot([a, b], [i, i], color=GRILLA, lw=2)
    ax.plot(a, i, "o", ms=8, color=NARANJA, mec=FONDO, mew=2)
    ax.plot(b, i, "o", ms=8, color=AZUL, mec=FONDO, mew=2)
ax.plot([], [], "o", color=NARANJA, label="Val (10 pares)"); ax.plot([], [], "o", color=AZUL, label="Test (20 pares)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_yticks(range(len(orden)), [NOMBRES[m] for m in orden[::-1]])
ax.set_xlabel("Win rate contra Usage estocástico")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "fig5_val_vs_test.png", dpi=200); plt.close(fig)

# --- Documento --------------------------------------------------------------------------
diag = json.loads((SRC / "diagnosticos_test.json").read_text(encoding="utf-8"))
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2); s.top_margin = s.bottom_margin = Cm(2)


def p(texto, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(texto); run.italic = italic
    return par


def fig_(nombre, pie):
    doc.add_picture(str(OUT / nombre), width=Cm(16))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(pie); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].italic = True; c.runs[0].font.size = Pt(9)


def fuente(texto):
    """Linea gris con de donde sale el dato (rutas relativas a la raiz del repo BringPredictorVGC)."""
    par = doc.add_paragraph()
    r = par.add_run("Fuente: " + texto); r.font.size = Pt(8.5); r.italic = True
    r.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


def tabla_doc(cab, filas):
    t = doc.add_table(rows=1, cols=len(cab)); t.style = "Light Grid Accent 1"
    for c, h in zip(t.rows[0].cells, cab):
        c.text = h
    for f in filas:
        for c, v in zip(t.add_row().cells, f):
            c.text = str(v)
    doc.add_paragraph()


doc.add_heading("BringPredictorVGC: resultados y conclusiones de la Etapa 2", 0)
p("Documento temporal generado con IA el 15-11-2026 a partir de src/output/ (commit 513505e). "
  "Borrador para revisar: las interpretaciones se tienen que defender con palabras propias.", italic=True)

doc.add_heading("1. Pregunta y protocolo", 1)
p("¿Una política de RL que ve los dos equipos completos elige un bring que gana al menos 55 % contra el baseline "
  "Usage estocástico? Cada método juega 1000 batallas por par (500 por lado) en 20 pares de equipos de test que no "
  "aparecen en train, con la misma política de batalla fija en los dos lados. Regla preregistrada (commit 2e7afb3, "
  "antes de test): el agente supera el 55 % solo si el límite inferior del intervalo t sobre los 20 pares es mayor "
  "que 0,55. Test se corrió una sola vez; las 12 evaluaciones terminaron completas (20 000 batallas cada una, sin "
  "empates ni timeouts).")
fuente("pregunta y regla: docs/EXPERIMENTS.md §1 y §5 (commit 2e7afb3). Una sola corrida de test: "
       "src/output/.test_run_done (inicio 2026-11-15 02:24:04, fin 03:28:55). Batallas, empates y timeouts: campo "
       "summary de src/output/arena_*_test_*/meta.json (n_pairs 20, battles 20000, draws 0, timeouts 0 en las 12).")

doc.add_heading("2. Resultado principal", 1)
tabla_doc(["Método", "Semillas", "Win rate", "Intervalo t 95 %", "Regla del 55 %"],
          [[NOMBRES[m], fila[m]["semillas"], f"{float(fila[m]['media']):.3f}",
            f"{float(fila[m]['t95_bajo']):.3f} a {float(fila[m]['t95_alto']):.3f}", fila[m]["regla_55"]] for m in orden])
fuente("src/output/tabla_test_vs_usage_argmax.md, escrita por python src/main.py test con "
       "src/training/eval/tablas.py: tabla() promedia el win rate de cada par sobre las semillas (archivos "
       "per_pair.csv de cada carpeta arena_<método>_vs_usage_test_*) y decision() calcula el intervalo t sobre los "
       "20 pares (t_interval en src/utils/estadistica.py) y aplica la regla del 55 %.")
fig_("fig1_test_winrate.png", "Figura 1. Win rate en test contra Usage estocástico. Punto: media sobre 20 pares; "
     "línea: intervalo t al 95 %. La línea punteada es el umbral preregistrado.")
p("El agente gana 59,5 % y el límite inferior de su intervalo es 0,561. Por la regla preregistrada, el agente "
  "supera el 55 %. La desviación estándar entre semillas es 0,013, así que el resultado es estable.")
fuente("columna sd_semillas de la misma tabla: desviación estándar (ddof=1) de la media sobre pares de cada semilla. "
       "Semillas del agente: rl_20261113-141045_s0, rl_20261113-202239_s1 y rl_20261114-130136_s2 "
       "(campo config.checkpoint de cada arena_agent_vs_usage_test_*/meta.json).")

doc.add_heading("3. Comparaciones pareadas", 1)
filas_d = []
for r, nom in refs:
    a = next(f for f in tabla(r) if f["metodo"] == "agent")
    d, se = [float(x) for x in a[f"dif_vs_{r}"].replace("+", "").split("±")]
    filas_d.append([f"Agente − {nom}", f"{d:+.3f} ± {se:.3f}", f"{d-1.96*se:+.3f} a {d+1.96*se:+.3f}",
                    f"{d-2.093*se:+.3f} a {d+2.093*se:+.3f}"])
tabla_doc(["Comparación", "Diferencia ± EE", "Intervalo 1,96 EE", "Intervalo t(19)"], filas_d)
fuente("columna dif_vs_<referencia> de src/output/tabla_test_vs_random.md, tabla_test_vs_usage_argmax.md, "
       "tabla_test_vs_bc.md y tabla_test_vs_tipos.md: media de (agente − referencia) por par y su error estándar "
       "(paired_difference en src/utils/estadistica.py). Los intervalos salen de esos dos números: media ± 1,96·EE "
       "(lo que dice el preregistro, docs/EXPERIMENTS.md §4) y media ± 2,093·EE (t con 19 grados de libertad).")
fig_("fig2_diferencias.png", "Figura 2. Diferencia pareada agente − referencia sobre los mismos 20 pares. "
     "Si la línea cruza el 0, la diferencia no es medible.")
p("El agente es claramente mejor que Random y mejor que Usage-argmax. Sobre BC (su inicialización) gana unos 3,4 "
  "puntos, en el límite: el intervalo con 1,96 EE apenas excluye el 0 y con t(19) lo incluye. Contra Tipos la "
  "diferencia es compatible con 0.")
fig_("fig3_agente_vs_tipos_por_par.png", f"Figura 3. Agente − Tipos en cada par de test. El agente gana en "
     f"{n_mejor} de {len(difs)} pares y pierde en {len(difs) - n_mejor}.")
fuente("columna win_rate de per_pair.csv en arena_agent_vs_usage_test_* (promedio de las 3 semillas por par) menos "
       "la de arena_tipos_vs_usage_test_20261115-031809_s0. El número de par es pair_id.")

doc.add_heading("4. Predicción del bring humano (BC)", 1)
tabla_doc(["Métrica en test (2070 brings)", "BC (3 semillas)", "Azar"],
          [["Par líder exacto", f"{100*diag['lead_pair_acc']['media']:.1f} %", "6,67 %"],
           ["Bring completo permisivo", f"{100*diag['full_bring_permissive_acc']['media']:.1f} %", "1,11 %"]])
p("BC predice el bring humano entre 2 y 4 veces mejor que el azar, pero falla en la mayoría de los casos.")
fuente("src/output/diagnosticos_test.json (n_player_brings, lead_pair_acc, full_bring_permissive_acc), escrito por "
       "python src/main.py test con src/training/eval/diagnosticos.py sobre los 3 modelos src/output/bc_2026111*_s*/model.npz. "
       "Pisos de azar: 1/15 y 1/90 (docs/EXPERIMENTS.md §6).")

doc.add_heading("5. Exploración y explotación", 1)
fig_("fig4_curvas.png", "Figura 4. Entrenamiento: win rate en val, entropía y KL a ρ_BC por episodio. "
     "Azul: β_H = 0,01 (3 semillas); naranja: β_H = 0 (semilla 0). Línea punteada: 15 000 episodios.")
p("A unos 15 000 episodios, con la misma semilla, β_H = 0 está en 0,525 y β_H = 0,01 en 0,539; los mejores "
  "checkpoints llegan a 0,623 y 0,616. Una evaluación en val tiene un error estándar de unos 0,04, así que ninguna "
  "diferencia es medible. La entropía es casi igual en las dos configuraciones y baja poco (de unos 4,15 a 4,0, "
  "con un máximo posible de ln 90 ≈ 4,50): con este peso, el término de entropía casi no influye. El KL a ρ_BC "
  "sube hasta 0,09 a 0,11: el agente se aleja poco del prior humano. El barrido es de una sola semilla.")
fuente("curves.csv de src/output/rl_20261113-141045_s0, rl_20261113-202239_s1, rl_20261114-130136_s2 (β_H = 0,01) y "
       "rl_20261115-015424_s0 (β_H = 0, python src/main.py rl --beta-H 0 --seed 0; el valor está en config.beta_H de "
       "su meta.json). Columnas val_win_rate, entropy y kl_to_bc; la fila de 14 080 episodios es la evaluación en val "
       "más cercana a 15 000. El error estándar de unos 0,04 sale de val_sd_pairs (≈ 0,12) dividido entre √10. En la "
       "figura, entropía y KL van con media móvil de 20 actualizaciones; el win rate no está suavizado.")

doc.add_heading("6. Validación contra test", 1)
fig_("fig5_val_vs_test.png", "Figura 5. Win rate de cada método en val (10 pares) y en test (20 pares).")
p("En val, Usage-argmax quedaba debajo de Random (0,467 contra 0,553); en test queda arriba (0,531 contra 0,492). "
  "Ese resultado de val no se repitió, así que probablemente era ruido de tener solo 10 pares. Los niveles absolutos cambian entre particiones (BC "
  "sube 8 puntos), por eso las conclusiones se basan en diferencias pareadas sobre los mismos pares.")
fuente("val: src/output/tabla_val_vs_usage_argmax.md (columna media, corridas del 13 y 14-11-2026); test: "
       "src/output/tabla_test_vs_usage_argmax.md.")

doc.add_heading("7. Conclusiones", 1)
doc.add_heading("Se puede afirmar", 2)
for t in ["Con la política de batalla fija en los dos lados, el agente RL gana 59,5 % contra Usage estocástico en "
          "equipos nunca vistos y cumple la regla preregistrada del 55 %.",
          "Gana más que Random (+10 puntos) y que Usage-argmax (+6,5 puntos); en los dos casos el intervalo excluye el 0.",
          "El RL aporta unos +3,4 puntos sobre BC, en el límite de lo medible.",
          "El resultado varía poco entre semillas (desviación estándar 0,013)."]:
    doc.add_paragraph(t, style="List Number")
doc.add_heading("No se puede afirmar", 2)
for t in ["Que el agente supera a una heurística de tipos simple: la diferencia es compatible con 0.",
          "Que los brings son buenos en juego humano: todo se mide con la misma heurística de batalla, y el agente "
          "puede haber aprendido a explotarla. La Etapa 3 lo prueba con otra política de batalla (MaxBasePower).",
          "Nada sobre otros rivales: hay uno solo.",
          "Que la entropía ayuda o estorba: no hay diferencia medible y el barrido es de una semilla.",
          "Que v-hat sirve para planificar: el planificador queda 5,5 puntos debajo de BC."]:
    doc.add_paragraph(t, style="List Number")
doc.add_heading("Limitaciones", 2)
p("Una sola política de batalla y un solo rival; política lineal (interacciones solo aditivas); 3 semillas y 20 pares "
  "de test; baselines deterministas evaluados una vez; checkpoint elegido en val; barrido de una semilla; los "
  "diagnósticos miden la imitación del humano, no la calidad del agente; la comparación de v-hat con Oblander no es directa.")

doc.add_heading("Anexo: dónde está cada número y cómo reproducirlo", 1)
p("Las rutas son relativas a la raíz del repositorio BringPredictorVGC. Los números de este documento se leyeron de "
  "esos archivos con el script Research/Etapa2_resultados/generar_informe.py (correrlo con un Python que tenga matplotlib y python-docx); ninguno se escribió a mano.")
tabla_doc(["Dato", "Archivo", "Cómo se produjo"], [
    ["Win rates de test, intervalos y veredicto", "src/output/tabla_test_vs_*.md", "python src/main.py test (una vez, 15-11-2026)"],
    ["Win rate por par y por semilla", "src/output/arena_<método>_vs_usage_test_*/per_pair.csv", "la misma corrida de test"],
    ["Checkpoints usados en test", "config.checkpoint en arena_*_test_*/meta.json", "la misma corrida de test"],
    ["Diagnósticos de BC", "src/output/diagnosticos_test.json", "la misma corrida de test"],
    ["Curvas de entrenamiento", "src/output/rl_*/curves.csv", "python src/main.py rl (13 y 14-11) y rl --beta-H 0 --seed 0 (15-11)"],
    ["Resultados de val", "src/output/tabla_val_vs_usage_argmax.md", "python src/main.py val y rl (13 y 14-11)"],
    ["Regla preregistrada", "docs/EXPERIMENTS.md", "commit 2e7afb3, antes de test"],
    ["Verificación del parser", "src/tests/fixtures/annotated_replays.json", "pytest src/tests/test_parser_annotated.py: 20 de 20"],
    ["Resumen en texto", "docs/RESULTADOS_ETAPA2.md", "commit 513505e"],
])
p("Figuras: fig1 a fig5 en esta misma carpeta, generadas por el mismo script con matplotlib. En la figura 3, el "
  "naranja marca los pares en que el agente pierde contra Tipos.")

doc.save(OUT / "Etapa2_resultados.docx")
print("ok", OUT)
