"""v-hat: evaluador de matchup bilineal antisimetrico, eqs. (8)-(11) del survey (numpy, gradientes a mano).

Version NumPy del modelo bilineal, migrada desde la implementacion Torch de `d3b1d7d`.
Solo lo usan el Planificador-v-hat (baselines/planner.py) y los diagnosticos. NUNCA es la
recompensa del agente RL (A2): evaluar con lo mismo que se optimiza seria circular.

Forma del modelo (el survey la define con eq:bilinear y las dos ecuaciones de h y f que le siguen):
  phi_x = MLP([E_especie(x), E_objeto(x), E_habilidad(x), media_mov E_mov(m)])      (un vector de d = 64 por Pokemon)
  h(b)  = ReLU(beta_h + W_L sum_{x en L} phi_x + W_R sum_{x en R} phi_x)           (hidden = 64)
  f(b)  = sigmoid(beta_z + W_z h(b))                                               (K = 16)
  v(b1, b2) = sigmoid( f(b1)^T M f(b2) ),   M = A - A^T                            (eq:bilinear, eq. 8)
  Pr(w=1 | o1, o2) = sum_{b1 en C(o1)} sum_{b2 en C(o2)} q(b1) q(b2) v(b1, b2)     (eq:marginal, eq. 11)
con q uniforme sobre cada C(o) (a lo sumo 6 x 6 = 36 terminos). Como M es antisimetrica,
v(b1,b2) + v(b2,b1) = 1 EXACTO por construccion.

Decision de proyecto (lo unico que NO sale del survey): phi_x. El survey dice "embedding layer
sobre especie, movimientos, objeto y habilidad"; aqui se concatenan las cuatro tablas de embeddings
(32 + 16 + 16 + 16) y se pasan por un MLP de 2 capas hacia d = 64, igual que la version torch (MonEmbed).
phi_x NO usa features de uso (como el survey). Sin dropout (en la version torch era 0.1): se usa L2.
"""
from __future__ import annotations

import numpy as np

from models.bc import Adam
from utils.brings import LEAD_MASK, RES_MASK

FORMATO = "vhat_bilineal_v1"             # marca del checkpoint; el formato logistico viejo no la tiene
DIMS_EMB = {"species": 32, "item": 16, "ability": 16, "move": 16}   # tamano de cada tabla de embeddings
N_MOV = 4                                # movimientos por Pokemon (los que falten quedan en <PAD> = 0)
D_ENTRADA = sum(DIMS_EMB.values())       # 80: la concatenacion de los cuatro embeddings


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Parametros: un diccionario de arreglos (cada uno es un "grupo" para el test de gradientes)
# ---------------------------------------------------------------------------
def nombres_parametros() -> list[str]:
    """Orden fijo de los grupos de parametros (sirve para aplanarlos y dárselos a Adam)."""
    return ["emb_especie", "emb_objeto", "emb_habilidad", "emb_movimiento",   # tablas de embeddings (una fila por token)
            "W1", "b1", "W2", "b2",                                           # MLP que produce phi_x
            "W_L", "W_R", "beta_h",                                           # h(b): proyeccion de lideres, de reservas y su sesgo
            "W_z", "beta_z",                                                  # f(b): proyeccion a K dimensiones y su sesgo
            "A"]                                                              # M = A - A^T, la matriz de matchup antisimetrica


# Grupos sobre los que se aplica L2 (los sesgos y la fila <PAD> no se regularizan)
SIN_L2 = {"b1", "b2", "beta_h", "beta_z"}


def iniciar_parametros(tamanos: dict[str, int], d: int, hidden: int, K: int, rng: np.random.Generator) -> dict:
    """Inicializacion igual a la version torch: embeddings N(0,1), capas U(-1/sqrt(entrada), +1/sqrt(entrada)), A N(0, 0.1)."""
    def capa(n_in, n_out):
        lim = 1.0 / np.sqrt(n_in)
        return rng.uniform(-lim, lim, (n_in, n_out)), rng.uniform(-lim, lim, n_out)

    p = {}
    for nombre, cat in (("emb_especie", "species"), ("emb_objeto", "item"),
                        ("emb_habilidad", "ability"), ("emb_movimiento", "move")):
        tabla = rng.normal(0, 1, (tamanos[cat], DIMS_EMB[cat]))
        tabla[0] = 0.0                                   # <PAD> = 0: embedding nulo y sin gradiente
        p[nombre] = tabla
    p["W1"], p["b1"] = capa(D_ENTRADA, d)
    p["W2"], p["b2"] = capa(d, d)
    p["W_L"], p["beta_h"] = capa(d, hidden)
    lim = 1.0 / np.sqrt(d)
    p["W_R"] = rng.uniform(-lim, lim, (d, hidden))       # sin sesgo propio: beta_h es unico (como el survey)
    p["W_z"], p["beta_z"] = capa(hidden, K)
    p["A"] = rng.normal(0, 0.1, (K, K))
    return p


# ---------------------------------------------------------------------------
# Paso hacia adelante: de los tokens de un equipo a f(b) de sus 90 brings
# ---------------------------------------------------------------------------
def _torre_adelante(p: dict, tok: np.ndarray):
    """tok (B, 6, 7) enteros -> f (B, 90, K) y la cache que necesita el paso hacia atras.

    tok[..., 0] = especie, 1 = objeto, 2 = habilidad, 3: = movimientos (0 = vacio).
    """
    mov = tok[..., 3:]                                                     # (B, 6, 4)
    no_vacio = (mov > 0)[..., None].astype(float)                          # (B, 6, 4, 1): 1 si el movimiento existe
    n_mov = np.maximum(no_vacio.sum(-2), 1.0)                              # (B, 6, 1): movimientos reales (para la media)
    media_mov = (p["emb_movimiento"][mov] * no_vacio).sum(-2) / n_mov      # (B, 6, 16): media de los embeddings de movimientos
    x = np.concatenate([p["emb_especie"][tok[..., 0]], p["emb_objeto"][tok[..., 1]],
                        p["emb_habilidad"][tok[..., 2]], media_mov], axis=-1)   # (B, 6, 80)

    # phi_x: MLP de dos capas por Pokemon -> (B, 6, d)
    pre1 = x @ p["W1"] + p["b1"]
    r = np.maximum(pre1, 0.0)
    phi = r @ p["W2"] + p["b2"]

    # Suma de phi dentro de cada subconjunto (lideres / reservas): invariante al orden dentro del par
    sL = LEAD_MASK @ phi                                                   # (B, 90, d): suma de phi de los 2 lideres de cada bring
    sR = RES_MASK @ phi                                                    # (B, 90, d): suma de phi de las 2 reservas
    # h(b) con proyecciones distintas para lideres y reservas (W_L != W_R los distingue)
    pre_h = sL @ p["W_L"] + sR @ p["W_R"] + p["beta_h"]
    h = np.maximum(pre_h, 0.0)
    f = _sigmoid(h @ p["W_z"] + p["beta_z"])                               # (B, 90, K)
    return f, (tok, no_vacio, n_mov, x, pre1, r, sL, sR, pre_h, h, f)


def _torre_atras(p: dict, cache, df: np.ndarray, g: dict) -> None:
    """Suma en g los gradientes de la perdida a traves de f (df = dperdida/df, (B, 90, K))."""
    tok, no_vacio, n_mov, x, pre1, r, sL, sR, pre_h, h, f = cache
    d, hid, K = p["W_L"].shape[0], p["W_L"].shape[1], p["W_z"].shape[1]

    # f = sigmoid(h W_z + beta_z)
    dz = df * f * (1 - f)
    g["W_z"] += h.reshape(-1, hid).T @ dz.reshape(-1, K)
    g["beta_z"] += dz.sum((0, 1))
    # h = ReLU(pre_h)
    dpre_h = (dz @ p["W_z"].T) * (pre_h > 0)
    g["beta_h"] += dpre_h.sum((0, 1))
    g["W_L"] += sL.reshape(-1, d).T @ dpre_h.reshape(-1, hid)
    g["W_R"] += sR.reshape(-1, d).T @ dpre_h.reshape(-1, hid)
    # cada phi_x recibe la masa de los brings donde juega de lider o de reserva
    dphi = LEAD_MASK.T @ (dpre_h @ p["W_L"].T) + RES_MASK.T @ (dpre_h @ p["W_R"].T)      # (B, 6, d)

    # MLP de phi
    g["W2"] += r.reshape(-1, d).T @ dphi.reshape(-1, d)
    g["b2"] += dphi.sum((0, 1))
    dpre1 = (dphi @ p["W2"].T) * (pre1 > 0)
    g["W1"] += x.reshape(-1, D_ENTRADA).T @ dpre1.reshape(-1, d)
    g["b1"] += dpre1.sum((0, 1))
    dx = dpre1 @ p["W1"].T                                                 # (B, 6, 80)

    # reparto del gradiente a las tablas de embeddings (las filas repetidas se suman)
    a, b, c = np.cumsum([0, *DIMS_EMB.values()])[1:4]
    np.add.at(g["emb_especie"], tok[..., 0], dx[..., :a])
    np.add.at(g["emb_objeto"], tok[..., 1], dx[..., a:b])
    np.add.at(g["emb_habilidad"], tok[..., 2], dx[..., b:c])
    np.add.at(g["emb_movimiento"], tok[..., 3:], no_vacio * (dx[..., c:] / n_mov)[..., None, :])


# ---------------------------------------------------------------------------
# Verosimilitud marginalizada (11) y su gradiente
# ---------------------------------------------------------------------------
def matriz_v(p: dict, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """(B, 90, 90): v(b1, b2) = sigmoid(f(b1)^T M f(b2)) con M = A - A^T (antisimetrica)."""
    M = p["A"] - p["A"].T
    return _sigmoid((f1 @ M) @ f2.transpose(0, 2, 1))


def perdida_marginal(p: dict, tok1, tok2, q1, q2, y, l2: float = 0.0, con_grad: bool = True):
    """NLL media de (11) + L2 y, si con_grad, su gradiente por grupo.

    tok1, tok2: (B, 6, 7) tokens de los equipos; q1, q2: (B, 90) distribucion sobre brings (uniforme
    sobre C(o), o sobre los 90 en la metrica de equipo); y: (B,) 1 si gano el lado 1.
    Devuelve (perdida, nll por ejemplo, grads | None, p = Pr(gana el lado 1)).
    """
    B = len(y)
    f1, c1 = _torre_adelante(p, tok1)
    f2, c2 = _torre_adelante(p, tok2)
    V = matriz_v(p, f1, f2)                                               # (B, 90, 90)
    pr = np.clip(np.einsum("bi,bij,bj->b", q1, V, q2), 1e-6, 1 - 1e-6)    # eq. (11): Pr(gana el lado 1)
    nll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    reg = 0.5 * l2 * sum(float((p[k] ** 2).sum()) for k in p if k not in SIN_L2)
    perdida = float(nll.mean() + reg)
    if not con_grad:
        return perdida, nll, None, pr

    # --- gradiente: dNLL/dlogit(b1,b2) = coef * q1 q2 v (1 - v), con coef = -(y/p - (1-y)/(1-p)) / B ---
    coef = -(y / pr - (1 - y) / (1 - pr)) / B
    G = coef[:, None, None] * q1[:, :, None] * q2[:, None, :] * V * (1 - V)    # (B, 90, 90) = dperdida/dlogit
    M = p["A"] - p["A"].T
    Gf2 = G @ f2                                                          # (B, 90, K)
    Gtf1 = G.transpose(0, 2, 1) @ f1
    dM = np.einsum("bik,bil->kl", f1, Gf2)                                # dperdida/dM
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["A"] = dM - dM.T                                                    # M = A - A^T  ->  dA = dM - dM^T
    _torre_atras(p, c1, Gf2 @ M.T, g)                                     # df1 = G f2 M^T
    _torre_atras(p, c2, Gtf1 @ M, g)                                      # df2 = G^T f1 M
    for k in g:
        if k not in SIN_L2:
            g[k] += l2 * p[k]
    for k in ("emb_especie", "emb_objeto", "emb_habilidad", "emb_movimiento"):
        g[k][0] = 0.0                                                     # la fila <PAD> no se entrena
    return perdida, nll, g, pr


# ---------------------------------------------------------------------------
# La clase que usan el Planificador, el entrenamiento y la arena
# ---------------------------------------------------------------------------
def aplanar(p: dict) -> np.ndarray:
    return np.concatenate([p[k].ravel() for k in nombres_parametros()])


def desaplanar(vec: np.ndarray, plantilla: dict) -> dict:
    """Inversa de aplanar: reparte el vector en arreglos con las formas de la plantilla."""
    out, i = {}, 0
    for k in nombres_parametros():
        n = plantilla[k].size
        out[k] = vec[i:i + n].reshape(plantilla[k].shape).copy()
        i += n
    return out


class VHat:
    """Evaluador v-hat bilineal con sus parametros. Interfaz para el Planificador: v_matrix(team1, team2) -> (90, 90)."""

    def __init__(self, codificador, d: int = 64, hidden: int = 64, K: int = 16, params: dict | None = None, seed: int = 0):
        self.cod = codificador
        self.d, self.hidden, self.K = d, hidden, K
        self.tamanos = {c: codificador.vocab.size(c) for c in DIMS_EMB}
        self.p = params if params is not None else iniciar_parametros(
            self.tamanos, d, hidden, K, np.random.default_rng(seed))

    def tokens(self, team: list[dict]) -> np.ndarray:
        """(6, 7) enteros del equipo: especie, objeto, habilidad y hasta 4 movimientos. <UNK> = 1 si no esta en el vocabulario de train."""
        v = self.cod.vocab
        tok = np.zeros((6, 3 + N_MOV), dtype=np.int64)
        for i, m in enumerate(team):
            tok[i, :3] = v.idx("species", m["species"]), v.idx("item", m["item"]), v.idx("ability", m["ability"])
            for j, mv in enumerate(m["moves"][:N_MOV]):
                tok[i, 3 + j] = v.idx("move", mv)
        return tok

    def v_matrix(self, team1, team2) -> np.ndarray:
        """(90, 90): v(b1, b2) = Pr(gana el equipo 1 con b1 contra el equipo 2 con b2)."""
        f1, _ = _torre_adelante(self.p, self.tokens(team1)[None])
        f2, _ = _torre_adelante(self.p, self.tokens(team2)[None])
        return matriz_v(self.p, f1, f2)[0]

    def entrenar_paso(self, opt: Adam, lote: dict, l2: float) -> float:
        """Un paso de Adam sobre un minilote (dict con tok1, tok2, q1, q2, y). Devuelve la perdida."""
        perdida, _, grad, _ = perdida_marginal(self.p, lote["tok1"], lote["tok2"], lote["q1"], lote["q2"], lote["y"], l2)
        self.p = desaplanar(opt.paso(aplanar(self.p), aplanar(grad)), self.p)
        return perdida

    def guardar(self, path) -> None:
        np.savez(path, formato=FORMATO, dims=np.array([self.d, self.hidden, self.K]),
                 tamanos=np.array([self.tamanos[c] for c in DIMS_EMB]), **self.p)

    @classmethod
    def cargar(cls, path, codificador):
        """Carga un checkpoint bilineal. Falla con un mensaje claro si es el formato logistico viejo o no calza con el vocabulario."""
        d = np.load(path)
        if "formato" not in d.files:
            raise ValueError(f"{path}: es un checkpoint de v-hat con formato logistico antiguo (claves {sorted(d.files)}). "
                             f"v-hat ahora es bilineal (eqs. 8-11): reentrenar con `python src/main.py vhat`.")
        if str(d["formato"]) != FORMATO:
            raise ValueError(f"{path}: formato '{d['formato']}' distinto de '{FORMATO}'")
        tam = {c: codificador.vocab.size(c) for c in DIMS_EMB}
        if list(d["tamanos"]) != [tam[c] for c in DIMS_EMB]:
            raise ValueError(f"{path}: el vocabulario del checkpoint {list(d['tamanos'])} no coincide con el actual {list(tam.values())}")
        dd, hh, kk = (int(x) for x in d["dims"])
        return cls(codificador, dd, hh, kk, params={k: d[k].astype(float) for k in nombres_parametros()})
