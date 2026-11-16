"""Extracts teams (packed format) from the Reg M-A log set for testing.

Imputacion TEMPORAL de stats para el smoke test:
los logs traen vacios naturaleza y stat points (A8 del plan). Aqui se rellena
cada Pokemon con una naturaleza neutra y un reparto aleatorio de stat points.
La regla definitiva de A8 la deciden y justifican Sergio y Jose.

Limites verificados en el Showdown fijado (cameronangliss/pokemon-showdown@913da36,
sim/team-validator.ts y sim/dex-formats.ts): <= 32 stat points por stat, 66 en total.

Uso: python scripts/extract_teams.py [N] [SEED]
"""
import json
import random
import re
import sys
from pathlib import Path

LOGS = Path(r"C:\Users\XPC\Documents\TEC\IA\Research Proyecto\data\logs_gen9championsvgc2026regma.json")
OUT = Path("data/sample_teams.txt")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

MAX_PER_STAT = 32
TOTAL = 66
# Naturalezas neutras (sin +/-). Serious se excluye: el validador la usa como
# senal de "0 stat points intencional".
NEUTRAL_NATURES = ["Hardy", "Docile", "Bashful", "Quirky"]

# Campos del formato packed de Showdown:
# name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|extra
NATURE, EVS = 5, 6

rng = random.Random(SEED)


def random_spread() -> list[int]:
    """Reparte TOTAL stat points al azar entre 6 stats, con tope por stat."""
    spread = [0] * 6
    for _ in range(TOTAL):
        stat = rng.choice([i for i in range(6) if spread[i] < MAX_PER_STAT])
        spread[stat] += 1
    return spread  # orden: hp, atk, def, spa, spd, spe


def impute(packed_team: str) -> str:
    mons = []
    for mon in packed_team.split("]"):
        fields = mon.split("|")
        if not fields[NATURE]:
            fields[NATURE] = rng.choice(NEUTRAL_NATURES)
        if not fields[EVS]:
            fields[EVS] = ",".join(str(x) for x in random_spread())
        mons.append("|".join(fields))
    return "]".join(mons)

data = json.loads(LOGS.read_text(encoding="utf-8"))
pat = re.compile(r"^\|showteam\|p[12]\|(.+)$", re.M)

teams = []
for battle_id, (_ts, log) in data.items():
    for packed in pat.findall(log):
        if packed not in teams:
            teams.append(packed)
    if len(teams) >= N:
        break

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(impute(t) for t in teams[:N]), encoding="utf-8")
print(f"{len(teams[:N])} teams -> {OUT} (seed {SEED})")