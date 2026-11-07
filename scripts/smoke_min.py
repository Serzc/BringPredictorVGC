import asyncio
from pathlib import Path
from poke_env.player import RandomPlayer

FMT = "gen9championsvgc2026regma"
teams = Path("data/sample_teams.txt").read_text(encoding="utf-8").splitlines()

async def main():
    p1 = RandomPlayer(battle_format=FMT, team=teams[0])
    p2 = RandomPlayer(battle_format=FMT, team=teams[1])
    await p1.battle_against(p2, n_battles=1000)
    print("p1 won", p1.n_won_battles, "of", p1.n_finished_battles)

asyncio.run(main())