"""BringPlayer: el jugador de batalla comun pi_b. Solo el Team Preview cambia entre metodos.

La interfaz de selección de bring se conserva desde la Fase 0.
- teampreview(): pregunta a `chooser` (lo pone la arena) que bring jugar y manda '/team abcd'.
- choose_move(): SimpleHeuristicsPlayer del fork de vgc-bench + auto_mega. En la Fase 0 se
  vio que SimpleHeuristicsPlayer nunca megaevoluciona; auto_mega marca la mega en el
  primer movimiento de ataque que la permita. Es un cambio a pi_b y se declara (docs/MDP.md).
La misma clase juega en los dos lados, asi que pi_b es identica por construccion.
"""
from __future__ import annotations

from typing import Callable

from poke_env.battle.move import Move
from poke_env.player import SimpleHeuristicsPlayer

from utils.brings import bring_to_team_order


class BringPlayer(SimpleHeuristicsPlayer):
    def __init__(self, *args, auto_mega: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_mega = auto_mega
        self.chooser: Callable[[], int] = lambda: 0      # devuelve el bring_id a jugar
        self.chosen: dict[str, int] = {}                 # battle_tag -> bring_id jugado

    def teampreview(self, battle):
        k = int(self.chooser())
        self.chosen[battle.battle_tag] = k
        return "/team " + bring_to_team_order(k)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        if not self.auto_mega or not hasattr(order, "first_order"):
            return order
        # orden doble: megaevolucionar con el primer slot que ataque y pueda hacerlo
        for slot, single in enumerate((order.first_order, order.second_order)):
            if (single is not None and isinstance(getattr(single, "order", None), Move)
                    and battle.can_mega_evolve[slot]):
                single.mega = True
                break
        return order
