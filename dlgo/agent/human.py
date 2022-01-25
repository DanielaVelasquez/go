from dlgo.agent.base import Agent
from dlgo.utils import point_from_coords
from dlgo.goboard import Move


class Human(Agent):
    def select_move(self, game_state):
        human_move = input('-- ')
        point = point_from_coords(human_move.strip())
        return Move.play(point)
