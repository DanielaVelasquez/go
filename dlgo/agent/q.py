import numpy as np
from keras.optimizer_v2.gradient_descent import SGD

from dlgo import goboard
from dlgo.agent.base import Agent
from dlgo.agent.helpers_fast import is_point_an_eye


class QAgent(Agent):
    def __init__(self, model, encoder, temperature=0, collector=None):
        self.model = model
        self.encoder = encoder
        self.collector = collector
        self.temperature = temperature

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue
            moves.append(self.encoder.encode_point(move.point))
            board_tensors.append(board_tensor)

        if not moves:
            return goboard.Move.pass_turn()

        num_moves = len(moves)
        board_tensors = np.array(board_tensors)
        move_vectors = np.zeros((num_moves, self.encoder.num_points()))

        for i, move in enumerate(moves):
            move_vectors[i][move] = 1

        values = self.model.predict([board_tensors, move_vectors])
        values = values.reshape(len(moves))

        ranked_moves = self.rank_moves_eps_greedy(values)
        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(moves[move_idx])
            if not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(state=board_tensor, action=moves[move_idx])
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            values = np.random.random(values.shape)
        # This ranks the moves from worst to best.
        ranked_moves = np.argsort(values)
        # Return them in best-to-worst order.
        return ranked_moves[::-1]

    def train(self, experience, lr=0.1, batch_size=128):
        opt = SGD(lr=lr)
        self.model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        y = np.zeros((n,))
        actions = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            actions[i][action] = 1
            y[i] = reward

        self.model.fit(
            [experience.states, actions], y,
            batch_size=batch_size,
            epochs=1)
