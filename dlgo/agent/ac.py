import numpy as np
from keras.optimizer_v2.gradient_descent import SGD

from dlgo import encoders, goboard, kerasutil
from dlgo.agent.base import Agent
from dlgo.agent.helpers_fast import is_point_an_eye
from dlgo.rl.experience import prepare_experience_data


class ACAgent(Agent):
    def __init__(self, model, encoder, collector=None):
        self.model = model
        self.encoder = encoder
        self.collector = collector

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()

        h5file['encoder'].attrs['board_width'] = self.encoder.board_width

        h5file['encoder'].attrs['board_height'] = self.encoder.board_height

        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def _clip_probs(original_probs):
        min_p = 1e-5
        max_p = 1 - min_p
        clipped_probs = np.clip(original_probs, min_p, max_p)
        clipped_probs = clipped_probs / np.sum(clipped_probs)
        return clipped_probs

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * \
                    self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        X = np.array([board_tensor])

        actions, values = self.model.predict(X)
        move_probs = actions[0]
        estimated_value = values[0][0]

        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            move = goboard.Move.play(point)
            move_is_valid = game_state.is_valid_move(move)
            fills_own_eye = is_point_an_eye(
                game_state.board, point,
                game_state.next_player)
            if move_is_valid and (not fills_own_eye):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value
                    )
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

    def train(self, experience, lr=0.1, batch_size=128):
        opt = SGD(lr=lr)
        self.model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'mse'],
            loss_weights=[1.0, 0.5])

        n = experience.states.shape[0]
        num_moves = self.encoder.num_points()
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))
        for i in range(n):
            action = experience.actions[i]
            policy_target[i][action] = experience.advantages[i]
            reward = experience.rewards[i]
            value_target[i] = reward

        self.model.fit(
            experience.states,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1)


def load_policy_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    return PolicyAgent(model, encoder)
