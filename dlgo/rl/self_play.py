import h5py

from dlgo import scoring, rl
from dlgo.agent.policy import load_policy_agent
from dlgo.goboard import GameState
from dlgo.gotypes import Player


def simulate_game(black_player, white_player, board_size=9):
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    game_result = scoring.compute_game_result(game)
    return game_result.winner


def self_play(num_games=1):
    agent1 = load_policy_agent(h5py.File('agent1'))
    agent2 = load_policy_agent(h5py.File('agent2'))
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)
        if game_record.winner == Player.black:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        experience = rl.combine_experience([collector1, collector2])
        with h5py.File('experience', 'w') as experience_outf:
            experience.serialize(experience_outf)
