import neat
import pickle
import os

from simulation import eval_genomes


def run_training(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))

    winner = p.run(eval_genomes, 100)

    with open("models/best_bot_raycast.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Best genome saved.")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config", "feedforward.txt")
    run_training(config_path)