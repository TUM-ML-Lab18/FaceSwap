import numpy as np
from Evaluator.Evaluator import Evaluator
from configuration.run_config import current_config


if __name__ == '__main__':
    model = current_config['model'](current_config['img_size'])
    scores = Evaluator.evaluate_model(current_config)

    sim = [s['sim'] for _,s in scores.items()]
    emo = [s['emo'] for _,s in scores.items()]

    sim = np.array(sim)
    emo = np.array(emo)

    alpha = 1
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("1-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")

    alpha = 2
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("2-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")

    alpha = 4
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("2-1")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")