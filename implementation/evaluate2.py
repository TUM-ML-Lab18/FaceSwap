import numpy as np

from Configuration.config_model import current_config
from Evaluator.Evaluator import Evaluator

if __name__ == '__main__':
    scores = Evaluator.evaluate_model(current_config)

    sim = [s['sim'] for _, s in scores.items()]
    emo = [s['emo'] for _, s in scores.items()]

    sim = np.array(sim)
    emo = np.array(emo)

    alpha = 6
    beta = 1
    scores = 1 / (1 + np.exp(alpha * emo - beta * (sim - 0.6)))

    print("--- 6-1 score ---")
    print(f"Average score: {scores.mean()}")
    print(f"Std: {scores.std()}")