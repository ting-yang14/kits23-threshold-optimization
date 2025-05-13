import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_agent(env, agent):
    """評估 DQN 代理"""
    state, _ = env.reset()
    reward = 0
    done = False
    truncated = False
    ground_truth = []
    predictions = []

    while not (done or truncated):
        action = agent.get_action(state, eval_mode=True)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        reward += reward
        ground_truth.append(info["ground_truth"])
        predictions.append(info["prediction"])

    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)

    return (
        reward,
        accuracy,
        precision,
        recall,
        f1,
        ground_truth,
        predictions,
    )
