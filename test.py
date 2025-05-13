import os
import sys
import torch
import yaml
import json
import argparse
from datetime import datetime

# 添加專案根目錄到路徑
# import setup_path

from envs.custom_env import ProbabilityThresholdEnv
from agents.dqn_agent import DQNAgent
from evaluate.eval_policy import evaluate_agent
from utils.plotting import plot_confusion_matrix


def load_config(config_path="configs/dqn_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test_trained_model(model_path):
    # 載入配置
    config = load_config()

    # 創建測試環境
    test_env = ProbabilityThresholdEnv(
        csv_path=config["env"]["test_data"],
        rf_bounds=config["env"]["rf_bounds"],
        xgb_bounds=config["env"]["xgb_bounds"],
        svm_bounds=config["env"]["svm_bounds"],
        step_size=config["env"]["step_size"],
        isTrain=False,
        num_clf=config["env"]["num_clf"],
        max_n_steps=config["env"]["max_n_steps"],
        random_seed=config["env"]["random_seed"],
    )

    # 獲取環境的維度
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n

    # 創建並載入模型
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    agent = DQNAgent(state_dim, action_dim, device)
    agent.load(model_path)

    # 評估模型
    reward, accuracy, precision, recall, f1, ground_truth, predictions = evaluate_agent(
        test_env, agent
    )

    results = {
        "model_path": model_path,
        "reward": reward,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ground_truth": ground_truth,
        "predictions": predictions,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model (.pth)"
    )
    args = parser.parse_args()

    results = test_trained_model(args.model)

    print(
        f"Testing Results - Reward: {results['reward']:.2f}, Accuracy: {results['accuracy']:.2f}, "
        f"Precision: {results['precision']:.2f}, Recall: {results['recall']:.2f}, F1: {results['f1']:.2f}"
    )

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    os.makedirs("logs/test", exist_ok=True)
    test_results_path = os.path.join("logs", "test", f"test_results_{model_name}.json")
    with open(test_results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Test results saved to {test_results_path}")
    # 繪製混淆矩陣
    save_file = os.path.join("logs", "test", f"{model_name}_confusion_matrix.png")

    plot_confusion_matrix(
        y_true=results["ground_truth"],
        y_pred=results["predictions"],
        title=f"Confusion Matrix - {model_name}",
        save_file=save_file,
    )


if __name__ == "__main__":
    main()

# sample command to run the script
# python test.py --model checkpoints/dqn_model_20250513_172716.pth
