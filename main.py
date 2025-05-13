import os
import sys
import torch
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 添加專案根目錄到路徑
# import setup_path

from envs.custom_env import ProbabilityThresholdEnv
from agents.dqn_agent import DQNAgent
from train.train_dqn import train_dqn
from evaluate.eval_policy import evaluate_agent
from utils.plotting import plot_metrics_with_rewards


def load_config(config_path="configs/dqn_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 載入配置
    config = load_config()

    # 創建環境
    train_env = ProbabilityThresholdEnv(
        csv_path=config["env"]["train_data"],
        rf_bounds=config["env"]["rf_bounds"],
        xgb_bounds=config["env"]["xgb_bounds"],
        svm_bounds=config["env"]["svm_bounds"],
        step_size=config["env"]["step_size"],
        isTrain=True,
        num_clf=config["env"]["num_clf"],
        max_n_steps=config["env"]["max_n_steps"],
        random_seed=config["env"]["random_seed"],
    )

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
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    # 創建 DQN 代理
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    agent = DQNAgent(state_dim, action_dim, device)

    # 訓練代理
    rewards, accuracies, precisions, recalls, f1s = train_dqn(
        train_env,
        agent,
        num_episodes=config["train"]["num_episodes"],
        max_steps=config["train"]["max_steps"],
        learn_interval=config["train"]["learn_interval"],
    )

    # 確保模型目錄存在
    os.makedirs("models", exist_ok=True)

    # 儲存時間戳
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存經過訓練的代理
    model_path = f"models/dqn_model_{run_id}.pth"
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # 評估代理
    reward, accuracy, precision, recall, f1, ground_truth, predictions = evaluate_agent(
        test_env, agent
    )

    # 確保logs目錄存在
    os.makedirs("logs", exist_ok=True)

    # 繪製學習曲線
    fig = plot_metrics_with_rewards(rewards, accuracies, precisions, recalls, f1s)
    curve_path = f"logs/learning_curve_{run_id}.png"
    plt.savefig(curve_path)
    print(f"Learning curve saved to {curve_path}")

    # 記錄結果
    results = {
        "run_id": run_id,
        "model_path": model_path,
        "training_reward": rewards,
        "training_accuracy": accuracies,
        "training_precision": precisions,
        "training_recall": recalls,
        "training_f1": f1s,
        "test_reward": reward,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "num_episodes": config["train"]["num_episodes"],
        "max_steps": config["train"]["max_steps"],
        "config": config,
    }

    # 儲存結果
    with open(f"logs/training_results_{run_id}.json", "w") as f:
        import json

        json.dump(results, f, indent=4)

    print(f"Training results saved to logs/training_results_{run_id}.json")


if __name__ == "__main__":
    main()
