import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse

from agents.dqn_agent import DQNAgent
from train.train_dqn import train_dqn
from evaluate.eval_policy import evaluate_agent
from utils.plotting import plot_metrics_with_rewards, plot_confusion_matrix
from utils.summarize import summarize_training_results
from utils.utils import (
    load_config,
    create_env,
    ensure_log_dirs_exist,
    save_results_to_json,
)


def main(config_file="configs/dqn_config.yaml", algorithm="dqn"):
    # 確保logs目錄存在
    ensure_log_dirs_exist()

    # 載入配置
    config = load_config(config_file)
    config["algorithm"] = algorithm
    # 創建環境
    train_env = create_env(config, is_train=True)

    # 創建測試環境
    test_env = create_env(config, is_train=False)

    # 獲取環境的維度
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    # 創建 DQN 代理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agent = DQNAgent(
        state_dim, action_dim, device, algorithm=algorithm, **config["agent"]
    )

    # 訓練代理
    rewards, accuracies, precisions, recalls, f1s = train_dqn(
        train_env,
        agent,
        num_episodes=config["train"]["num_episodes"],
        max_steps=config["train"]["max_steps"],
        learn_interval=config["train"]["learn_interval"],
    )

    # 儲存時間戳
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存經過訓練的代理
    model_path = f"logs/checkpoints/{algorithm}_{run_id}.pth"
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    learning_curve_path = f"logs/train/{algorithm}_{run_id}_learning_curve.png"

    # 繪製學習曲線
    plot_metrics_with_rewards(
        rewards,
        accuracies,
        precisions,
        recalls,
        f1s,
        save_path=learning_curve_path,
        show=False,
    )
    print(f"Learning curve saved to {learning_curve_path}")

    # 評估代理
    (
        test_reward,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        ground_truth,
        predictions,
    ) = evaluate_agent(test_env, agent)

    confusion_matrix_path = f"logs/test/{algorithm}_{run_id}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=ground_truth,
        y_pred=predictions,
        save_path=confusion_matrix_path,
        show=False,
    )
    print(f"Confusion Matrix saved to {confusion_matrix_path}")

    # 記錄結果
    results = {
        "run_id": run_id,
        "model_path": model_path,
        "training_reward": rewards,
        "training_accuracy": accuracies,
        "training_precision": precisions,
        "training_recall": recalls,
        "training_f1": f1s,
        "test_reward": test_reward,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "config": config,
    }

    # 儲存結果
    training_results_path = f"logs/train/{algorithm}_{run_id}_train_results.json"
    save_results_to_json(results, training_results_path)
    print(f"Training results saved to {training_results_path}")

    time.sleep(3)  # 等待文件系統穩定
    # 總結訓練結果
    summarize_training_results("logs/train", "training_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dqn_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dqn",
        choices=["dqn", "ddqn", "dueling_dqn"],
        help="Algorithm type",
    )
    args = parser.parse_args()
    main(config_file=args.config, algorithm=args.algorithm)

# sample command to run the script
# python main.py --config configs/dqn_config.yaml --algorithm dqn
