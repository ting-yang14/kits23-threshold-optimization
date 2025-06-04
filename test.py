import os
import torch
import json
import argparse
from datetime import datetime

# 添加專案根目錄到路徑
# import setup_path
from agents.dqn_agent import DQNAgent
from evaluate.eval_policy import evaluate_agent
from utils.plotting import plot_confusion_matrix
from utils.utils import (
    load_config,
    create_env,
    ensure_log_dirs_exist,
    save_results_to_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model (.pth)"
    )
    args = parser.parse_args()

    # 確保logs目錄存在
    ensure_log_dirs_exist()

    # 載入配置
    config = load_config()

    # 創建測試環境
    test_env = create_env(config, is_train=False)

    # 獲取環境的維度
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n

    # 創建並載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DQNAgent(state_dim, action_dim, device)
    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    reward, accuracy, precision, recall, f1, ground_truth, predictions = evaluate_agent(
        test_env, agent
    )

    results = {
        "model_path": args.model,
        "reward": reward,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ground_truth": ground_truth,
        "predictions": predictions,
    }

    print(
        f"Testing Results - Reward: {results['reward']:.2f}, Accuracy: {results['accuracy']:.2f}, "
        f"Precision: {results['precision']:.2f}, Recall: {results['recall']:.2f}, F1: {results['f1']:.2f}"
    )

    model_name = os.path.splitext(os.path.basename(args.model))[0]

    test_results_path = f"logs/test/test_results_{model_name}.json"
    save_results_to_json(results, test_results_path)
    print(f"Test results saved to {test_results_path}")
    # 繪製混淆矩陣

    confusion_matrix_path = f"logs/test/{model_name}_confusion_matrix.png"
    _ = plot_confusion_matrix(
        y_true=results["ground_truth"],
        y_pred=results["predictions"],
        save_path=confusion_matrix_path,
        show=False,
    )
    print(f"Confusion Matrix saved to {confusion_matrix_path}")


if __name__ == "__main__":
    main()

# sample command to run the script
# python test.py --model logs/checkpoints/dqn_model_20250513_172716.pth
