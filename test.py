import os
import sys
import torch
import yaml
import json
from datetime import datetime

# 添加專案根目錄到路徑
# import setup_path

from envs.custom_env import ProbabilityThresholdEnv
from agents.dqn_agent import DQNAgent
from evaluate.eval_policy import evaluate_agent


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device)
    agent.load(model_path)

    # 評估模型
    avg_reward, std_reward = evaluate_agent(
        test_env, agent, config["eval"]["num_episodes"]
    )

    # 記錄結果
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model_path": model_path,
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "timestamp": run_id,
    }

    # 確保logs目錄存在
    os.makedirs("logs", exist_ok=True)

    # 儲存結果
    with open(f"logs/test_results_{run_id}.json", "w") as f:
        json.dump(results, f, indent=4)

    return avg_reward, std_reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型路徑")
    args = parser.parse_args()

    test_trained_model(args.model)
