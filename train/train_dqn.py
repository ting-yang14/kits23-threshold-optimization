import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_path


def train_dqn(env, agent, num_episodes=1000, max_steps=500, learn_interval=10):
    """訓練 DQN 代理"""
    rewards = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_ground_truth = []
        episode_predictions = []
        for step in range(max_steps):
            # 選擇動作
            action = agent.get_action(state)

            # 執行動作
            next_state, reward, done, truncated, info = env.step(action)

            # 保存經驗
            agent.memory.push(state, action, next_state, reward, done)

            # 更新狀態和累積獎勵
            state = next_state
            episode_reward += reward
            episode_ground_truth.append(info["ground_truth"])
            episode_predictions.append(info["prediction"])
            # 更新模型
            if step % learn_interval == 0:
                loss = agent.learn()

            # 更新目標網絡
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()

            if done or truncated:
                print(f"Episode {episode} finished after {step + 1} steps")
                break

        # 計算評估指標
        episode_accuracy = accuracy_score(episode_ground_truth, episode_predictions)
        episode_f1 = f1_score(
            episode_ground_truth, episode_predictions, zero_division=0
        )
        episode_precision = precision_score(
            episode_ground_truth, episode_predictions, zero_division=0
        )
        episode_recall = recall_score(
            episode_ground_truth, episode_predictions, zero_division=0
        )

        rewards.append(episode_reward)
        accuracies.append(episode_accuracy)
        precisions.append(episode_precision)
        recalls.append(episode_recall)
        f1s.append(episode_f1)
        # 打印進度
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_accuracy = np.mean(accuracies[-10:])
            avg_precision = np.mean(precisions[-10:])
            avg_recall = np.mean(recalls[-10:])
            avg_f1 = np.mean(f1s[-10:])

            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                f"Avg Accuracy: {avg_accuracy:.2f}, Avg Precision: {avg_precision:.2f}, "
                f"Avg Recall: {avg_recall:.2f}, Avg F1: {avg_f1:.2f}"
            )

            # 儲存訓練記錄
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                "episode": episode,
                "avg_reward": float(avg_reward),
                "avg_accuracy": float(avg_accuracy),
                "avg_precision": float(avg_precision),
                "avg_recall": float(avg_recall),
                "avg_f1": float(avg_f1),
                "epsilon": float(agent.epsilon),
                "timestamp": run_id,
            }

            # 確保logs目錄存在
            os.makedirs("logs", exist_ok=True)

            # 儲存結果
            with open(f"logs/training_progress_{run_id}.json", "w") as f:
                json.dump(results, f, indent=4)

    return rewards, accuracies, precisions, recalls, f1s
