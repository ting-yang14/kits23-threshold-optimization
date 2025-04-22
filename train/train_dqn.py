import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import setup_path

from agents.dqn_agent import DQNAgent
from envs.custom_env import ProbabilityThresholdEnv
from evaluate.eval_policy import evaluate_agent


def train_dqn(env, agent, num_episodes=1000, max_steps=500, learn_interval=10):
    """訓練 DQN 代理"""
    rewards = []

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 選擇動作
            action = agent.get_action(state)

            # 執行動作
            next_state, reward, done, truncated, _ = env.step(action)

            # 保存經驗
            agent.memory.push(state, action, next_state, reward, done)

            # 更新狀態和累積獎勵
            state = next_state
            episode_reward += reward

            # 更新模型
            if step % learn_interval == 0:
                loss = agent.learn()

            # 更新目標網絡
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()

            if done or truncated:
                print(f"Episode {episode} finished after {step + 1} steps")
                break

        rewards.append(episode_reward)

        # 打印進度
        if episode % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
            )

            # 儲存訓練記錄
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                "episode": episode,
                "avg_reward": float(avg_reward),
                "epsilon": float(agent.epsilon),
                "timestamp": run_id,
            }

            # 確保logs目錄存在
            os.makedirs("logs", exist_ok=True)

            # 儲存結果
            with open(f"logs/training_progress_{run_id}.json", "w") as f:
                json.dump(results, f, indent=4)

    return rewards
