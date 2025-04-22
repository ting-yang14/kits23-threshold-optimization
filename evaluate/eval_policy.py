import numpy as np


def evaluate_agent(env, agent, num_episodes=10):
    """評估 DQN 代理"""
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.get_action(state, eval_mode=True)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Evaluation: Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f}")

    return avg_reward, std_reward
