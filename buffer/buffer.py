from collections import deque
import random
import numpy as np
import torch
from collections import namedtuple

# 定義 Experience 命名元組來存儲經驗樣本
Experience = namedtuple(
    "Experience", ("state", "action", "next_state", "reward", "done")
)


# 定義經驗回放緩衝區
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """保存一個轉換"""
        self.memory.append(Experience(state, action, next_state, reward, done))

    def sample(self, batch_size):
        """隨機抽取一批經驗"""
        experiences = random.sample(self.memory, batch_size)

        # 將樣本轉換為張量批次
        states = torch.FloatTensor(np.array([exp.state for exp in experiences]))
        actions = torch.LongTensor(
            np.array([exp.action for exp in experiences]).reshape(-1, 1)
        )
        next_states = torch.FloatTensor(
            np.array([exp.next_state for exp in experiences])
        )
        rewards = torch.FloatTensor(
            np.array([exp.reward for exp in experiences]).reshape(-1, 1)
        )
        dones = torch.FloatTensor(
            np.array([exp.done for exp in experiences]).reshape(-1, 1)
        )

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.memory)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
