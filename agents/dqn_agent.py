import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from typing import Optional
from buffer.buffer import ReplayBuffer
from models.dqn_model import DQN
from models.dueling_dqn_model import DuelingDQN


# 定義 DQN Agent
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        algorithm: str = "dqn",
        **kwargs,
    ):
        """
        Initalize the DQN Agent.
        :param state_dim: Dimension of the state space
        :param action_dim: Number of possible actions
        :param device: Device for computation ('cpu' or 'cuda')
        :param algorithm: Algorithm type ('dqn', 'ddqn', 'dueling_dqn')
        :param kwargs: Additional hyperparameters for the agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.algorithm = algorithm.lower()
        if self.algorithm not in ["dqn", "ddqn", "dueling_dqn"]:
            raise ValueError(
                f"Unsupported algorithm: {self.algorithm}. Supported algorithms are 'dqn', 'ddqn', 'dueling_dqn'."
            )

        # 讀取超參數 (可從 config 傳入)
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon_start = kwargs.get("epsilon_start", 0.9)
        self.epsilon_end = kwargs.get("epsilon_end", 0.05)
        self.epsilon_decay = kwargs.get("epsilon_decay", 1000)
        self.memory_size = kwargs.get("memory_size", 10000)
        self.batch_size = kwargs.get("batch_size", 64)
        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.target_update = kwargs.get("target_update", 1000)
        self.hidden_dim = kwargs.get("hidden_dim", 128)
        # Q 網絡
        if self.algorithm == "dueling_dqn":
            self.policy_net = DuelingDQN(state_dim, action_dim, self.hidden_dim).to(
                device
            )
            self.target_net = DuelingDQN(state_dim, action_dim, self.hidden_dim).to(
                device
            )
        else:
            self.policy_net = DQN(state_dim, action_dim, self.hidden_dim).to(device)
            self.target_net = DQN(state_dim, action_dim, self.hidden_dim).to(device)

        self.update_target_network()
        self.target_net.eval()  # 設置為評估模式

        # 當前狀態
        self.epsilon = self.epsilon_start
        self.steps_done = 0

        # 優化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # 回放緩衝區
        self.memory = ReplayBuffer(self.memory_size)

    def update_target_network(self):
        """將策略網絡的權重複製到目標網絡"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """使用epsilon-greedy策略選擇動作"""
        # 逐步減小 epsilon 值
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if eval_mode:
            # 在評估模式下，我們總是選擇最佳動作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

        # Epsilon-greedy 動作選擇
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        return random.randrange(self.action_dim)

    def learn(self) -> Optional[float]:
        """從回放緩衝區更新策略網絡"""
        if not self.memory.can_provide_sample(self.batch_size):
            return None

        # 採樣一批經驗
        states, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # 計算當前 Q 值：Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            if self.algorithm == "dueling_dqn":
                # Dueling DQN 用policy_net計算Q值
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # DQN 和 DDQN 用target_net計算Q值
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            # 計算目標 Q 值：r + γ * max_a' Q_target(s', a')
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 計算 Huber 損失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # 優化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，以防止爆炸梯度
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "epsilon": self.epsilon,
                "algorithm": self.algorithm,
            },
            path,
        )

    def load(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, weights_only=False)
        if checkpoint.get("algorithm", "dqn") != self.algorithm:
            print(
                f"Warning: Loading checkpoint from {checkpoint.get('algorithm')} into {self.algorithm} agent."
            )

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint.get("steps_done", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
