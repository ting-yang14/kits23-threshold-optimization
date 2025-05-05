import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from sklearn.utils import resample
from typing import List, Tuple, Dict
from stable_baselines3.common.vec_env import DummyVecEnv


class ProbabilityThresholdEnv(gym.Env):
    def __init__(
        self,
        csv_path: str,
        rf_bounds=(0.85, 0.90),
        xgb_bounds=(0.85, 0.90),
        svm_bounds=(0.85, 0.90),
        step_size=0.01,
        isTrain: bool = True,
        num_clf: int = 3,
        max_n_steps: int = 10000,
        random_seed: int = 42,
    ):

        super(ProbabilityThresholdEnv, self).__init__()
        self.random_seed = random_seed
        # Load data from CSV
        self.data = pd.read_csv(csv_path)
        assert all(
            col in self.data.columns for col in ["rf_prob", "xgb_prob", "svm_prob"]
        ), "CSV must contain columns: rf_prob, xgb_prob, svm_prob"

        # Ensure probability values are between 0 and 1
        for col in ["rf_prob", "xgb_prob", "svm_prob"]:
            assert (
                (self.data[col] >= 0) & (self.data[col] <= 1)
            ).all(), f"All values in {col} must be between 0 and 1"

        # Set observation space: 3 values between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Calculate threshold values
        assert (
            0 <= rf_bounds[0] < rf_bounds[1] <= 1
        ), f"Invalid rf_bounds: {rf_bounds}. Must satisfy 0 <= rf_bounds[0] < rf_bounds[1] <= 1."

        self.rf_thresholds = np.arange(rf_bounds[0], rf_bounds[1], step_size)
        assert (
            0 <= xgb_bounds[0] < xgb_bounds[1] <= 1
        ), f"Invalid xgb_bounds: {xgb_bounds}. Must satisfy 0 <= xgb_bounds[0] < xgb_bounds[1] <= 1."
        self.xgb_thresholds = np.arange(xgb_bounds[0], xgb_bounds[1], step_size)
        assert (
            0 <= svm_bounds[0] < svm_bounds[1] <= 1
        ), f"Invalid svm_bounds: {svm_bounds}. Must satisfy 0 <= svm_bounds[0] < svm_bounds[1] <= 1."
        self.svm_thresholds = np.arange(svm_bounds[0], svm_bounds[1], step_size)

        # Calculate action space size: product of threshold options
        num_rf_thresholds = len(self.rf_thresholds)
        num_xgb_thresholds = len(self.xgb_thresholds)
        num_svm_thresholds = len(self.svm_thresholds)

        self.num_clf = num_clf

        if self.num_clf == 1:
            total_actions = num_rf_thresholds + num_xgb_thresholds + num_svm_thresholds
        elif self.num_clf == 2:
            total_actions = (
                num_rf_thresholds * num_xgb_thresholds
                + num_rf_thresholds * num_svm_thresholds
                + num_xgb_thresholds * num_svm_thresholds
            )
        else:
            total_actions = num_rf_thresholds * num_xgb_thresholds * num_svm_thresholds

        # Set action space
        self.action_space = spaces.Discrete(total_actions)

        # Shuffle and balance data
        if isTrain:
            self._balance_dataset(max_n_steps)
            print(f"Balanced training dataset with {len(self.data)} samples")
        else:
            self.num_rows = len(self.data)
            print(f"Loaded testing dataset with {len(self.data)} samples")
        # Environment configuration
        self.current_index = 0
        self.y_true = []
        self.y_pred = []

    def _balance_dataset(self, target_samples: int):
        """
        Balance the dataset by resampling majority and minority classes.

        Args:
            target_samples (int): Target number of samples per class
        """
        data_majority = self.data[self.data["ground_truth"] == 1]
        data_minority = self.data[self.data["ground_truth"] == 0]

        def resample_class(data, target):
            if len(data) > target:
                return resample(
                    data, replace=False, n_samples=target, random_state=self.random_seed
                )
            return resample(
                data, replace=True, n_samples=target, random_state=self.random_seed
            )

        target_per_class = target_samples // 2
        data_majority = resample_class(data_majority, target_per_class)
        data_minority = resample_class(data_minority, target_per_class)

        self.data = (
            pd.concat([data_majority, data_minority])
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )
        self.num_rows = len(self.data)

    def reset(self, seed=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        self.current_index = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        if self.current_index >= self.num_rows:  # 修正: 使用 >= 而不是 +1 >
            return np.zeros(3, dtype=np.float32)

        row = self.data.iloc[self.current_index]
        return np.array(
            [row["rf_prob"], row["xgb_prob"], row["svm_prob"]], dtype=np.float32
        )

    def _get_action_clf_threshold(self, action) -> Dict[str, float]:
        """
        將動作轉換為分類器閾值配置

        Args:
            action (int): 動作索引

        Returns:
            Dict[str, float]: 分類器及其閾值的字典映射
        """
        clf_thresholds = {}
        num_rf_thresholds = len(self.rf_thresholds)
        num_xgb_thresholds = len(self.xgb_thresholds)
        num_svm_thresholds = len(self.svm_thresholds)

        # 一個分類器的情況
        if self.num_clf == 1:
            if action < num_rf_thresholds:
                clf_thresholds["rf"] = self.rf_thresholds[action]
            elif action < num_rf_thresholds + num_xgb_thresholds:
                clf_thresholds["xgb"] = self.xgb_thresholds[action - num_rf_thresholds]
            else:
                clf_thresholds["svm"] = self.svm_thresholds[
                    action - num_rf_thresholds - num_xgb_thresholds
                ]

        # 兩個分類器的情況
        elif self.num_clf == 2:
            # RF + XGB 組合
            rf_xgb_combinations = num_rf_thresholds * num_xgb_thresholds
            # RF + SVM 組合
            rf_svm_combinations = num_rf_thresholds * num_svm_thresholds

            if action < rf_xgb_combinations:
                # 計算 RF 和 XGB 的閾值索引
                rf_idx = action // num_xgb_thresholds
                xgb_idx = action % num_xgb_thresholds
                clf_thresholds["rf"] = self.rf_thresholds[rf_idx]
                clf_thresholds["xgb"] = self.xgb_thresholds[xgb_idx]

            elif action < rf_xgb_combinations + rf_svm_combinations:
                # 計算 RF 和 SVM 的閾值索引
                remaining_action = action - rf_xgb_combinations
                rf_idx = remaining_action // num_svm_thresholds
                svm_idx = remaining_action % num_svm_thresholds
                clf_thresholds["rf"] = self.rf_thresholds[rf_idx]
                clf_thresholds["svm"] = self.svm_thresholds[svm_idx]

            else:
                # 計算 XGB 和 SVM 的閾值索引
                remaining_action = action - rf_xgb_combinations - rf_svm_combinations
                xgb_idx = remaining_action // num_svm_thresholds
                svm_idx = remaining_action % num_svm_thresholds
                clf_thresholds["xgb"] = self.xgb_thresholds[xgb_idx]
                clf_thresholds["svm"] = self.svm_thresholds[svm_idx]

        # 三個分類器的情況
        else:
            rf_idx = action // (num_xgb_thresholds * num_svm_thresholds)
            remaining = action % (num_xgb_thresholds * num_svm_thresholds)
            xgb_idx = remaining // num_svm_thresholds
            svm_idx = remaining % num_svm_thresholds

            clf_thresholds["rf"] = self.rf_thresholds[rf_idx]
            clf_thresholds["xgb"] = self.xgb_thresholds[xgb_idx]
            clf_thresholds["svm"] = self.svm_thresholds[svm_idx]

        return clf_thresholds

    def _get_prediction(self, row, clf_thresholds):
        """
        Calculate prediction based on classifier probabilities and thresholds.

        Args:
            row (pd.Series): Data row containing classifier probabilities
            clf_thresholds (dict): Classifier thresholds

        Returns:
            int: Prediction value
        """
        votes = []
        for clf, threshold in clf_thresholds.items():
            pred = int(row[f"{clf}_prob"] >= threshold)
            votes.append(pred)
        return 1 if sum(votes) >= len(votes) / 2 else 0

    def _get_reward(self, pred, ground_truth):
        """
        Calculate reward based on prediction and ground truth.

        Args:
            pred (int): Prediction value
            ground_truth (int): Ground truth value

        Returns:
            float: Reward value
        """
        return 1.0 if pred == ground_truth else 0

    def step(self, action):
        """
        Execute an action in the environment.

        Args:
            action (int): Action index that will be translated to classifier thresholds

        Returns:
            Tuple containing next state, reward, done flag, truncated flag, and info dict
        """
        # 修正：不再傳入多餘的參數
        clf_thresholds = self._get_action_clf_threshold(action)

        # 如果已經超出數據範圍，返回終止狀態
        if self.current_index >= self.num_rows:
            return (np.zeros(3, dtype=np.float32), 0.0, True, False, {})

        # 獲取當前數據行
        row = self.data.iloc[self.current_index]

        # 計算獎勵
        pred = self._get_prediction(row, clf_thresholds)
        ground_truth = row["ground_truth"]
        reward = self._get_reward(pred, ground_truth)
        info = {"prediction": pred, "ground_truth": ground_truth}

        # 移動到下一個數據點
        self.current_index += 1

        # 檢查是否結束
        done = True if self.current_index >= self.num_rows else False

        # 獲取下一個狀態
        next_state = self._get_observation()

        return (next_state, reward, done, False, info)

    def render(self, mode="human"):
        """Render the current environment state."""
        if self.current_index < self.num_rows:
            print(f"Step {self.current_index}: State = {self._get_observation()}")
        else:
            print("Environment finished")
