import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from sklearn.utils import resample
from typing import Tuple, Dict, Optional
from stable_baselines3.common.vec_env import DummyVecEnv


class ProbabilityThresholdEnv(gym.Env):
    """A Gymnasium environment for optimizing classifier thresholds in a reinforcement learning setting."""

    metadata = {"render_modes": ["human"]}
    MAX_EPISODE_SEEDS = 1000
    MAX_SEED_VALUE = 2**32

    def __init__(
        self,
        csv_path: str,
        rf_bounds: Tuple[float, float] = (0.85, 0.90),
        xgb_bounds: Tuple[float, float] = (0.85, 0.90),
        svm_bounds: Tuple[float, float] = (0.85, 0.90),
        step_size: float = 0.01,
        isTrain: bool = True,
        num_clf: int = 3,
        max_n_steps: int = 10000,
        random_seed: int = 42,
    ):
        """
        Initialize the ProbabilityThresholdEnv environment.

        Args:
            csv_path (str): Path to CSV file containing classifier probabilities and ground truth.
            rf_bounds (Tuple[float, float]): Bounds for Random Forest threshold (min, max).
            xgb_bounds (Tuple[float, float]): Bounds for XGBoost threshold (min, max).
            svm_bounds (Tuple[float, float]): Bounds for SVM threshold (min, max).
            step_size (float): Step size for generating threshold values.
            isTrain (bool): Whether to use training mode (with data balancing).
            num_clf (int): Number of classifiers to use (1, 2, or 3).
            max_n_steps (int): Target number of samples for training data balancing.
            random_seed (int): Random seed for reproducibility.
        """
        super(ProbabilityThresholdEnv, self).__init__()
        self.random_seed = random_seed
        # 使用隨機數生成器
        self.rng = np.random.default_rng(random_seed)
        # 為每個episode生成隨機種子
        self.episode_seeds = self.rng.integers(
            0, self.MAX_SEED_VALUE, size=self.MAX_EPISODE_SEEDS
        )
        self.current_episode = 0
        self.isTrain = isTrain

        # Validate num_clf
        assert num_clf in [1, 2, 3], "num_clf must be 1, 2, or 3"
        self.num_clf = num_clf

        # Load and validate data from CSV
        self.data = pd.read_csv(csv_path)
        required_columns = ["rf_prob", "xgb_prob", "svm_prob", "ground_truth"]
        assert all(
            col in self.data.columns for col in required_columns
        ), f"CSV must contain columns: {', '.join(required_columns)}"
        assert (
            self.data["ground_truth"].isin([0, 1]).all()
        ), "ground_truth column must contain only 0 or 1"

        for col in ["rf_prob", "xgb_prob", "svm_prob"]:
            assert (
                (self.data[col] >= 0) & (self.data[col] <= 1)
            ).all(), f"All values in {col} must be between 0 and 1"

        # Set observation space: 3 values between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Calculate and validate threshold values
        assert (
            0 <= rf_bounds[0] < rf_bounds[1] <= 1
        ), f"Invalid rf_bounds: {rf_bounds}. Must satisfy 0 <= rf_bounds[0] < rf_bounds[1] <= 1"
        assert (
            0 <= xgb_bounds[0] < xgb_bounds[1] <= 1
        ), f"Invalid xgb_bounds: {xgb_bounds}. Must satisfy 0 <= xgb_bounds[0] < xgb_bounds[1] <= 1"
        assert (
            0 <= svm_bounds[0] < svm_bounds[1] <= 1
        ), f"Invalid svm_bounds: {svm_bounds}. Must satisfy 0 <= svm_bounds[0] < svm_bounds[1] <= 1"

        self.rf_thresholds = np.unique(np.arange(rf_bounds[0], rf_bounds[1], step_size))
        self.xgb_thresholds = np.unique(
            np.arange(xgb_bounds[0], xgb_bounds[1], step_size)
        )
        self.svm_thresholds = np.unique(
            np.arange(svm_bounds[0], svm_bounds[1], step_size)
        )

        # Validate thresholds
        for name, thresholds in [
            ("rf", self.rf_thresholds),
            ("xgb", self.xgb_thresholds),
            ("svm", self.svm_thresholds),
        ]:
            if len(thresholds) == 0:
                raise ValueError(
                    f"No valid thresholds for {name} with bounds {locals()[f'{name}_bounds']} and step_size {step_size}"
                )
        # Calculate action space size
        num_rf = len(self.rf_thresholds)
        num_xgb = len(self.xgb_thresholds)
        num_svm = len(self.svm_thresholds)

        if self.num_clf == 1:
            total_actions = num_rf + num_xgb + num_svm
        elif self.num_clf == 2:
            total_actions = num_rf * num_xgb + num_rf * num_svm + num_xgb * num_svm
        else:
            total_actions = num_rf * num_xgb * num_svm

        self.action_space = spaces.Discrete(total_actions)

        # Shuffle and balance data
        if isTrain:
            self._balance_dataset(max_n_steps)
            print(f"Balanced training dataset with {len(self.data)} samples")
        else:
            self.num_rows = len(self.data)
            print(f"Loaded testing dataset with {len(self.data)} samples")

        self.current_index = 0
        self.y_true = []
        self.y_pred = []
        self.shuffled_indices = list(range(self.num_rows))

    def _balance_dataset(self, target_samples: int):
        """
        Balance the dataset by resampling majority and minority classes.

        Args:
            target_samples (int): Target number of samples per class
        """
        data_majority = self.data[self.data["ground_truth"] == 1]
        data_minority = self.data[self.data["ground_truth"] == 0]

        if len(data_majority) == 0 or len(data_minority) == 0:
            raise ValueError("Dataset must contain samples for both classes (0 and 1)")

        target_per_class = max(
            target_samples // 2, max(len(data_majority), len(data_minority))
        )

        def resample_class(data, target):
            if len(data) == 0:
                raise ValueError("Cannot resample an empty class")
            if len(data) > target:
                return resample(
                    data, replace=False, n_samples=target, random_state=self.random_seed
                )
            return resample(
                data, replace=True, n_samples=target, random_state=self.random_seed
            )

        data_majority = resample_class(data_majority, target_per_class)
        data_minority = resample_class(data_minority, target_per_class)

        self.data = (
            pd.concat([data_majority, data_minority])
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )
        self.num_rows = len(self.data)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed (Optional[int]): Seed for reproducibility. Overrides default seed if provided.

        Returns:
            Tuple[np.ndarray, Dict]: Initial observation and info dictionary.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.episode_seeds = self.rng.integers(
                0, self.MAX_SEED_VALUE, size=self.MAX_EPISODE_SEEDS
            )
            self.current_episode = 0

        super().reset(seed=seed)
        if self.isTrain:
            episode_seed = self.episode_seeds[
                self.current_episode % len(self.episode_seeds)
            ]
            self.episode_rng = np.random.default_rng(episode_seed)
            self.current_episode += 1
            self.shuffled_indices = self.episode_rng.permutation(self.num_rows).tolist()
        else:
            self.episode_rng = self.rng
            self.shuffled_indices = list(range(self.num_rows))

        self.current_index = 0
        self.y_true = []
        self.y_pred = []
        self.episode_step_count = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Get current environment observation.

        Returns:
            np.ndarray: Array of classifier probabilities [rf_prob, xgb_prob, svm_prob].
        """
        if self.current_index >= self.num_rows:
            return np.zeros(3, dtype=np.float32)

        row = self.data.iloc[self.shuffled_indices[self.current_index]]
        return np.array(
            [row["rf_prob"], row["xgb_prob"], row["svm_prob"]], dtype=np.float32
        )

    def _get_action_clf_threshold(self, action) -> Dict[str, float]:
        """
        Convert action index to classifier threshold configuration.

        Args:
            action (int): Action index

        Returns:
            Dict[str, float]: Dictionary mapping classifiers to their thresholds

        Action indexing scheme:
        - num_clf=1: [rf thresholds, xgb thresholds, svm thresholds]
        - num_clf=2: [rf+xgb pairs, rf+svm pairs, xgb+svm pairs]
        - num_clf=3: [rf * xgb * svm combinations]
        """
        clf_thresholds = {}
        num_rf = len(self.rf_thresholds)
        num_xgb = len(self.xgb_thresholds)
        num_svm = len(self.svm_thresholds)

        if self.num_clf == 1:
            if action < num_rf:
                clf_thresholds["rf"] = self.rf_thresholds[action]
            elif action < num_rf + num_xgb:
                clf_thresholds["xgb"] = self.xgb_thresholds[action - num_rf]
            else:
                clf_thresholds["svm"] = self.svm_thresholds[action - num_rf - num_xgb]

        elif self.num_clf == 2:
            rf_xgb_combinations = num_rf * num_xgb
            rf_svm_combinations = num_rf * num_svm

            if action < rf_xgb_combinations:
                rf_idx = action // num_xgb
                xgb_idx = action % num_xgb
                clf_thresholds["rf"] = self.rf_thresholds[rf_idx]
                clf_thresholds["xgb"] = self.xgb_thresholds[xgb_idx]
            elif action < rf_xgb_combinations + rf_svm_combinations:
                action = action - rf_xgb_combinations
                rf_idx = action // num_svm
                svm_idx = action % num_svm
                clf_thresholds["rf"] = self.rf_thresholds[rf_idx]
                clf_thresholds["svm"] = self.svm_thresholds[svm_idx]
            else:
                action = action - rf_xgb_combinations - rf_svm_combinations
                xgb_idx = action // num_svm
                svm_idx = action % num_svm
                clf_thresholds["xgb"] = self.xgb_thresholds[xgb_idx]
                clf_thresholds["svm"] = self.svm_thresholds[svm_idx]

        else:  # num_clf == 3
            rf_idx = action // (num_xgb * num_svm)
            remaining = action % (num_xgb * num_svm)
            xgb_idx = remaining // num_svm
            svm_idx = remaining % num_svm
            clf_thresholds["rf"] = self.rf_thresholds[rf_idx]
            clf_thresholds["xgb"] = self.xgb_thresholds[xgb_idx]
            clf_thresholds["svm"] = self.svm_thresholds[svm_idx]

        return clf_thresholds

    def _get_prediction(self, row: pd.Series, clf_thresholds: Dict[str, float]) -> int:
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

    def _get_reward(self, pred: int, ground_truth: int) -> float:
        """
        Calculate reward based on prediction and ground truth using reward scheme.

        Args:
            pred (int): Prediction value
            ground_truth (int): Ground truth value

        Returns:
            float: Reward value
        """
        if pred == 1 and ground_truth == 1:
            return 1.0
        else:
            # 不懲罰誤判
            return 0.0
            # 惡性被判為良性扣較多分
            # return -0.5 if pred == 0 and ground_truth == 1 else -0.2
            # 良性被判為惡性扣較多分
            # return -0.5 if pred == 1 and ground_truth == 0 else -0.2

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute an action in the environment.

        Args:
            action (int): Action index that will be translated to classifier thresholds

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: Next state, reward, done flag, truncated flag, and info dict
        """
        # Check for termination
        if (self.isTrain and self.current_index >= self.num_rows) or (
            not self.isTrain and self.current_index >= self.num_rows
        ):
            return np.zeros(3, dtype=np.float32), 0.0, True, False, {}

        clf_thresholds = self._get_action_clf_threshold(action)
        row = self.data.iloc[self.shuffled_indices[self.current_index]]

        # Calculate reward and prediction
        pred = self._get_prediction(row, clf_thresholds)
        ground_truth = row["ground_truth"]
        self.y_true.append(ground_truth)
        self.y_pred.append(pred)
        reward = self._get_reward(pred, ground_truth)

        # Move to next data point
        self.current_index += 1
        done = self.current_index >= self.num_rows

        # Get next state
        next_state = self._get_observation()

        info = {"prediction": pred, "ground_truth": ground_truth}

        return next_state, reward, done, False, info

    def render(self, mode: str = "human"):
        """
        Render the current environment state.

        Args:
            mode (str): Rendering mode, only 'human' is supported.
        """
        if mode != "human":
            raise NotImplementedError(f"Render mode {mode} not supported")

        if self.current_index < self.num_rows:
            state = self._get_observation()
            row = self.data.iloc[self.shuffled_indices[self.current_index]]
            print(f"Step {self.current_index}:")
            print(
                f"  State: rf_prob={state[0]:.3f}, xgb_prob={state[1]:.3f}, svm_prob={state[2]:.3f}"
            )
            print(f"  Ground Truth: {row['ground_truth']}")
            print(
                f"  Episode Accuracy: {np.mean([p == t for p, t in zip(self.y_pred, self.y_true)]):.3f}"
            )
        else:
            print("Environment finished")
