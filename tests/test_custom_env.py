import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.custom_env import ProbabilityThresholdEnv


@pytest.fixture(scope="module")
def test_env(tmp_path_factory):
    # 建立簡單測試資料
    data = pd.DataFrame(
        {
            "rf_prob": [0.9, 0.7, 0.95],
            "xgb_prob": [0.85, 0.6, 0.92],
            "svm_prob": [0.88, 0.4, 0.96],
            "ground_truth": [1, 0, 1],
        }
    )

    test_csv = tmp_path_factory.mktemp("data") / "test.csv"
    data.to_csv(test_csv, index=False)

    env = ProbabilityThresholdEnv(csv_path=str(test_csv), isTrain=False)
    return env


def test_env_initialization(test_env):
    assert test_env.observation_space.shape == (3,)


def test_env_reset_returns_valid_observation(test_env):
    obs, _ = test_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3,)
    assert np.all(obs >= 0) and np.all(obs <= 1)


def test_step_returns_valid_outputs_with_metrics(test_env):
    test_env.reset()
    action = test_env.action_space.sample()
    obs, reward, done, truncated, info = test_env.step(action)

    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert "accuracy" in info
    assert "precision" in info
    assert "recall" in info
    assert "f1" in info


def test_prediction_output(test_env):
    row = test_env.data.iloc[0]
    thresholds = {"rf": 0.85, "xgb": 0.85, "svm": 0.85}
    pred = test_env._get_prediction(row, thresholds)
    assert pred in [0, 1]
