import yaml
import subprocess
import itertools
import time
import os
from utils.utils import load_config, save_config

CONFIG_PATH = "configs/dqn_config.yaml"
# CONFIG_PATH = "configs/dqn_config2.yaml"

# 實驗設置
algorithms = ["dqn", "ddqn", "dueling_dqn", "dueling_ddqn"]
hidden_dims = [64]  # 128
max_steps = [5000]  # 3000
num_episodes = [500]  # 300
num_clf_options = [1, 2, 3]
batch_sizes = [64, 128]

bound_sets = [
    # {
    #     "rf_bounds": [0.5, 1.0],
    #     "xgb_bounds": [0.5, 1.0],
    #     "svm_bounds": [0.5, 1.0],
    # },
    {
        "rf_bounds": [0.85, 0.95],
        "xgb_bounds": [0.9, 1.0],
        "svm_bounds": [0.85, 0.95],
    },
    {
        "rf_bounds": [0.89, 0.93],
        "xgb_bounds": [0.96, 1.0],
        "svm_bounds": [0.88, 0.92],
    },
]
# [TP, TN, FP, FN]
reward_schemes = [[1, 1, 0, 0], [1, 1, -0.2, -0.8], [1, 2, 0, 0], [1, 2, -0.2, -0.8]]


def create_experiment_name(params):
    """創建實驗名稱用於識別"""
    return (
        f"{params['algorithm']}_"
        f"h{params['hidden_dim']}_"
        f"ms{params['max_steps']}_"
        f"ep{params['num_episodes']}_"
        f"clf{params['num_clf']}_"
        f"bs{params['batch_size']}_"
        f"bounds{params['bound_set_id']}_"
        f"reward{params['reward_id']}"
    )


def run_experiment(config, experiment_params):
    """運行單個實驗"""
    # 更新配置
    save_config(config, CONFIG_PATH)

    experiment_name = create_experiment_name(experiment_params)

    print("\n" + "=" * 80)
    print(f"🚀 實驗開始: {experiment_name}")
    print("=" * 80)
    print(f"Algorithm: {experiment_params['algorithm']}")
    print(f"Hidden Dim: {experiment_params['hidden_dim']}")
    print(f"Max Steps: {experiment_params['max_steps']}")
    print(f"Num Episodes: {experiment_params['num_episodes']}")
    print(f"Batch Size: {experiment_params['batch_size']}")
    print(f"Num Classifiers: {experiment_params['num_clf']}")
    print(f"RF Bounds: {config['env']['rf_bounds']}")
    print(f"XGB Bounds: {config['env']['xgb_bounds']}")
    print(f"SVM Bounds: {config['env']['svm_bounds']}")
    print(f"Reward Scheme: {config['env']['reward_scheme']}")
    print("-" * 80)

    try:
        # 執行訓練
        print("📊 開始訓練...")
        start_time = time.time()

        # 根據算法類型設置不同的參數或腳本
        train_cmd = [
            "python",
            "main.py",
            "--config",
            CONFIG_PATH,
            "--algorithm",
            experiment_params["algorithm"],
        ]

        result = subprocess.run(train_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ 訓練失敗: {result.stderr}")
            return False

        train_time = time.time() - start_time
        print(f"✅ 訓練完成 (耗時: {train_time:.2f}秒)")

        return True

    except Exception as e:
        print(f"❌ 實驗失敗: {str(e)}")
        return False


def calculate_total_experiments():
    """計算總實驗數量"""
    total = (
        len(algorithms)
        * len(hidden_dims)
        * len(max_steps)
        * len(num_episodes)
        * len(num_clf_options)
        * len(batch_sizes)
        * len(bound_sets)
        * len(reward_schemes)
    )
    return total


def print_experiment_info():
    """打印實驗信息"""
    total_experiments = calculate_total_experiments()

    print("🔬 DQN Grid Search 實驗配置")
    print("=" * 60)
    print(f"Algorithms: {algorithms}")
    print(f"Hidden Dimensions: {hidden_dims}")
    print(f"Max Steps: {max_steps}")
    print(f"Num Episodes: {num_episodes}")
    print(f"Num Classifiers: {num_clf_options}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Bound Sets: {len(bound_sets)} sets")
    print(f"Reward Schemes: {len(reward_schemes)} schemes")
    print("=" * 60)
    print(f"📊 總實驗數量: {total_experiments}")
    print(f"⏱️  預估時間: ~{total_experiments * 2:.0f} 分鐘 (假設每個實驗2分鐘)")
    print("=" * 60)


def save_experiment_log(experiment_params, success, log_file="experiment_log.txt"):
    """記錄實驗結果"""
    with open(log_file, "a") as f:
        status = "SUCCESS" if success else "FAILED"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_name = create_experiment_name(experiment_params)
        f.write(f"{timestamp} - {status} - {experiment_name}\n")


if __name__ == "__main__":
    # 打印實驗信息
    print_experiment_info()

    # 確認是否繼續
    response = input("\n是否繼續執行Grid Search? (y/n): ")
    if response.lower() != "y":
        print("實驗已取消")
        exit()

    # 載入基礎配置
    base_config = load_config(CONFIG_PATH)

    # 創建實驗日誌文件
    log_file = f"experiment_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, "w") as f:
        f.write(
            f"DQN Grid Search Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write("=" * 80 + "\n")

    # 計數器
    total_experiments = calculate_total_experiments()
    current_experiment = 0
    successful_experiments = 0
    failed_experiments = 0

    # 開始Grid Search
    print(f"\n🚀 開始執行 {total_experiments} 個實驗...")
    overall_start_time = time.time()

    # 生成所有參數組合
    for (
        algorithm,
        hidden_dim,
        max_step,
        num_episode,
        num_clf,
        batch_size,
        bound_set_idx,
        reward_scheme_idx,
    ) in itertools.product(
        algorithms,
        hidden_dims,
        max_steps,
        num_episodes,
        num_clf_options,
        batch_sizes,
        range(len(bound_sets)),
        range(len(reward_schemes)),
    ):
        current_experiment += 1

        # 準備實驗參數
        experiment_params = {
            "algorithm": algorithm,
            "hidden_dim": hidden_dim,
            "max_steps": max_step,
            "num_episodes": num_episode,
            "num_clf": num_clf,
            "batch_size": batch_size,
            "bound_set_id": bound_set_idx,
            "reward_id": reward_scheme_idx,
        }

        # 準備配置
        config = base_config.copy()

        # 更新 agent 設定
        config["agent"]["hidden_dim"] = hidden_dim
        config["agent"]["batch_size"] = batch_size

        # 更新 train 設定
        config["train"]["num_episodes"] = num_episode
        config["train"]["max_steps"] = max_step

        # 更新環境設定
        config["env"].update(bound_sets[bound_set_idx])
        config["env"]["num_clf"] = num_clf
        config["env"]["reward_scheme"] = reward_schemes[reward_scheme_idx]

        # 如果你的代碼需要算法類型，可以添加到配置中
        config["algorithm"] = algorithm

        # 進度顯示
        progress = (current_experiment / total_experiments) * 100
        elapsed_time = time.time() - overall_start_time
        avg_time_per_exp = (
            elapsed_time / current_experiment if current_experiment > 0 else 0
        )
        remaining_time = avg_time_per_exp * (total_experiments - current_experiment)

        print(f"\n📈 進度: {current_experiment}/{total_experiments} ({progress:.1f}%)")
        print(f"⏱️  已用時間: {elapsed_time/60:.1f}分鐘")
        print(f"⏳ 預估剩餘時間: {remaining_time/60:.1f}分鐘")

        # 執行實驗
        success = run_experiment(config, experiment_params)

        # 記錄結果
        save_experiment_log(experiment_params, success, log_file)

        if success:
            successful_experiments += 1
            print("✅ 實驗成功完成")
        else:
            failed_experiments += 1
            print("❌ 實驗失敗")

        # 簡短延遲，避免系統過載
        time.sleep(1)

    # 總結
    total_time = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print("🎉 Grid Search 完成!")
    print("=" * 80)
    print(f"總實驗數量: {total_experiments}")
    print(f"成功實驗: {successful_experiments}")
    print(f"失敗實驗: {failed_experiments}")
    print(f"成功率: {(successful_experiments/total_experiments)*100:.1f}%")
    print(f"總耗時: {total_time/60:.1f}分鐘")
    print(f"平均每個實驗: {total_time/total_experiments:.1f}秒")
    print(f"詳細日誌已保存至: {log_file}")
    print("=" * 80)
