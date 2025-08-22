import pandas as pd
import time
import subprocess
import os
import argparse
from utils.utils import load_config, save_config

CONFIG_PATH = "configs/dqn_config.yaml"


def create_experiment_name(params):
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
        print(f"✅ 訓練完成 (耗時: {time.time() - start_time:.2f}秒)")
        return True
    except Exception as e:
        print(f"❌ 實驗失敗: {str(e)}")
        return False


def save_experiment_log(experiment_params, success, log_file):
    with open(log_file, "a") as f:
        status = "SUCCESS" if success else "FAILED"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_name = create_experiment_name(experiment_params)
        f.write(f"{timestamp} - {status} - {experiment_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DQN grid search with custom repeat count."
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=5,
        help="Number of times to repeat each experiment (default: 5)",
    )
    parser.add_argument(
        "--csv",
        "-c",
        type=str,
        default="experiment_conditions.csv",
        help="Path to CSV file containing experiment conditions",
    )
    args = parser.parse_args()
    repeat_per_condition = args.repeat
    csv_path = args.csv

    condition_df = pd.read_csv(csv_path)
    base_config = load_config(CONFIG_PATH)

    total_experiments = len(condition_df) * repeat_per_condition
    current_experiment = 0
    successful_experiments = 0
    failed_experiments = 0
    overall_start_time = time.time()

    print(
        f"\n🚀 從 CSV 讀取條件，共 {total_experiments} 個實驗將被執行 (每列重複 {repeat_per_condition} 次)..."
    )
    response = input("\n是否繼續執行Repeated Experiments? (y/n): ")
    if response.lower() != "y":
        print("實驗已取消")
        exit()

    log_file = f"experiment_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, "w") as f:
        f.write(f"CSV-based Experiment Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    for idx, row in condition_df.iterrows():
        for repeat in range(repeat_per_condition):
            current_experiment += 1
            experiment_params = {
                "algorithm": row["algorithm"],
                "hidden_dim": int(row["hidden_dim"]),
                "max_steps": int(row["max_steps"]),
                "num_episodes": int(row["num_episodes"]),
                "num_clf": int(row["num_clf"]),
                "batch_size": int(row["batch_size"]),
                "bound_set_id": idx,
                "reward_id": repeat,
            }

            config = base_config.copy()
            config["agent"]["hidden_dim"] = int(row["hidden_dim"])
            config["agent"]["batch_size"] = int(row["batch_size"])
            config["train"]["num_episodes"] = int(row["num_episodes"])
            config["train"]["max_steps"] = int(row["max_steps"])
            config["env"]["num_clf"] = int(row["num_clf"])
            config["env"]["rf_bounds"] = [float(row["rf_low"]), float(row["rf_high"])]
            config["env"]["xgb_bounds"] = [
                float(row["xgb_low"]),
                float(row["xgb_high"]),
            ]
            config["env"]["svm_bounds"] = [
                float(row["svm_low"]),
                float(row["svm_high"]),
            ]
            config["env"]["reward_scheme"] = [
                float(row["TP_reward"]),
                float(row["TN_reward"]),
                float(row["FP_reward"]),
                float(row["FN_reward"]),
            ]
            config["algorithm"] = row["algorithm"]

            progress = (current_experiment / total_experiments) * 100
            elapsed_time = time.time() - overall_start_time
            avg_time = elapsed_time / current_experiment
            remaining_time = avg_time * (total_experiments - current_experiment)

            print(
                f"\n📈 進度: {current_experiment}/{total_experiments} ({progress:.1f}%)"
            )
            print(f"⏱️  已用時間: {elapsed_time/60:.1f}分鐘")
            print(f"⏳ 預估剩餘時間: {remaining_time/60:.1f}分鐘")

            success = run_experiment(config, experiment_params)
            save_experiment_log(experiment_params, success, log_file)

            if success:
                successful_experiments += 1
                print("✅ 實驗成功完成")
            else:
                failed_experiments += 1
                print("❌ 實驗失敗")

            time.sleep(1)

    total_time = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print("🎉 CSV Grid Search 完成!")
    print("=" * 80)
    print(f"總實驗數量: {total_experiments}")
    print(f"成功實驗: {successful_experiments}")
    print(f"失敗實驗: {failed_experiments}")
    print(f"成功率: {(successful_experiments/total_experiments)*100:.1f}%")
    print(f"總耗時: {total_time/60:.1f}分鐘")
    print(f"平均每個實驗: {total_time/total_experiments:.1f}秒")
    print(f"詳細日誌已保存至: {log_file}")
    print("=" * 80)
