import yaml
import subprocess
import itertools
import time
import os

CONFIG_PATH = "configs/dqn_config.yaml"

# 所有組合
bound_sets = [
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

num_clf_options = [1, 2, 3]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    """使用 os.listdir() 取得 checkpoints 資料夾中最新的 .pth 檔案"""
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    if not pth_files:
        raise FileNotFoundError("No .pth checkpoint found!")

    # 加上完整路徑後排序
    full_paths = [os.path.join(checkpoint_dir, f) for f in pth_files]
    latest_file = max(full_paths, key=os.path.getmtime)

    return latest_file


def run_experiment(config):
    save_config(config, CONFIG_PATH)

    print(
        f"\n▶️ Running: num_clf={config['env']['num_clf']}, "
        f"rf={config['env']['rf_bounds']}, "
        f"xgb={config['env']['xgb_bounds']}, "
        f"svm={config['env']['svm_bounds']}"
    )

    # 執行 main.py
    subprocess.run(["python", "main.py", "--config", CONFIG_PATH])
    time.sleep(1)  # 等 checkpoint 寫入完成

    # 找最新的 checkpoint
    latest_checkpoint = get_latest_checkpoint()

    print(f"🧪 Testing latest model: {latest_checkpoint}")

    # 執行 test.py
    subprocess.run(["python", "test.py", "--model", latest_checkpoint])


if __name__ == "__main__":
    base_config = load_config(CONFIG_PATH)

    for bounds, num_clf in itertools.product(bound_sets, num_clf_options):
        config = base_config.copy()
        config["env"].update(bounds)
        config["env"]["num_clf"] = num_clf

        run_experiment(config)
