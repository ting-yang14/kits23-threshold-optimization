from envs.custom_env import ProbabilityThresholdEnv
import yaml
import os
import json


def ensure_log_dirs_exist():
    """Ensure required logs subdirectories exist."""
    subdirs = ["train", "test", "checkpoints", "progress"]
    for subdir in subdirs:
        os.makedirs(os.path.join("logs", subdir), exist_ok=True)


def save_results_to_json(results, save_path):
    """Save the results dictionary to a JSON file at the given path."""
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)


def load_config(config_path="configs/dqn_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def create_env(config, is_train=True):
    """創建訓練或測試環境"""
    return ProbabilityThresholdEnv(
        csv_path=(
            config["env"]["train_data"] if is_train else config["env"]["test_data"]
        ),
        rf_bounds=config["env"]["rf_bounds"],
        xgb_bounds=config["env"]["xgb_bounds"],
        svm_bounds=config["env"]["svm_bounds"],
        reward_scheme=config["env"]["reward_scheme"],
        step_size=config["env"]["step_size"],
        isTrain=is_train,
        num_clf=config["env"]["num_clf"],
        max_n_steps=config["env"]["max_n_steps"],
        random_seed=config["env"]["random_seed"],
    )


def evaluate_and_save(agent, env, model_path, log_dir="logs/test"):
    reward, accuracy, precision, recall, f1, ground_truth, predictions = evaluate_agent(
        env, agent
    )
    results = {
        "model_path": model_path,
        "reward": reward,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ground_truth": ground_truth,
        "predictions": predictions,
    }
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, f"test_results_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Test results saved to {results_path}")

    # Plot confusion matrix
    save_file = os.path.join(log_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(
        y_true=ground_truth,
        y_pred=predictions,
        title=f"Confusion Matrix - {model_name}",
        save_file=save_file,
    )
    return results
