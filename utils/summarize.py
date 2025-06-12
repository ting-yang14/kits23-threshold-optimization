import json
import os
import pandas as pd


def summarize_training_results(
    log_dir="logs/train", output_file="training_summary.csv"
):
    """
    總結訓練結果，將 JSON 檔案轉換為 CSV 格式。
    如果 CSV 檔案已存在，則只處理尚未包含的新檔案。

    :param log_dir: 存放訓練結果 JSON 檔案的資料夾路徑
    :param output_file: 輸出 CSV 檔案的名稱
    :return: pandas.DataFrame - 包含所有訓練結果的 DataFrame
    """
    existing_df = pd.DataFrame()
    existing_files = set()

    # 檢查 CSV 檔案是否已存在
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file, encoding="utf-8")
            existing_files = set(existing_df["filename"])
            print(f"已讀取{len(existing_files)} 筆現有記錄")
        except Exception as e:
            print(f"讀取現有 CSV 檔案時發生錯誤：{e}")

    # 檢查資料夾是否存在
    if not os.path.exists(log_dir):
        print(f"警告：資料夾 '{log_dir}' 不存在")
        return existing_df

    # 處理新的 JSON 檔
    new_records = []
    for filename in os.listdir(log_dir):
        if not filename.endswith(".json") or filename in existing_files:
            continue

        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            config = data.get("config", {})
            env_config = config.get("env", {})
            agent_config = config.get("agent", {})
            train_config = config.get("train", {})

            record = {
                "filename": filename,
                "test_reward": data.get("test_reward"),
                "test_accuracy": data.get("test_accuracy"),
                "test_precision": data.get("test_precision"),
                "test_recall": data.get("test_recall"),
                "test_f1": data.get("test_f1"),
                "num_episodes": train_config.get("num_episodes"),
                "max_steps": train_config.get("max_steps"),
                "num_clf": env_config.get("num_clf"),
                "batch_size": agent_config.get("batch_size"),
                "hidden_dim": agent_config.get("hidden_dim"),
                "algorithm": config.get("algorithm", "dqn"),
            }

            for bound_type in ["rf_bounds", "xgb_bounds", "svm_bounds"]:
                bounds = env_config.get(bound_type)
                if bounds and len(bounds) == 2:
                    prefix = bound_type.replace("_bounds", "")
                    record[f"{prefix}_low"] = bounds[0]
                    record[f"{prefix}_high"] = bounds[1]

            reward_scheme = env_config.get("reward_scheme")
            if reward_scheme:
                record.update(
                    {
                        k + "_reward": v
                        for k, v in zip(["TP", "TN", "FP", "FN"], reward_scheme)
                    }
                )

            test_confusion_matrix = data.get("test_confusion_matrix")
            if test_confusion_matrix:
                for k in ["TP", "TN", "FP", "FN"]:
                    record[f"{k}_count"] = test_confusion_matrix.get(k, 0)
            new_records.append(record)
            print(f"已處理新檔案：{filename}")

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤：{e}")

    # 如果沒有找到新的檔案
    if not new_records:
        print(f"無新檔案需要處理")
        return existing_df

    # 轉為 DataFrame
    new_df = pd.DataFrame(new_records)
    combined_df = (
        pd.concat([existing_df, new_df], ignore_index=True)
        if not existing_df.empty
        else new_df
    )

    # 儲存為 CSV
    try:
        combined_df.sort_values("filename").to_csv(
            output_file, index=False, encoding="utf-8"
        )
        print(f"已儲存{len(combined_df)}筆記錄至{output_file}")
    except Exception as e:
        print(f"儲存錯誤：{e}")

    return combined_df


if __name__ == "__main__":
    # 測試用途
    df = summarize_training_results()
    print(f"最終處理了 {len(df)} 筆記錄")
