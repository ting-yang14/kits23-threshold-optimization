import json
import os
import pandas as pd


def summarize_training_results(
    log_dir="logs/train", output_filename="training_summary.csv"
):
    """
    總結訓練結果，將所有符合條件的 JSON 檔案轉換為 CSV 格式。
    如果 CSV 檔案已存在，則只處理尚未包含的新檔案。

    :param log_dir: 存放訓練結果 JSON 檔案的資料夾路徑
    :param output_filename: 輸出 CSV 檔案的名稱
    :return: pandas.DataFrame - 包含所有訓練結果的 DataFrame
    """

    result_list = []
    existing_filenames = set()

    # 檢查 CSV 檔案是否已存在
    if os.path.exists(output_filename):
        try:
            existing_df = pd.read_csv(output_filename, encoding="utf-8")
            existing_filenames = set(existing_df["filename"].tolist())
            print(
                f"讀取現有 CSV 檔案：{output_filename}，已包含 {len(existing_filenames)} 筆記錄"
            )
        except Exception as e:
            print(f"讀取現有 CSV 檔案時發生錯誤：{e}")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()
        print(f"未找到現有 CSV 檔案，將建立新檔案：{output_filename}")

    # 檢查資料夾是否存在
    if not os.path.exists(log_dir):
        print(f"警告：資料夾 '{log_dir}' 不存在")
        return existing_df

    # 遍歷所有符合名稱的 JSON 檔
    new_files_count = 0
    for filename in os.listdir(log_dir):
        if filename.startswith("training_results_") and filename.endswith(".json"):
            if filename in existing_filenames:
                print(f"跳過已存在的檔案：{filename}")
                continue

            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # 提取需要的欄位
                record = {
                    "filename": filename,
                    "test_reward": data.get("test_reward"),
                    "test_accuracy": round(data.get("test_accuracy", 0), 3),
                    "test_precision": round(data.get("test_precision", 0), 3),
                    "test_recall": round(data.get("test_recall", 0), 3),
                    "test_f1": round(data.get("test_f1", 0), 3),
                    "num_episodes": data.get("num_episodes"),
                    "max_steps": data.get("max_steps"),
                    "rf_bounds": data["config"]["env"].get("rf_bounds"),
                    "xgb_bounds": data["config"]["env"].get("xgb_bounds"),
                    "svm_bounds": data["config"]["env"].get("svm_bounds"),
                    "num_clf": data["config"]["env"].get("num_clf"),
                }

                result_list.append(record)
                new_files_count += 1
                print(f"已處理新檔案：{filename}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"處理檔案 {filename} 時發生錯誤：{e}")
                continue

    # 如果沒有找到新的檔案
    if not result_list:
        print("未找到新的 JSON 檔案，共處理{new_files_count}個新檔案")
        return existing_df

    # 轉為 DataFrame
    new_df = pd.DataFrame(result_list)
    # 將 bounds 欄位展平（例如 rf_bounds[0], rf_bounds[1]）
    try:
        # 檢查 bounds 欄位是否存在且不為 None
        if not new_df["rf_bounds"].isna().all():
            new_df[["rf_low", "rf_high"]] = pd.DataFrame(
                new_df["rf_bounds"].tolist(), index=new_df.index
            )

        if not new_df["xgb_bounds"].isna().all():
            new_df[["xgb_low", "xgb_high"]] = pd.DataFrame(
                new_df["xgb_bounds"].tolist(), index=new_df.index
            )

        if not new_df["svm_bounds"].isna().all():
            new_df[["svm_low", "svm_high"]] = pd.DataFrame(
                new_df["svm_bounds"].tolist(), index=new_df.index
            )

        # 刪除原始 bounds 欄位
        new_df.drop(columns=["rf_bounds", "xgb_bounds", "svm_bounds"], inplace=True)

    except (ValueError, TypeError) as e:
        print(f"處理 bounds 欄位時發生錯誤：{e}")
        # 如果展平失敗，保留原始欄位
        pass

    # 合併現有資料和新資料
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(
            f"合併資料：原有 {len(existing_df)} 筆 + 新增 {len(new_df)} 筆 = 總共 {len(combined_df)} 筆"
        )
    else:
        combined_df = new_df
        print(f"建立新資料：總共 {len(combined_df)} 筆")

    # 儲存為 CSV
    try:
        combined_df = combined_df.sort_values("filename").reset_index(drop=True)
        combined_df.to_csv(output_filename, index=False, encoding="utf-8")
        print(f"CSV 儲存成功：{output_filename}")
    except Exception as e:
        print(f"儲存 CSV 時發生錯誤：{e}")

    return combined_df


if __name__ == "__main__":
    # 測試用途
    df = summarize_training_results()
    print(f"最終處理了 {len(df)} 筆記錄")
