import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

app = Flask(__name__)


def load_data():
    with open("filtered_releases.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_model(model_dir="tree_model"):
    """載入已儲存的模型和相關資訊"""
    try:
        # 檢查模型目錄是否存在
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目錄 '{model_dir}' 不存在")

        # 載入模型
        model_path = os.path.join(model_dir, "random_forest_model.joblib")
        print(f"載入模型: {model_path}")
        model = joblib.load(model_path)

        # 載入編碼器
        encoders_path = os.path.join(model_dir, "encoders.joblib")
        print(f"載入編碼器: {encoders_path}")
        encoders = joblib.load(encoders_path)

        # 載入縮放器
        scalers_path = os.path.join(model_dir, "scalers.joblib")
        print(f"載入縮放器: {scalers_path}")
        scalers = joblib.load(scalers_path)

        # 載入模型信息
        model_info_path = os.path.join(model_dir, "model_info.joblib")
        print(f"載入模型信息: {model_info_path}")
        model_info = joblib.load(model_info_path)

        # 整合所有信息
        model_dict = {
            "model": model,
            "encoders": encoders,
            "scalers": scalers,
            "feature_cols": model_info["feature_cols"],
            "target_col": model_info["target_col"],
            "target_transform": model_info["target_transform"],
        }

        return model_dict
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        return None


def preprocess_prediction_data(df, model_dict):
    """預處理預測數據，與訓練數據保持一致"""
    # 複製數據以避免修改原始數據
    df_copy = df.copy()

    # 確保所有必要的欄位都存在
    required_columns = [
        "日期",
        "上市日",
        "資料抓取日",
        "廠商",
        "型號",
        "容量",
        "電池健康度",
        "上市日建議售價",
        "抓取日建議售價",
    ]

    missing_columns = [col for col in required_columns if col not in df_copy.columns]
    if missing_columns:
        raise ValueError(f"預測數據缺少必要欄位: {missing_columns}")

    # 1. 確保日期欄位是 datetime 類型
    date_cols = ["日期", "上市日", "資料抓取日"]
    for col in date_cols:
        if col in df_copy.columns and not pd.api.types.is_datetime64_dtype(
            df_copy[col]
        ):
            df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")

    # 2. 從日期提取時間特徵
    for col in ["日期", "上市日"]:
        if col in df_copy.columns:
            df_copy[f"{col}_year"] = df_copy[col].dt.year
            df_copy[f"{col}_quarter"] = df_copy[col].dt.quarter

    # 3. 計算市場年齡（月）
    if "上市日" in df_copy.columns and "日期" in df_copy.columns:
        df_copy["市場年齡_月"] = (
            df_copy["日期"].dt.year - df_copy["上市日"].dt.year
        ) * 12 + (df_copy["日期"].dt.month - df_copy["上市日"].dt.month)

    # 4. 計算價格相關特徵
    if "抓取日建議售價" in df_copy.columns and "上市日建議售價" in df_copy.columns:
        # 官方價格貶值率
        df_copy["官方價格貶值率"] = (
            df_copy["抓取日建議售價"] / df_copy["上市日建議售價"]
        ) * 100

        # 計算建議售價的變化率 (每月)
        if "市場年齡_月" in df_copy.columns:
            # 避免除以零
            safe_months = df_copy["市場年齡_月"].copy()
            safe_months = safe_months.replace(0, 0.1)  # 將0替換為0.1，避免除以零

            df_copy["每月建議售價變化率"] = (
                (
                    (df_copy["抓取日建議售價"] - df_copy["上市日建議售價"])
                    / df_copy["上市日建議售價"]
                )
                / safe_months
                * 100
            )

            # 限制極端值
            df_copy["每月建議售價變化率"] = df_copy["每月建議售價變化率"].clip(-50, 50)

    # 5. 將容量轉換為分類特徵
    if "容量" in df_copy.columns:
        # 確保容量是數值型
        df_copy["容量"] = pd.to_numeric(df_copy["容量"], errors="coerce")

        # 使用與訓練時相同的分類參數
        bins = [0, 128, 256, float("inf")]
        labels = ["小容量", "中容量", "大容量"]

        # 創建容量類別
        df_copy["容量類別"] = pd.cut(
            df_copy["容量"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        ).astype(
            str
        )  # 確保轉換為字符串類型

    # 6. 編碼類別特徵
    encoders = model_dict.get("encoders", {})
    for col, encoder in encoders.items():
        if col in df_copy.columns:
            # 確保是字符串類型
            df_copy[col] = df_copy[col].astype(str)

            try:
                # 根據編碼器類型進行處理
                if isinstance(encoder, OneHotEncoder):
                    # OneHotEncoder
                    encoded = encoder.transform(df_copy[[col]])

                    # 如果是稀疏矩陣，轉換為密集矩陣
                    if hasattr(encoded, "toarray"):
                        encoded = encoded.toarray()

                    # 使用保存的特徵名稱
                    if hasattr(encoder, "feature_names"):
                        feature_names = encoder.feature_names
                    else:
                        # 如果沒有保存特徵名稱，嘗試生成
                        feature_names = [
                            f"{col}_{name}" for name in encoder.categories_[0]
                        ]

                    # 添加編碼後的特徵
                    for i, name in enumerate(feature_names):
                        df_copy[name] = encoded[:, i]

                elif isinstance(encoder, LabelEncoder):
                    # LabelEncoder
                    try:
                        df_copy[f"{col}_encoded"] = encoder.transform(df_copy[col])
                    except ValueError:
                        # 處理未知類別
                        print(f"警告: 處理 {col} 時遇到未知類別")
                        # 為未知類別分配一個特殊值
                        unknown_value = len(encoder.classes_)
                        df_copy[f"{col}_encoded"] = -1  # 初始化
                        known_mask = df_copy[col].isin(encoder.classes_)
                        df_copy.loc[known_mask, f"{col}_encoded"] = encoder.transform(
                            df_copy.loc[known_mask, col]
                        )
                        df_copy.loc[~known_mask, f"{col}_encoded"] = unknown_value
            except Exception as e:
                print(f"編碼 '{col}' 時出錯: {e}")
                # 對於新類別值，使用全零編碼或特殊值
                if isinstance(encoder, OneHotEncoder):
                    # 對於 OneHotEncoder，使用全零編碼
                    existing_encoded_cols = [
                        c for c in model_dict["feature_cols"] if c.startswith(f"{col}_")
                    ]
                    for encoded_col in existing_encoded_cols:
                        df_copy[encoded_col] = 0
                else:
                    # 對於 LabelEncoder，使用特殊值（類別數量）
                    df_copy[f"{col}_encoded"] = (
                        len(encoder.classes_) if hasattr(encoder, "classes_") else -1
                    )

    # 7. 縮放數值特徵
    scalers = model_dict.get("scalers", {})
    for col, scaler in scalers.items():
        if col in df_copy.columns:
            # 處理可能的缺失值
            df_copy[col] = df_copy[col].fillna(
                df_copy[col].median() if not df_copy[col].empty else 0
            )
            try:
                df_copy[f"{col}_scaled"] = scaler.transform(df_copy[[col]])
            except Exception as e:
                print(f"縮放 '{col}' 時出錯: {e}")
                # 使用平均值
                df_copy[f"{col}_scaled"] = 0

    # 8. 確保所有特徵都存在
    for feature in model_dict["feature_cols"]:
        if feature not in df_copy.columns:
            print(f"警告: 缺少特徵 '{feature}'，使用零值替代")
            df_copy[feature] = 0

    return df_copy


def predict_prices(df, model_dict):
    """使用載入的模型進行價格預測"""
    try:
        # 預處理數據
        df_processed = preprocess_prediction_data(df, model_dict)

        # 提取模型所需的特徵
        X_pred = df_processed[model_dict["feature_cols"]]

        # 進行預測
        model = model_dict["model"]
        y_pred = model.predict(X_pred)

        # 如果目標進行了對數轉換，需要反轉
        if model_dict.get("target_transform") == "log1p":
            y_pred = np.expm1(y_pred)

        return y_pred
    except Exception as e:
        print(f"預測過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        return None


# 全局變量，只載入一次模型
MODEL_DICT = None


@app.route("/", methods=["GET", "POST"])
def index():
    data = load_data()

    # 廠商排序 (A-Z)
    vendors = list(set([item["廠商"] for item in data]))
    vendors.sort()

    # 型號排序 (依上市日期，新到舊)
    model_date_map = {item["型號"]: item["上市日"] for item in data}
    models = list(set([item["型號"] for item in data]))
    models.sort(key=lambda model: model_date_map[model], reverse=True)

    launch_date = data[0]["上市日"] if data else ""

    # --- 初始容量列表 (根據第一個產品) ---
    if data:
        initial_capacities = sorted(data[0]["容量"])
    else:
        initial_capacities = []

    if request.method == "POST":
        selected_vendor = request.form["vendor"]
        selected_model = request.form["model"]
        selected_capacity = request.form["capacity"]  # 從下拉選單選取的
        date = request.form["date"]
        battery_health = request.form["battery_health"]

        selected_product = None
        for item in data:
            if item["廠商"] == selected_vendor and item["型號"] == selected_model:
                selected_product = item
                break

        if selected_product:
            launch_date = selected_product["上市日"]
            data_capture_date = selected_product["資料抓取日"]

            # 修改這部分 - 處理容量可能是小數點的情況
            selected_capacity_float = float(selected_capacity)

            # 找到最接近的容量值
            closest_capacity_index = 0
            min_diff = float("inf")
            for i, cap in enumerate(selected_product["容量"]):
                diff = abs(float(cap) - selected_capacity_float)
                if diff < min_diff:
                    min_diff = diff
                    closest_capacity_index = i

            launch_price = selected_product["上市日建議售價"][closest_capacity_index]
            capture_price = selected_product["抓取日建議售價"][closest_capacity_index]

            if capture_price == "N/A":
                capture_price = None
            else:
                capture_price = int(capture_price)

            # 創建預測數據框
            prediction_data = {
                "廠商": [selected_vendor],
                "型號": [selected_model],
                "上市日": [launch_date],
                "容量": [selected_capacity_float],  # 使用浮點數
                "日期": [date],
                "電池健康度": [int(battery_health)],
                "上市日建議售價": [launch_price],
                "抓取日建議售價": [capture_price],
                "資料抓取日": [data_capture_date],
            }

            df_pred = pd.DataFrame(prediction_data)

            # 載入模型（如果尚未載入）
            global MODEL_DICT
            if MODEL_DICT is None:
                MODEL_DICT = load_model()
                if MODEL_DICT is None:
                    return jsonify({"error": "無法載入模型"}), 500

            # 進行預測
            prediction = predict_prices(df_pred, MODEL_DICT)

            if prediction is not None and len(prediction) > 0:
                predicted_price = int(round(prediction[0]))

                # 將預測結果添加到回傳數據
                backend_data = {
                    "廠商": selected_vendor,
                    "型號": selected_model,
                    "上市日": launch_date,
                    "容量": selected_capacity,
                    "日期": date,
                    "電池健康度": int(battery_health),
                    "上市日建議售價": launch_price,
                    "抓取日建議售價": capture_price,
                    "資料抓取日": data_capture_date,
                    "預測金額": predicted_price,
                }
                return jsonify(backend_data)
            else:
                return jsonify({"error": "預測失敗"}), 500
        else:
            return jsonify({"error": "Product not found."}), 400

    return render_template(
        "index.html",
        vendors=vendors,
        models=models,
        launch_date=launch_date,
        capacities=initial_capacities,
    )  # 傳入初始容量


@app.route("/get_models")
def get_models():
    vendor = request.args.get("vendor")
    data = load_data()
    vendor_models = [
        (item["型號"], item["上市日"]) for item in data if item["廠商"] == vendor
    ]
    vendor_models.sort(key=lambda x: x[1], reverse=True)
    sorted_models = [model for model, _ in vendor_models]
    return jsonify(sorted_models)


# 新增的路由：根據廠商和型號獲取容量
@app.route("/get_capacities")
def get_capacities():
    vendor = request.args.get("vendor")
    model = request.args.get("model")
    data = load_data()

    for item in data:
        if item["廠商"] == vendor and item["型號"] == model:
            # 排序容量 (從小到大)
            capacities = sorted(item["容量"])
            return jsonify(capacities)

    return jsonify([])  # 如果找不到，返回空列表


if __name__ == "__main__":
    # 預先載入模型
    MODEL_DICT = load_model()
    app.run(debug=True)
