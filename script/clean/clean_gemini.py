import json
import os
import re
import shutil
import time
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# 你的 .env 有幾個API這裡就要有幾個
# how much APIs in your .env, here need how much rows
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
]
current_key_index = 0
model = None

task_count = 1
dataLoss_count = 0


def initialize_model():
    global model, current_key_index
    if current_key_index >= len(API_KEYS):
        current_key_index = 0  # 重置為第一個 API Key
        print(
            f"所有 API Key 已達到配額限制，開始休眠 30 分鐘 (從 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        )
        time.sleep(1800)  # 休眠 30 分鐘
        print(f"休眠結束，繼續執行 (在 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    print(f"切換至 API Key {current_key_index + 1}")
    genai.configure(api_key=API_KEYS[current_key_index])
    model = genai.GenerativeModel("gemini-1.5-flash-8b")


def ask_gemini(prompt):
    global task_count, dataLoss_count, current_key_index, model

    if model is None:
        initialize_model()

    max_retries = 3  # 每個請求最多重試3次
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = model.generate_content(prompt)
            time.sleep(8)  # 限制請求頻率，避免 429 錯誤
            return response.text
        except Exception as e:
            if "429" in str(e):  # 如果是 429 錯誤
                print(f"API Key {current_key_index + 1} 達到配額限制")
                current_key_index += 1
                try:
                    initialize_model()
                    retry_count += 1
                    continue  # 重試當前請求
                except Exception as e2:
                    print(f"初始化新 API 時發生錯誤: {e2}")
                    retry_count += 1
                    continue
            else:
                with open(f"requestError.txt", "a", encoding="utf-8") as file:
                    file.write(f"row {task_count} ~ row {task_count + 4}\n")
                dataLoss_count += 1
                print(f"AI 請求錯誤: {e}")
                return None
        finally:
            if retry_count >= max_retries:
                print(f"請求在重試 {max_retries} 次後仍然失敗")
            task_count += 5


def read_json(file_path):
    """讀取 JSON 檔案"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"JSON 檔案讀取錯誤: {e}")
        return None
    except Exception as e:
        print(f"檔案讀取錯誤: {e}")
        return None


def split_json_list(json_list, chunk_size):
    """將 JSON 陣列分成小區塊"""
    for i in range(0, len(json_list), chunk_size):
        yield json_list[i : i + chunk_size]


fail_count = 0
success_count = 0


def filter_json_data(json_data):
    """
    篩選 JSON 資料，只保留 title 欄位符合特定條件的資料。
    不符合條件的資料將儲存到 skipped_data.json 檔案中。
    """
    qualified_data = []
    skipped_data = []
    keywords = ["徵", "買", "賣", "售"]

    for item in json_data:
        if isinstance(item, dict) and "title" in item:
            title = item["title"]
            if any(keyword in title for keyword in keywords):
                qualified_data.append(item)
            else:
                skipped_data.append(item)
        else:
            skipped_data.append(item)

    if skipped_data:
        save_json(skipped_data, "skipped_data.json")
        print(
            f"有 {len(skipped_data)} 筆資料因 title 欄位不符合條件，已儲存至 skipped_data.json"
        )

    return qualified_data


def process_json_in_batches(
    json_data, chunk_size=10, source_filename=None
):  # 新增 source_filename 參數
    global fail_count, success_count, task_count
    """分批傳送 JSON 並讓 Gemini 擷取摘要"""
    summarized_results = []

    for batch in split_json_list(json_data, chunk_size):
        json_text = json.dumps(batch, ensure_ascii=False, indent=2)
        prompt = f"""
        {json_text}\n\n
        
        上方的json檔為各家電商的手機商品資訊，請根據資料清整出重要資訊後，將結果以json檔的格式給我即可\n\n
        重要資料請依照以下順序廠商、型號、金額、容量、電池健康度、顏色、日期、地區、福利品、配件、女用機、是否全新保固時長等\n
        備註 : 1、廠商請填寫如Apple/Sony/htc等\n
        2、金額為純數值不需要$或是幣別表示，若金額以 - 或 ~ 或其他符號隔開兩個或數個數字，則以這些數字的平均值為該項的金額\n
        3、容量通常為16的倍數，最小為32g\n
        4、電池健康度請填寫如90/80/70等，不需要%，若遇到非數值(如:良好)則一律視為null\n
        5、請盡量找出這些重要資料，若找不到符合的則以null代替\n
        6、在所有的 }} 及 ] 前都不要有 , \n
        7、顏色必為字串,\n
        8、日期直接從"time"抓取
        """

        response = ask_gemini(prompt)
        if response:
            try:
                summarized_data = json.loads(response)
                summarized_results.extend(summarized_data)
                success_count += 1
            except json.JSONDecodeError:
                cleaned_content = clean_json(response)
                try:
                    summarized_data = json.loads(cleaned_content)
                    summarized_results.extend(summarized_data)
                    success_count += 1
                except json.JSONDecodeError as e:
                    print(f"第 {task_count-5} 到 {task_count-1} 筆資料處理失敗: {e}")
                    fail_count += 1

                    # 使用來源檔案名稱建立失敗回應檔案
                    if source_filename:
                        fail_file = f"./cleaned_data/failRespon_{source_filename}"
                        # 將失敗的回應寫入檔案，並加入時間戳記
                        with open(fail_file, "a", encoding="utf-8") as file:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            file.write(
                                f"\n\n=== 失敗回應 #{fail_count} (時間: {timestamp}) ===\n"
                            )
                            file.write(
                                f"處理第 {task_count-5} 到 {task_count-1} 筆資料\n"
                            )
                            file.write(response)
                            file.write("\n=== 回應結束 ===\n")

    return summarized_results


def save_json(data, output_file):
    """將結果儲存為新的 JSON 檔案"""
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"儲存檔案時發生錯誤: {e}")


def clean_json(content):
    """清理 JSON 格式，確保符合要求"""
    try:
        content = re.sub(r"^```json\s*", "", content)  # 移除開頭的 ```json
        content = re.sub(r"\s*```.*$", "", content, flags=re.DOTALL)
        content = re.sub(r"^.*?(\[)", r"\1", content, flags=re.DOTALL)
        content = re.sub(r",\s*(\])", r"\1", content)
        content = re.sub(r"(\])\s*$", r"\1", content)
        return content
    except Exception as e:
        print(f"JSON 清理過程發生錯誤: {e}")
        return content


# 主程式
# [前面的程式碼保持不變，主要修改 main 函數]


def main():
    input_dir = "./data"
    output_dir = "./cleaned_data"
    done_dir = "./done"
    false_dir = "./false"

    # 確保所需的目錄都存在
    for directory in [output_dir, done_dir, false_dir]:
        os.makedirs(directory, exist_ok=True)

    # 取得所有 .json 檔案
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for json_file in json_files:
        print(f"\n處理檔案: {json_file}")
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, f"cleaned_{json_file}")
        done_path = os.path.join(done_dir, json_file)
        false_path = os.path.join(false_dir, json_file)

        # 重置計數器
        global task_count, fail_count, success_count, dataLoss_count
        task_count = 1
        fail_count = 0
        success_count = 0
        dataLoss_count = 0

        try:
            json_data = read_json(input_path)
            if json_data is None:
                print(f"無法讀取檔案 {json_file}，跳過處理")
                shutil.move(input_path, false_path)
                continue

            if isinstance(json_data, list):
                total_items = len(json_data)
                json_data = filter_json_data(json_data)
                clean_data = process_json_in_batches(
                    json_data, chunk_size=5, source_filename=json_file
                )

                # 計算處理失敗的比例
                failed_items = (fail_count + dataLoss_count) * 5

                # 如果所有資料都處理失敗，移到 FALSE 資料夾
                if failed_items >= total_items:
                    shutil.move(input_path, false_path)
                    print(f"檔案全部處理失敗，移至 FALSE 資料夾: {json_file}")
                else:
                    # 如果只有部分失敗，仍然儲存結果並移到 DONE 資料夾
                    save_json(clean_data, output_path)
                    shutil.move(input_path, done_path)
                    print(f"檔案部分成功處理，移至 DONE 資料夾: {json_file}")

                print(f"總資料數: {total_items} 筆")
                print(f"處理失敗: {fail_count * 5} 筆資料")
                if dataLoss_count > 0:
                    print(f"AI 拒絕處理: {dataLoss_count * 5} 筆資料")
                print(f"成功處理: {len(clean_data)} 筆資料")
            else:
                print(f"檔案 {json_file} 格式錯誤，應該是一個陣列！")
                shutil.move(input_path, false_path)
        except Exception as e:
            print(f"處理檔案 {json_file} 時發生未預期的錯誤: {e}")
            shutil.move(input_path, false_path)


if __name__ == "__main__":
    main()
