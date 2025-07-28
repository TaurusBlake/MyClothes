import os
import json
import csv
import time
from dotenv import load_dotenv
import google.generativeai as genai
from elasticsearch import Elasticsearch
from rembg import remove
from PIL import Image
import io

# --- 1. 設定與初始化 ---
load_dotenv()
print("正在讀取環境變數...")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("錯誤：找不到 GOOGLE_API_KEY。請確認您的 .env 檔案已設定正確。")

genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-pro"

# --- 新增：定義處理後圖片的儲存資料夾 ---
IMAGE_DIRECTORY = "./test_pic" 
PROCESSED_IMAGE_DIRECTORY = "./my_clothes_processed" # <--- 去背後的圖片將存放在這裡
PROMPT_FILE = "./prompts/gemini_prompt.txt"
ES_HOST = "http://localhost:9200"
INDEX_NAME = "virtual_closet"
CSV_FILENAME = "gemini_ingestion_report_v2.csv"

CSV_HEADERS = [
    "original_image_name", "processed_image_path", "processing_time_seconds", "status", "error_message",
    "primary_category", "sub_category", "main_color", "secondary_colors",
    "pattern", "sleeve_length", "neckline", "fit", "material_guess",
    "suitable_seasons", "style_tags", "occasion_tags"
]

def process_and_save_image(image_name, input_dir, output_dir):
    """讀取圖片，移除背景，儲存為新的 PNG 檔案，並返回 PIL Image 物件"""
    input_path = os.path.join(input_dir, image_name)
    
    # 建立輸出檔案名稱與路徑
    file_name_without_ext = os.path.splitext(image_name)[0]
    output_filename = f"{file_name_without_ext}_processed.png"
    output_path = os.path.join(output_dir, output_filename)

    with open(input_path, "rb") as input_file:
        input_data = input_file.read()
        
        # 使用 rembg 移除背景
        output_data = remove(input_data)
        
        # 將去背後的圖片儲存到新資料夾
        with open(output_path, "wb") as output_file:
            output_file.write(output_data)
        
        # 返回 PIL Image 物件以供 Gemini 分析，以及新的檔案路徑
        return Image.open(io.BytesIO(output_data)), output_path

def main():
    print(f"正在初始化 Gemini 模型: {GEMINI_MODEL_NAME}")
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    print(f"正在讀取 Prompt 模板: {PROMPT_FILE}")
    prompt_template = load_prompt(PROMPT_FILE) # 假設 load_prompt 函式已定義

    print(f"正在連接 Elasticsearch: {ES_HOST}")
    es = Elasticsearch(hosts=[ES_HOST])
    
    # --- 新增：自動建立處理後的圖片資料夾 ---
    os.makedirs(PROCESSED_IMAGE_DIRECTORY, exist_ok=True)
    print(f"已確認處理後圖片儲存目錄: {PROCESSED_IMAGE_DIRECTORY}")

    if es.indices.exists(index=INDEX_NAME):
        print(f"發現舊索引 '{INDEX_NAME}'，正在刪除...")
        es.indices.delete(index=INDEX_NAME)
    print(f"正在建立新索引: {INDEX_NAME}")
    es.indices.create(index=INDEX_NAME)

    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CSV_HEADERS)
        print(f"已建立報告檔案: {CSV_FILENAME}")

        image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_name in image_files:
            print(f"\n{'='*20} 正在處理圖片: {image_name} {'='*20}")
            
            row_data = {"original_image_name": image_name}
            start_time = time.monotonic()

            try:
                # 步驟 1: 圖片預處理與儲存
                pil_image, processed_path = process_and_save_image(image_name, IMAGE_DIRECTORY, PROCESSED_IMAGE_DIRECTORY)
                row_data["processed_image_path"] = processed_path
                print(f"  -> 圖片已去背並儲存至: {processed_path}")
                
                # 步驟 2: 呼叫 Gemini API
                response = model.generate_content([prompt_template, pil_image])
                
                # 步驟 3: 清理並解析 JSON
                cleaned_response = response.text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-3].strip()
                
                tags_data = json.loads(cleaned_response)
                
                # 步驟 4: 寫入 Elasticsearch (使用處理後的圖片路徑)
                doc = {"image_path": processed_path, "tags": tags_data} # <--- 使用新的路徑
                es.index(index=INDEX_NAME, document=doc)
                
                row_data["status"] = "SUCCESS"
                row_data["error_message"] = ""
                for key, value in tags_data.items():
                    row_data[key] = ", ".join(value) if isinstance(value, list) else value

                print(f"  -> 處理成功！")

            except Exception as e:
                print(f"  !!! 處理時發生錯誤: {e}")
                row_data["status"] = "FAILED"
                row_data["error_message"] = str(e)
            
            end_time = time.monotonic()
            duration = end_time - start_time
            row_data["processing_time_seconds"] = f"{duration:.2f}"
            
            # 步驟 5: 將結果寫入 CSV 報告
            csv_writer.writerow([row_data.get(h, '') for h in CSV_HEADERS])
            print(f"  -> 結果已寫入 CSV (耗時: {duration:.2f} 秒)")

# 假設 load_prompt 函式
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    main()