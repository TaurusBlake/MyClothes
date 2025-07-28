import os
import base64
import json
import csv
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rembg import remove
from PIL import Image
import io

# --- 1. 設定 ---
# 讀取 .env 檔案中的環境變數
load_dotenv()

TEST_IMAGE_DIRECTORY = "./my_clothes"
PROMPT_FOLDER = "./prompts"
OLLAMA_HOST_IP = "localhost"  # <--- 請務必確認這是您正確的 Windows IP
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST_IP}:11434"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ollama 專家團隊
VISION_MODEL = "llava:13b"
DATA_MODEL = "gemma3:12b"
# Gemini 挑戰者
GEMINI_MODEL = "gemini-2.5-pro" 

# 報告檔案
CSV_FILENAME = "ollama_vs_gemini_comparison.csv"
CSV_HEADERS = [
    "image_name", "method", "processing_time_seconds", "primary_category", "sub_category", "main_color",
    "secondary_colors", "pattern", "sleeve_length", "neckline", "fit",
    "material_guess", "suitable_seasons", "style_tags", "occasion_tags", "raw_json_output"
]

# --- 2. 輔助函式 ---
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def image_to_bytes(image_path):
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
        return remove(input_data)

def get_schema_and_constraints():
    # ... (此函式與您之前的版本完全相同，此處省略以保持簡潔)
    return """
    The output must be a single, valid JSON object... (略)
    """

# --- 3. 核心處理流程 ---
def process_with_ollama_chain(image_name, image_bytes, prompts):
    print("\n--- 正在使用 Ollama 專家鏈處理 ---")
    start_time = time.monotonic()

    vision_expert = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    data_expert = ChatOllama(model=DATA_MODEL, base_url=OLLAMA_BASE_URL, format="json", temperature=0)
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # 步驟 1: Llava 生成描述
    vision_msg = vision_expert.invoke([HumanMessage(content=[
        {"type": "text", "text": prompts["vision"]},
        {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
    ])])
    description = vision_msg.content
    
    # 步驟 2: Gemma 3 轉換為 JSON
    final_prompt = prompts["data"].format(
        description_from_llava=description,
        schema_and_constraints=prompts["schema"]
    )
    json_msg = data_expert.invoke(final_prompt)
    
    end_time = time.monotonic()
    return json_msg.content, end_time - start_time

def process_with_gemini(image_name, image_bytes, prompt):
    print("\n--- 正在使用 Gemini 2.5 Pro 處理 ---")
    start_time = time.monotonic()
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # 將圖片 bytes 轉為 Gemini API 需要的格式
    image_pil = Image.open(io.BytesIO(image_bytes))
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # 建立多模態請求
    response = model.generate_content([prompt, image_pil])
    
    end_time = time.monotonic()
    # Gemini 可能會在回應中加入 Markdown，我們需要清理它
    cleaned_response = response.text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:-3].strip()
        
    return cleaned_response, end_time - start_time

def main():
    if not GOOGLE_API_KEY:
        print("錯誤：找不到 GOOGLE_API_KEY。請確認您的 .env 檔案已設定正確。")
        return

    # 載入所有需要的 Prompts
    prompts = {
        "vision": load_prompt(os.path.join(PROMPT_FOLDER, "1_vision_expert_prompt.txt")),
        "data": load_prompt(os.path.join(PROMPT_FOLDER, "2_data_expert_prompt.txt")),
        "gemini": load_prompt(os.path.join(PROMPT_FOLDER, "llava_prompt_template.txt")),
        "schema": get_schema_and_constraints()
    }

    test_image_files = [f for f in os.listdir(TEST_IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CSV_HEADERS)
        print(f"已建立報告檔案: {CSV_FILENAME}")

        for image_name in test_image_files:
            print(f"\n{'='*20} 正在處理圖片: {image_name} {'='*20}")
            image_path = os.path.join(TEST_IMAGE_DIRECTORY, image_name)
            image_bytes = image_to_bytes(image_path) # 移除背景後的圖片 bytes
            
            # 執行兩個流程
            all_results = {
                "ollama_chain": process_with_ollama_chain(image_name, image_bytes, prompts),
                "gemini_api": process_with_gemini(image_name, image_bytes, prompts["gemini"])
            }
            
            # 將結果寫入 CSV
            for method, (json_str, duration) in all_results.items():
                row_data = {"image_name": image_name, "method": method, "processing_time_seconds": f"{duration:.2f}", "raw_json_output": json_str}
                try:
                    parsed_json = json.loads(json_str)
                    for header in CSV_HEADERS:
                        if header not in row_data:
                            value = parsed_json.get(header, 'N/A')
                            row_data[header] = ", ".join(value) if isinstance(value, list) else value
                except (json.JSONDecodeError, AttributeError):
                    for header in CSV_HEADERS:
                        if header not in row_data:
                            row_data[header] = "INVALID_JSON"
                
                csv_writer.writerow([row_data.get(h, '') for h in CSV_HEADERS])
                print(f"--> {method} 的結果已寫入 CSV (耗時: {duration:.2f} 秒)")

if __name__ == "__main__":
    main()