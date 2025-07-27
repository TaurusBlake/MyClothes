import os
import base64
import json
import csv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rembg import remove

# --- 1. 設定 ---
TEST_IMAGE_DIRECTORY = "./my_clothes"
PROMPT_FOLDER = "./prompts"
OLLAMA_HOST_IP = "localhost"  # <--- 請務必確認這是您正確的 Windows IP
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST_IP}:11434"

VISION_MODEL = "llava:13b"
DATA_MODELS_TO_TEST = ["llama3.1:8b", "gemma3:12b"] # <--- 兩位數據專家選手

CSV_FILENAME = "data_specialist_comparison.csv"
CSV_HEADERS = [
    "image_name", "model_name", "primary_category", "sub_category", "main_color",
    "secondary_colors", "pattern", "sleeve_length", "neckline", "fit",
    "material_guess", "suitable_seasons", "style_tags", "occasion_tags", "raw_json_output"
]

# --- 2. 輔助函式 ---
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def image_to_base64(image_path):
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
        output_data = remove(input_data)
        return base64.b64encode(output_data).decode('utf-8')

def get_schema_and_constraints():
    # 將 Schema 和 Constraints 集中管理，方便維護
    return """
    The output must be a single, valid JSON object. Do not add any text before or after the JSON object. The JSON object must conform to the following schema:
    {
      "primary_category": "string", "sub_category": "string", "main_color": "string", "secondary_colors": ["string"],
      "pattern": "string", "sleeve_length": "string", "neckline": "string", "fit": "string",
      "material_guess": "string", "suitable_seasons": ["string"], "style_tags": ["string"], "occasion_tags": ["string"]
    }
    
    CONSTRAINTS:
    - primary_category: ["上衣", "下著", "連身裙", "外套", "配件"]
    - pattern: ["素色", "條紋", "格紋", "印花", "波點", "迷彩"]
    - sleeve_length: ["無袖", "短袖", "五分袖", "七分袖", "長袖", "不適用"]
    - neckline: ["圓領", "V領", "方領", "高領", "Polo領", "連帽", "不適用"]
    - fit: ["緊身", "合身", "常規", "寬鬆", "Oversized"]
    - suitable_seasons: ["春季", "夏季", "秋季", "冬季"]
    - style_tags": ["日常休閒", "商務休閒", "正式", "街頭潮流", "運動機能", "簡約", "甜美", "復古"]
    - occasion_tags": ["上班通勤", "商務會議", "約會", "派對晚宴", "戶外運動", "旅行度假", "居家"]
    CRITICAL: If a field is not applicable, you MUST use the string "不適用".
    """

def main():
    try:
        vision_prompt = load_prompt(os.path.join(PROMPT_FOLDER, "1_vision_expert_prompt.txt"))
        data_prompt_template = load_prompt(os.path.join(PROMPT_FOLDER, "2_data_expert_prompt.txt"))
    except FileNotFoundError:
        print(f"錯誤：找不到 Prompt 檔案。請確認 '{PROMPT_FOLDER}' 資料夾及內部檔案是否存在。")
        return

    schema_and_constraints = get_schema_and_constraints()
    vision_expert = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    test_image_files = [f for f in os.listdir(TEST_IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CSV_HEADERS)
        print(f"已建立報告檔案: {CSV_FILENAME}")

        for image_name in test_image_files:
            print(f"\n{'='*20} 正在處理圖片: {image_name} {'='*20}")
            image_path = os.path.join(TEST_IMAGE_DIRECTORY, image_name)
            image_base64 = image_to_base64(image_path)
            
            # --- 步驟 1: Llava 生成一次性的描述 ---
            print("\n--- 視覺專家 (Llava) 正在生成描述... ---")
            vision_msg = vision_expert.invoke([HumanMessage(content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ])])
            description = vision_msg.content
            print(f"生成描述: {description[:100]}...") # 打印部分描述以供參考

            # --- 步驟 2: 遍歷數據專家，進行挑戰賽 ---
            for model_name in DATA_MODELS_TO_TEST:
                print(f"\n--- 數據專家 ({model_name}) 正在轉換為 JSON... ---")
                data_expert = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, format="json", temperature=0)
                
                final_prompt = data_prompt_template.format(
                    description_from_llava=description,
                    schema_and_constraints=schema_and_constraints
                )
                
                json_msg = data_expert.invoke(final_prompt)
                json_result_str = json_msg.content

                # --- 步驟 3: 將結果寫入 CSV ---
                row_data = {"image_name": image_name, "model_name": model_name, "raw_json_output": json_result_str}
                try:
                    parsed_json = json.loads(json_result_str)
                    for header in CSV_HEADERS:
                        if header not in ["image_name", "model_name", "raw_json_output"]:
                            value = parsed_json.get(header, 'N/A')
                            row_data[header] = ", ".join(value) if isinstance(value, list) else value
                except (json.JSONDecodeError, AttributeError):
                    for header in CSV_HEADERS:
                        if header not in ["image_name", "model_name", "raw_json_output"]:
                            row_data[header] = "INVALID_JSON"
                
                csv_writer.writerow([row_data.get(h, '') for h in CSV_HEADERS])
                print(f"--> {model_name} 的結果已寫入 {CSV_FILENAME}")

if __name__ == "__main__":
    main()