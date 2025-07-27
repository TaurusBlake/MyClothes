import os
import base64
import json
from elasticsearch import Elasticsearch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rembg import remove

# --- 1. 設定 ---
IMAGE_DIRECTORY = "./my_clothes"  # <--- 請將此路徑替換成您存放44張照片的資料夾
PROMPT_FOLDER = "./prompts"
ES_HOST = "http://localhost:9200"
INDEX_NAME = "virtual_closet"
OLLAMA_HOST_IP = "localhost"  # <--- 請務必確認這是您正確的 Windows IP
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST_IP}:11434"

# --- 我們的冠軍團隊 ---
VISION_MODEL = "llava:13b"
DATA_MODEL = "gemma3:12b"

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
    # 將 Schema 和 Constraints 集中管理
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
    - suitable_seasons": ["春季", "夏季", "秋季", "冬季"]
    - style_tags": ["日常休閒", "商務休閒", "正式", "街頭潮流", "運動機能", "簡約", "甜美", "復古"]
    - occasion_tags": ["上班通勤", "商務會議", "約會", "派對晚宴", "戶外運動", "旅行度假", "居家"]
    CRITICAL: If a field is not applicable, you MUST use the string "不適用".
    """

def main():
    # --- 初始化 ---
    try:
        vision_prompt = load_prompt(os.path.join(PROMPT_FOLDER, "1_vision_expert_prompt.txt"))
        data_prompt_template = load_prompt(os.path.join(PROMPT_FOLDER, "2_data_expert_prompt.txt"))
    except FileNotFoundError:
        print(f"錯誤：找不到 Prompt 檔案。請確認 '{PROMPT_FOLDER}' 資料夾及內部檔案是否存在。")
        return

    schema_and_constraints = get_schema_and_constraints()
    vision_expert = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    data_expert = ChatOllama(model=DATA_MODEL, base_url=OLLAMA_BASE_URL, format="json", temperature=0)
    
    es = Elasticsearch(hosts=[ES_HOST])
    
    # 每次運行前，都先刪除舊索引，確保資料最新
    if es.indices.exists(index=INDEX_NAME):
        print(f"發現舊索引 '{INDEX_NAME}'，正在刪除...")
        es.indices.delete(index=INDEX_NAME)
    print(f"正在建立新索引: {INDEX_NAME}")
    es.indices.create(index=INDEX_NAME)
    
    image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]
    
    for image_name in image_files:
        print(f"\n{'='*20} 正在處理圖片: {image_name} {'='*20}")
        image_path = os.path.join(IMAGE_DIRECTORY, image_name)
        
        try:
            # --- 步驟 1: Llava 生成描述 ---
            print("  [1/3] 視覺專家 (Llava) 正在描述...")
            image_base64 = image_to_base64(image_path)
            vision_msg = vision_expert.invoke([HumanMessage(content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ])])
            description = vision_msg.content
            
            # --- 步驟 2: Gemma 3 轉換為 JSON ---
            print("  [2/3] 數據專家 (Gemma 3) 正在轉換...")
            final_prompt = data_prompt_template.format(
                description_from_llava=description,
                schema_and_constraints=schema_and_constraints
            )
            json_msg = data_expert.invoke(final_prompt)
            final_json_str = json_msg.content
            tags_data = json.loads(final_json_str)
            
            # --- 步驟 3: 存入 Elasticsearch ---
            print("  [3/3] 正在存入 Elasticsearch...")
            doc = { "image_path": image_path, "tags": tags_data }
            res = es.index(index=INDEX_NAME, document=doc)
            print(f"  --> 成功存入! ID: {res['_id']}")
            print("  --> JSON 內容:", json.dumps(tags_data, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"  !!! 處理圖片 {image_name} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()