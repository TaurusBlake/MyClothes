import os
import base64
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rembg import remove

# --- 1. 設定 ---
TEST_IMAGE_DIRECTORY = "./my_clothes"
PROMPT_FOLDER = "./prompts" # 建議將 prompts 放在專門的資料夾
OLLAMA_HOST_IP = "localhost"  # <--- 請務必確認這是您正確的 Windows IP
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST_IP}:11434"

# --- 2. 建立 Prompt 檔案 ---
# 請建立一個 prompts 資料夾，並在其中建立以下兩個 .txt 檔案

# prompts/1_vision_expert_prompt.txt 的內容：
# You are a fashion assistant. Describe the clothing item in the image in a detailed paragraph. Focus on its primary category (like top, bottom, dress), sub-category (like t-shirt, jeans), main color, any other visible colors, pattern, and style. Be objective and descriptive. Ignore the background completely.

# prompts/2_data_expert_prompt.txt 的內容 (注意 {description_from_llava} 和 {schema_and_constraints} 是稍後會被替換的變數)：
# You are a data processing expert. Your task is to read the following clothing description text and convert it into a structured JSON object based on the provided schema and constraints.
#
# ### DESCRIPTION TEXT ###
# {description_from_llava}
#
# ### JSON SCHEMA AND CONSTRAINTS ###
# {schema_and_constraints}
#
# Now, generate the JSON object.

def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def image_to_base64(image_path):
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
        output_data = remove(input_data)
        return base64.b64encode(output_data).decode('utf-8')

def main():
    # 載入我們之前定義好的 Schema 和 Constraints
    # 為了方便，這裡先寫死，您也可以將它們存成檔案讀取
    schema_and_constraints = """
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
    
    try:
        vision_prompt = load_prompt(os.path.join(PROMPT_FOLDER, "1_vision_expert_prompt.txt"))
        data_prompt_template = load_prompt(os.path.join(PROMPT_FOLDER, "2_data_expert_prompt.txt"))
    except FileNotFoundError:
        print(f"錯誤：找不到 Prompt 檔案。請確認 '{PROMPT_FOLDER}' 資料夾及內部檔案是否存在。")
        return

    # 初始化兩位專家
    vision_expert = ChatOllama(model="llava:13b", base_url=OLLAMA_BASE_URL, temperature=0)
    data_expert = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL, format="json", temperature=0)

    test_image_files = [f for f in os.listdir(TEST_IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in test_image_files:
        print(f"\n{'='*20} 正在處理圖片: {image_name} {'='*20}")
        image_path = os.path.join(TEST_IMAGE_DIRECTORY, image_name)
        image_base64 = image_to_base64(image_path)
        
        # --- 步驟 1: 視覺專家 (Llava) 進行描述 ---
        print("\n--- 視覺專家 (Llava) 正在描述圖片... ---")
        vision_msg = vision_expert.invoke([HumanMessage(content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        ])])
        description = vision_msg.content
        print(description)
        
        # --- 步驟 2: 數據專家 (Llama 3.1) 進行結構化轉換 ---
        print("\n--- 數據專家 (Llama 3.1) 正在轉換為 JSON... ---")
        final_prompt = data_prompt_template.format(
            description_from_llava=description,
            schema_and_constraints=schema_and_constraints
        )
        json_msg = data_expert.invoke(final_prompt)
        final_json = json_msg.content
        print(final_json)

if __name__ == "__main__":
    main()