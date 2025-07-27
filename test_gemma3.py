import os
import base64
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from rembg import remove

# --- 1. 設定 ---
TEST_IMAGE_DIRECTORY = "./my_clothes"  # 使用我們之前建立的測試圖片資料夾
PROMPT_FILE = "llava_prompt_template.txt" # 使用我們之前優化過的 v2 版本的 Prompt
OLLAMA_HOST_IP = "localhost"  # <--- 請務必確認這是您正確的 Windows IP
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST_IP}:11434"
MODEL_TO_TEST = "gemma3:12b"  # <--- 我們的挑戰者！

def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def image_to_base64(image_path):
    with open(image_path, "rb") as input_file:
        input_data = input_file.read()
        output_data = remove(input_data)
        return base64.b64encode(output_data).decode('utf-8')

def main():
    try:
        prompt_template = load_prompt(PROMPT_FILE)
        print(f"成功讀取提示詞模板檔案: {PROMPT_FILE}")
    except FileNotFoundError:
        print(f"錯誤：找不到提示詞模板檔案 '{PROMPT_FILE}'。")
        return

    test_image_files = [f for f in os.listdir(TEST_IMAGE_DIRECTORY) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"\n===== 開始測試單一全能模型: {MODEL_TO_TEST} =====")

    # 初始化我們的挑戰者模型
    llm = ChatOllama(model=MODEL_TO_TEST, base_url=OLLAMA_BASE_URL, temperature=0, format="json")

    for image_name in test_image_files:
        image_path = os.path.join(TEST_IMAGE_DIRECTORY, image_name)
        print(f"\n{'='*20} 正在測試圖片: {image_name} {'='*20}")
        
        image_base64 = image_to_base64(image_path)
        
        try:
            msg = llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt_template},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"},
                        ]
                    )
                ]
            )
            
            # 嘗試直接解析JSON，因為我們在模型初始化時加了 format="json"
            result_json = json.loads(msg.content)
            
            print("成功解析 JSON！輸出結果：")
            # 使用 json.dumps 美化打印輸出
            print(json.dumps(result_json, indent=2, ensure_ascii=False))

        except json.JSONDecodeError:
            print(f"!!! 錯誤：模型回傳的內容不是有效的 JSON。")
            print("--- 原始輸出 ---")
            print(msg.content if 'msg' in locals() else "無法獲取模型輸出")
            print("-----------------")
        except Exception as e:
            print(f"!!! 處理圖片時發生未知錯誤: {e}")

if __name__ == "__main__":
    main()