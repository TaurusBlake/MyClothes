import json
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
# 1. 修改 Imports：導入 Gemini 的 Chat Model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. 設定 ---
# 2. 修改設定與認證：載入 .env 檔案
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ES_HOST = "http://localhost:9200"
INDEX_NAME = "virtual_closet"
PROMPT_FOLDER = "./prompts"
# 3. 修改設定與認證：選擇 Gemini 模型
# 即使您提到 gemini-2.5-pro，目前 API 中最穩定且推薦的頂級模型是 gemini-1.5-pro-latest
DATA_MODEL = "gemini-1.5-flash" 

# --- 2. 輔助函式 ---
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 4. 更新函式型別提示：將 ChatOllama 替換為 ChatGoogleGenerativeAI
def generate_es_queries(user_request: str, llm: ChatGoogleGenerativeAI) -> dict:
    """步驟 1: 將自然語言轉換為 Elasticsearch 查詢"""
    print(f"\n[1/4] 🧠 正在將 '{user_request}' 轉換為 ES 查詢 (使用 Gemini)...")
    
    prompt_template_str = load_prompt(os.path.join(PROMPT_FOLDER, "3_rag_query_prompt.txt"))
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
    
    chain = prompt_template | llm | StrOutputParser()
    
    query_str = chain.invoke({"user_request": user_request})
    print(f"  -> 生成的查詢指令: {query_str}")
    
    if query_str.strip().startswith("```json"):
        cleaned_query_str = query_str.strip()[7:-3].strip()
    else:
        cleaned_query_str = query_str
    
    return json.loads(cleaned_query_str)

def search_clothes(es_client: Elasticsearch, query: dict, size: int = 3) -> list:
    """步驟 2: 在 Elasticsearch 中搜尋衣服"""
    print(f"\n[2/4] 🔍 正在 Elasticsearch 中搜尋...")
    response = es_client.search(index=INDEX_NAME, query=query, size=size)
    hits = [hit['_source'] for hit in response['hits']['hits']]
    print(f"  -> 找到了 {len(hits)} 件相符的衣物。")
    return hits

# 4. 更新函式型別提示：將 ChatOllama 替換為 ChatGoogleGenerativeAI
def generate_recommendation_text(outfits: list, user_request: str, llm: ChatGoogleGenerativeAI) -> str:
    """步驟 4: 生成人性化的推薦文案"""
    print(f"\n[4/4] ✍️ 正在生成推薦文案 (使用 Gemini)...")

    context = f"使用者的需求是：'{user_request}'。\n"
    context += "根據需求，我為他搭配了以下幾套穿搭：\n"
    for i, (top, bottom) in enumerate(outfits):
        context += f"\n套裝 {i+1}:\n"
        context += f"- 上衣: {top['tags']['sub_category']} (風格: {', '.join(top['tags']['style_tags'])})\n"
        context += f"- 下著: {bottom['tags']['sub_category']} (風格: {', '.join(bottom['tags']['style_tags'])})\n"

    prompt = f"""
    You are a friendly and professional fashion stylist.
    Based on the following context, write a short, encouraging, and descriptive recommendation summary for the user.
    Address the user directly and explain in 1-2 sentences why these combinations work for their request.
    Keep the language natural and engaging. Speak in Traditional Chinese.

    CONTEXT:
    {context}

    YOUR RECOMMENDATION:
    """
    
    return llm.invoke(prompt).content

def main():
    if not GOOGLE_API_KEY:
        print("錯誤：找不到 GOOGLE_API_KEY。請確認您的 .env 檔案已設定正確。")
        return

    user_request = "我明天要去郊外踏青，我該怎麼搭配呢？"
    
    # 5. 修改模型初始化：使用 ChatGoogleGenerativeAI
    print(f"正在初始化 Gemini 模型: {DATA_MODEL}")
    data_expert = ChatGoogleGenerativeAI(model=DATA_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.5)
    
    es = Elasticsearch(hosts=[ES_HOST])

    try:
        es_queries = generate_es_queries(user_request, data_expert)
        
        top_results = search_clothes(es, es_queries["top_query"])
        bottom_results = search_clothes(es, es_queries["bottom_query"])

        if not top_results or not bottom_results:
            print("\n❌ 抱歉，您的衣櫥中找不到足夠的衣物來進行搭配。")
            return
            
        num_outfits = min(len(top_results), len(bottom_results), 3)
        outfits = []
        for i in range(num_outfits):
            outfits.append((top_results[i], bottom_results[i]))
        print(f"\n[3/4] 👕👖 已成功組合 {num_outfits} 套穿搭。")

        recommendation_text = generate_recommendation_text(outfits, user_request, data_expert)

        print("\n" + "="*50)
        print("✨ 為您專屬的穿搭建議 ✨")
        print("="*50)
        print(recommendation_text)
        print("\n--- 推薦組合詳情 ---")
        for i, (top, bottom) in enumerate(outfits):
            print(f"\n組合 {i+1}:")
            print(f"  - 上衣 👚: {top['image_path']}")
            print(f"  - 下著 👖: {bottom['image_path']}")
        print("="*50)

    except Exception as e:
        print(f"\n❌ 執行過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()