import json
import os
from elasticsearch import Elasticsearch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. 設定 ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "virtual_closet"
PROMPT_FOLDER = "./prompts"
OLLAMA_HOST_IP = "localhost"  # <--- 請務必確認這是您正確的 Windows IP
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST_IP}:11434"

DATA_MODEL = "gemma3:12b"

# --- 2. 輔助函式 ---
def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_es_queries(user_request: str, llm: ChatOllama) -> dict:
    """步驟 1: 將自然語言轉換為 Elasticsearch 查詢"""
    print(f"\n[1/4] 🧠 正在將 '{user_request}' 轉換為 ES 查詢...")
    
    prompt_template_str = load_prompt(os.path.join(PROMPT_FOLDER, "3_rag_query_prompt.txt"))
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
    
    chain = prompt_template | llm | StrOutputParser()
    
    query_str = chain.invoke({"user_request": user_request})
    print(f"  -> 生成的查詢指令: {query_str}")
    
    # --- 【本次修正】 ---
    # 在解析 JSON 之前，先清理字串，移除可能的 Markdown 標記
    if query_str.strip().startswith("```json"):
        cleaned_query_str = query_str.strip()[7:-3].strip()
    else:
        cleaned_query_str = query_str
    
    return json.loads(cleaned_query_str)
    # --- 【修正結束】 ---

def search_clothes(es_client: Elasticsearch, query: dict, size: int = 3) -> list:
    """步驟 2: 在 Elasticsearch 中搜尋衣服"""
    print(f"\n[2/4] 🔍 正在 Elasticsearch 中搜尋...")
    response = es_client.search(index=INDEX_NAME, query=query, size=size)
    hits = [hit['_source'] for hit in response['hits']['hits']]
    print(f"  -> 找到了 {len(hits)} 件相符的衣物。")
    return hits

def generate_recommendation_text(outfits: list, user_request: str, llm: ChatOllama) -> str:
    """步驟 4: 生成人性化的推薦文案"""
    print(f"\n[4/4] ✍️ 正在生成推薦文案...")

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
    user_request = "我明天要去郊外踏青，我該怎麼搭配呢？"
    
    data_expert = ChatOllama(model=DATA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.5)
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