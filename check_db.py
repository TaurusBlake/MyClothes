from elasticsearch import Elasticsearch
import json

# --- 設定 ---
ES_HOST = "http://localhost:9200"
INDEX_NAME = "virtual_closet"

def main():
    print("正在連接 Elasticsearch...")
    es = Elasticsearch(hosts=[ES_HOST])

    # 1. 確認文件總數
    print("\n--- 1. 正在確認文件總數 ---")
    count_response = es.count(index=INDEX_NAME)
    print(f"索引 '{INDEX_NAME}' 中共有 {count_response['count']} 筆文件。")

    # 2. 查看 2 筆資料的詳細內容
    print("\n--- 2. 正在抽樣查看 2 筆資料 ---")
    search_response = es.search(index=INDEX_NAME, size=2)
    for hit in search_response['hits']['hits']:
        # 使用 json.dumps 美化打印
        print(json.dumps(hit['_source'], indent=2, ensure_ascii=False))

    # 3. 進行一個簡單的關鍵字搜尋
    print("\n--- 3. 正在搜尋風格為 '日常休閒' 的衣物 ---")
    query = {
        "match": {
            "tags.style_tags": "日常休閒"
        }
    }
    search_response = es.search(index=INDEX_NAME, query=query)
    print(f"找到了 {len(search_response['hits']['hits'])} 件 '日常休閒' 風格的衣物。")
    for hit in search_response['hits']['hits']:
        print(f"  - 圖片路徑: {hit['_source']['image_path']}")

if __name__ == "__main__":
    main()