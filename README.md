# MyCloset AI 衣櫃助手

一個基於AI的智能衣櫃管理系統，使用計算機視覺和自然語言處理來分析衣物並提供搭配建議。

## 功能特色

- **衣物分析**: 使用AI模型分析衣物圖片，識別類別、顏色、材質等屬性
- **智能搭配**: 基於衣物屬性提供個性化的搭配建議
- **數據管理**: 使用Elasticsearch存儲和管理衣物數據
- **多模型支持**: 支持多種AI模型進行衣物分析

## 專案結構

```
mycloset/
├── my_clothes/          # 衣物圖片目錄
├── prompts/             # AI提示詞模板
├── ingest_clothes.py    # 衣物數據導入腳本
├── recommend_outfits.py # 搭配推薦系統
├── test_*.py           # 測試腳本
├── requirements.txt    # Python依賴
└── README.md          # 專案說明
```

## 安裝與設置

1. 克隆專案
```bash
git clone <repository-url>
cd mycloset
```

2. 創建虛擬環境
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

3. 安裝依賴
```bash
pip install -r requirements.txt
```

4. 設置環境變數
```bash
cp .env.example .env
# 編輯 .env 文件配置必要的API密鑰和設置
```

## 使用方法

### 導入衣物數據
```bash
python ingest_clothes.py
```

### 獲取搭配建議
```bash
python recommend_outfits.py
```

### 測試AI模型
```bash
python test_models.py
```

## 技術棧

- **Python 3.8+**
- **OpenCV** - 圖像處理
- **Pillow** - 圖像操作
- **Elasticsearch** - 數據存儲
- **LangChain** - AI模型集成
- **Ollama** - 本地AI模型
- **SQLAlchemy** - 數據庫ORM

## 開發

### 運行測試
```bash
python test_data_specialists.py
python test_gemma3.py
python test_models.py
```

### 檢查數據庫
```bash
python check_db.py
```

## 貢獻

歡迎提交Issue和Pull Request來改進這個專案。

## 授權

MIT License 