# ETF 損益分析與預測系統

資工系畢業專題 —— 用 Python + Streamlit 做台股／ETF 的技術分析、定期定額回測、
市場掃描、AI 預測與新聞情緒分析。

---

## 一、安裝

```bash
pip install -r requirements.txt
```

> **TensorFlow 很大（350 MB+），下載慢是正常的。**
> 如果只想先看 TAB1、TAB2、TAB3、TAB5，可以先把 requirements.txt 最後一行
> `tensorflow-cpu` 註解掉（前面加 `#`），之後要訓練模型再裝。
> 程式偵測不到 TensorFlow 時，TAB4 會自動切換成「只能讀已存結果」模式，不會壞掉。

下載太慢時可以改用鏡像站：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 二、執行

```bash
streamlit run app.py
```

## 三、第一次使用的建議順序

開啟後第一個分頁就是 **🩺 系統健檢**，由上往下跑一次即可：

| 步驟 | 內容 | 說明 |
|---|---|---|
| 一 | 股價資料來源 | 逐一測試 FinMind 與 Yahoo，列出真實錯誤訊息 |
| 二 | FinBERT 情緒模型 | 用三句測試句確認正向／負向／中立判斷正確 |
| 三 | 執行環境 | Python 版本、套件版本、有沒有在用 GPU |
| 四 | 本機資料存量 | 已存幾份預測結果、累積幾天新聞 |

最下方還有「系統紀錄」，任何分頁出問題時來這裡看真正的原因。

兩項健檢通過後：

1. 到 **🤖 AI 多模態月度預測** 選「🔥 重新訓練」跑一次（會自動存檔）
2. 之後展示改選「📂 讀取已存結果」，秒開
3. 到 **📰 新聞情緒分析引擎** 搜尋後按「存入本機資料庫」，開始累積資料

> 💡 `taiwan_stock_daily_adj` 顯示失敗是**正常的** —— 還原股價 API 需要
> FinMind 付費贊助帳號，程式會自動改用免費 API，不影響功能。

## 四、資料夾結構

```
專題資料夾/
├── app.py                    ← 主程式，只負責組裝分頁（很短）
├── config.py                 ← 所有設定值都在這裡
├── collect_news.py           ← 每日新聞蒐集腳本（雙擊或指令執行）
├── export_all.py             ← 程式碼彙整工具（做簡報用）
├── requirements.txt
│
├── core/                     ← 核心功能，每個檔案只做一件事
│   ├── datasource.py         ← 抓股價（FinMind / Yahoo）
│   ├── indicators.py         ← 技術指標（MA、RSI、MACD…）
│   ├── news.py               ← 抓新聞
│   ├── sentiment.py          ← FinBERT 情緒分析
│   ├── model.py              ← KMeans + TCN 模型
│   ├── finance.py            ← XIRR 年化報酬率、最大回撤
│   └── storage.py            ← 本機存檔讀檔
│
├── tabs/                     ← 五個分頁的畫面，每個一個檔案
│   ├── tab0_health.py        ← 系統健檢（第一個分頁）
│   ├── tab1_technical.py
│   ├── tab2_dca.py
│   ├── tab3_scanner.py
│   ├── tab4_predict.py
│   └── tab5_news.py
│
└── data/                     ← 程式自動建立，不用手動處理
    ├── news/                 ← 每日新聞 CSV
    ├── results/              ← 預測結果 .pkl
    └── app.log               ← 出錯時來這裡看真正的原因
```

**要改東西時去哪裡找：**

| 想做的事 | 改哪個檔案 |
|---|---|
| 調預測天數、訓練輪數、新聞關鍵字 | `config.py` |
| 改某個分頁的畫面 | `tabs/tabN_xxx.py` |
| 改技術指標算法 | `core/indicators.py` |
| 改模型架構 | `core/model.py` |
| 換情緒模型 | `config.py` 的 `SENTIMENT_MODEL_CANDIDATES` |

---

## 五、FinMind Token（選用，但建議申請）

沒有 token 也能跑，只是：

| | 未登入 | 已登入 |
|---|---|---|
| 請求上限 | 300 次／小時 | 600 次／小時 |
| 還原股價 API | 不可用 | 需付費贊助帳號才可用 |

**申請後怎麼設定（三選一，程式會自己找）：**

最簡單的方式 —— 建立 `data/finmind_token.txt`，裡面只放 token 這一行字。

或者建立 `.streamlit/secrets.toml`：

```toml
FINMIND_TOKEN = "你的token"
```

> ⚠️ 未來若要上傳 GitHub，**token 檔案千萬不要跟著上傳**。到時候再處理即可。

---

## 六、每日新聞蒐集

```bash
python collect_news.py
```

會把當天的新聞情緒存到 `data/news/YYYY-MM-DD.csv`。
同一天重複執行會自動合併去重，跑幾次都沒關係。

要改追蹤哪些股票，編輯 `config.py`：

```python
WATCHLIST = {"2330": "台積電", "0050": "元大台灣50", "2454": "聯發科"}
COLLECT_TERMS = ["營收", "財報", "法說會", "訂單", "展望"]
```

### Windows 設定自動執行（不用每天記得跑）

1. 建立 `run_collect.bat`，內容是：
   ```bat
   cd /d "C:\你的專題資料夾路徑"
   python collect_news.py
   ```
2. 開「工作排程器」（在開始功能表搜尋 Task Scheduler）
3. 建立基本工作 → 每天 → 設定時間（建議 09:00）→ 啟動程式 → 選 `run_collect.bat`

設好之後電腦開著就會自動跑，不需要人工介入。

---

## 七、已知限制（報告請誠實寫出來）

1. **新聞情緒覆蓋率低** —— Google News RSS 只提供近期新聞，五年訓練資料中
   絕大多數交易日的情緒值為 0，FinBERT 對 TCN 的實際貢獻有限。
   `collect_news.py` 每天累積資料就是為了改善這一點。

2. **股價預測的本質困難** —— 以每日報酬率為目標搭配 MSE 損失時，模型容易
   收斂到「永遠輸出接近 0」。TAB4 的「模型健檢」區塊會用標準差比與方向準確率
   主動揭露這個現象，而不是只報一個好看的 MAE。

3. **FinMind 免費帳戶** —— 還原股價 API 需付費贊助帳號，免費帳戶只能取得
   未還原原價。程式已自動降級處理，但長期回測建議使用 Yahoo Finance。

4. **本系統僅供學術展示，不構成任何投資建議。**

---

## 八、做簡報時：一鍵彙整全部程式碼

模組化之後檔案變多，要整份貼給 AI 或放進報告附錄時很麻煩。
這支工具會把所有程式碼合併成「一份帶目錄」的文件：

```bash
python export_all.py
```

會產生兩個檔案：

| 檔案 | 用途 |
|---|---|
| `專題程式碼彙整.md` | 有目錄、檔案結構圖、行數統計、每段說明。**貼給簡報 AI 用這份** |
| `專題程式碼彙整.txt` | 純文字版，適合貼進不支援 Markdown 的地方 |

文件開頭已經寫好「專案簡介」「技術組成表」「設計重點」三段，
可以直接當簡報的開場，不用自己再整理一次。

要調整章節順序或說明文字，編輯 `export_all.py` 最上方的 `FILES` 清單即可。
