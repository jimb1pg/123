# -*- coding: utf-8 -*-
"""
程式碼彙整工具 —— 把整個專案打包成「一份」帶目錄的文件。

用途：
  做簡報或報告時，把整份程式碼交給 AI 或直接貼進文件，
  不用一個檔案一個檔案複製。

用法：
    python export_all.py

會在專案根目錄產生兩個檔案：
    專題程式碼彙整.md    帶目錄與說明，適合貼給 AI 或轉成報告附錄
    專題程式碼彙整.txt    純文字版，適合貼進不支援 Markdown 的地方
"""
import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent

# 匯出順序＝閱讀順序：先總覽，再核心，最後畫面
FILES = [
    ("HANDOFF.md", "專題交接文件", "開發日誌、技術決策、已知問題、待辦清單"),
    ("app.py", "主程式", "只負責組裝分頁，實作都在 tabs/ 底下"),
    ("config.py", "全域設定", "所有可調參數集中在此：預測天數、關鍵字、模型名稱、資料夾路徑"),
    ("core/datasource.py", "股價資料來源", "FinMind 與 Yahoo Finance 的統一入口，含快取、自動降級與健檢"),
    ("core/indicators.py", "技術指標", "MA、RSI、MACD、ATR、波動率。全系統共用同一套定義"),
    ("core/news.py", "新聞抓取", "Google News RSS 搜尋與去重，不含情緒分析"),
    ("core/sentiment.py", "情緒分析", "FinBERT 模型載入、批次推論、每日聚合、關鍵字權重"),
    ("core/model.py", "預測模型", "KMeans 市場狀態 + TCN；含 purged split 與 baseline 診斷"),
    ("core/finance.py", "金融計算", "XIRR 年化報酬率、最大回撤 MDD"),
    ("core/storage.py", "本機存取", "預測結果、每日新聞、蒐集清單、預測紀錄的讀寫"),
    ("tabs/tab0_health.py", "分頁零：系統健檢", "資料來源、情緒模型、執行環境、資料存量的一站式診斷"),
    ("tabs/tab1_technical.py", "分頁一：技術分析", "K 線／折線圖、MA 均線、RSI 超買超賣區填色"),
    ("tabs/tab2_dca.py", "分頁二：定期定額回測", "ROI、XIRR 年化報酬率、最大回撤、資產累積走勢"),
    ("tabs/tab3_scanner.py", "分頁三：市場掃描", "多標的批次掃描與排序，支援自訂清單"),
    ("tabs/tab4_predict.py", "分頁四：AI 預測", "訓練／讀檔雙模式、模型健檢、預測區間圖、預測 vs 實際事後驗證"),
    ("tabs/tab5_news.py", "分頁五：新聞情緒", "即時搜尋、情緒趨勢、關鍵字權重、本機資料累積"),
    ("collect_news.py", "每日蒐集腳本", "獨立執行，不依賴 Streamlit，支援指令參數，可設定為排程自動執行"),
    ("requirements.txt", "套件需求", "相依套件清單"),
]

HEADER = """# {title}

> 本文件由 `export_all.py` 自動產生
> 產生時間：{now}
> 共 {n_files} 個檔案、{n_lines} 行程式碼

## 專案簡介

以 Python + Streamlit 建置的台股／ETF 分析系統，包含技術分析、定期定額回測、
市場掃描、AI 預測與新聞情緒分析五大功能模組。

**技術組成**

| 層面 | 使用技術 |
|---|---|
| 前端介面 | Streamlit |
| 圖表 | Plotly |
| 資料來源 | FinMind API、Yahoo Finance |
| 技術指標 | pandas 自行實作（MA / RSI / MACD / ATR） |
| 市場分群 | scikit-learn KMeans |
| 情緒分析 | HuggingFace Transformers（FinBERT） |
| 預測模型 | TensorFlow / Keras（TCN，時間卷積網路） |

**設計重點**

1. **模組化分層** —— 設定、核心邏輯、畫面三層分離，每個檔案只負責一件事
2. **資料來源自動降級** —— 主來源失效時自動切換備援，並記錄真實錯誤原因
3. **避免資料洩漏** —— scaler 僅以訓練區間 fit，訓練／驗證之間採 purged split
4. **誠實評估** —— 以 naive baseline 對照，並揭露預測輸出的標準差比與方向準確率
5. **一站式健檢** —— 獨立診斷分頁，可在使用前確認所有外部相依是否正常

---

## 目錄

{toc}

---

## 檔案結構

```
{tree}
```

## 程式碼統計

| 檔案 | 說明 | 行數 |
|---|---|---:|
{stats}
| **合計** | | **{n_lines}** |

---

"""

TREE = """專題資料夾/
├── app.py                    主程式
├── config.py                 全域設定
├── collect_news.py           每日新聞蒐集腳本
├── export_all.py             程式碼彙整工具（本文件產生器）
├── requirements.txt
│
├── core/                     核心功能層
│   ├── datasource.py         股價資料來源
│   ├── indicators.py         技術指標
│   ├── news.py               新聞抓取
│   ├── sentiment.py          情緒分析
│   ├── model.py              預測模型
│   ├── finance.py            金融計算
│   └── storage.py            本機存取
│
├── tabs/                     介面層（一個分頁一個檔案）
│   ├── tab0_health.py        系統健檢
│   ├── tab1_technical.py     技術分析
│   ├── tab2_dca.py           定期定額回測
│   ├── tab3_scanner.py       市場掃描
│   ├── tab4_predict.py       AI 預測
│   └── tab5_news.py          新聞情緒
│
└── data/                     執行時自動產生
    ├── news/                 每日新聞 CSV
    ├── results/              預測結果 .pkl
    └── app.log               系統紀錄"""


def anchor(text: str) -> str:
    """產生 Markdown 標題對應的錨點。"""
    out = text.lower()
    for ch in "：:，,。.／/（）()【】[]":
        out = out.replace(ch, "")
    return out.replace(" ", "-")


def main():
    sections, toc_lines, stats_lines = [], [], []
    total = 0

    for i, (rel, title, desc) in enumerate(FILES, 1):
        path = BASE / rel
        if not path.exists():
            print(f"⚠️ 找不到 {rel}，略過")
            continue

        code = path.read_text(encoding="utf-8")
        n = len(code.splitlines())
        total += n

        heading = f"{i}. {title}"
        lang = "python" if rel.endswith(".py") else ("markdown" if rel.endswith(".md") else "text")

        toc_lines.append(f"{i}. [{title}](#{anchor(heading)}) —— `{rel}`")
        stats_lines.append(f"| `{rel}` | {desc} | {n} |")

        sections.append(
            f"## {heading}\n\n"
            f"**檔案位置**：`{rel}` 　**行數**：{n}\n\n"
            f"{desc}\n\n"
            f"```{lang}\n{code.rstrip()}\n```\n\n---\n"
        )

    header = HEADER.format(
        title="ETF 損益分析與預測系統　程式碼彙整",
        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_files=len(sections),
        n_lines=total,
        toc="\n".join(toc_lines),
        tree=TREE,
        stats="\n".join(stats_lines),
    )

    md = header + "\n".join(sections)
    md_path = BASE / "專題程式碼彙整.md"
    md_path.write_text(md, encoding="utf-8")

    # 純文字版：移除 Markdown 語法
    txt_lines = []
    for line in md.splitlines():
        if line.startswith("```"):
            continue
        line = line.lstrip("#").lstrip() if line.startswith("#") else line
        txt_lines.append(line.replace("**", "").replace("`", ""))
    txt_path = BASE / "專題程式碼彙整.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("=" * 56)
    print(f"✅ 已產生 {md_path.name}")
    print(f"✅ 已產生 {txt_path.name}")
    print(f"　 共 {len(sections)} 個檔案、{total} 行程式碼")
    print("=" * 56)


if __name__ == "__main__":
    main()
