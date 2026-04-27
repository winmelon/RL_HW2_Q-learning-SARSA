# Q-Learning vs SARSA — Cliff Walking

強化學習作業：比較 Q-Learning 與 SARSA 演算法在 Cliff Walking 環境的學習行為。

## 環境需求

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) 套件管理工具

## 快速開始

```bash
# 建立虛擬環境並安裝依賴
uv venv
uv sync

# 執行實驗（Windows）
.venv\Scripts\python main.py

# 執行實驗（Linux/macOS）
.venv/bin/python main.py
```

## 專案結構

```
RLHW2/
├── environment.py    # Cliff Walking 環境
├── algorithms.py     # Q-Learning & SARSA 實作
├── training.py       # 訓練迴圈
├── visualization.py  # 視覺化模組
├── main.py           # 主程式入口
├── pyproject.toml    # uv 專案設定
├── figures/          # 輸出圖片（自動建立）
└── report.md         # 實驗報告（自動生成）
```

## 超參數設定

| 參數 | 數值 |
|------|------|
| 學習率 α | 0.1 |
| 折扣因子 γ | 0.9 |
| 探索率 ε | 0.1 |
| 訓練回合數 | 500 |
| 隨機種子 | 42 |
