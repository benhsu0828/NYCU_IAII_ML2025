# 作業連結
https://www.kaggle.com/competitions/nycu-iaii-ml-2025-regression/overview

簡介
---
這個專案實作了一組房地產價格迴歸模型（樹模型、線性模型、深度學習模型與 Stacking 集成）。主要程式與邏輯都放在 `src/` 中，並支援：

- 資料前處理 pipeline（`data_preprocess.py` / `data_preprocessV2.py`）
- 多種模型訓練（XGBoost、LightGBM、CatBoost、RandomForest、線性模型）
- 深度學習（TensorFlow / Keras）與可序列化的 DNN wrapper（便於 `joblib` 儲存）
- Stacking / Voting 集成（DNN + 樹模型）
- 模型儲存與測試流程（`models/` 與 `results/`）

目錄結構（重點）
---
```
Ass1-Regression/
│
├── Dataset/ or data/         # 原始與已處理資料（raw/ processed/）
├── src/                      # 程式碼（主要開發位置）
│   ├── data_preprocess.py    # 資料前處理函式
│   ├── data_preprocessV2.py  # 進階或備用前處理
│   ├── model.py              # 模型定義、包裝器、訓練流程
│   ├── main.py               # 互動式 CLI 主程式（訓練與測試入口）
│   ├── gpu_diagnostic.py     # GPU / 系統檢查工具
│   └── ...                   # 測試與輔助腳本
│
├── models/                   # 訓練後儲存模型（.joblib、_keras/、_scaler.joblib）
├── results/                  # 儲存預測結果、特徵重要性、報表
├── requirements.txt          # 建議儲存的 Python 套件清單
├── environment.yml           # （可選）conda environment 匯出檔
└── README.md                 # 專案說明（本檔案）
```
快速開始（Step-by-step）
---
以下示範在 Windows + PowerShell 下的常用流程。請先確定你位於專案根目錄（含 `src/`、`models/` 等）。

1) 建立與啟動 Python 環境（建議使用 conda）

```powershell
# 建立 conda 環境（範例使用 Python 3.8）
conda create -n NYCUML python=3.8 -y

# 啟動環境
conda activate NYCUML
```

2) 安裝套件（範例：用 pip 或 conda）

```powershell
pip install -r requirements.txt
# 若尚未有 requirements.txt，可先手動安裝（範例）
pip install numpy pandas scikit-learn xgboost lightgbm catboost joblib tensorflow matplotlib seaborn
```

3) 如何匯出 / 儲存當前環境的依賴（產生 requirements）

- 使用 pip（簡單、可複製）

```powershell
pip freeze > requirements.txt
```

- 使用 conda（若使用 conda 安裝大量二進位套件或 CUDA 相關套件，推薦匯出 environment.yml）

```powershell
conda env export --name NYCUML > environment.yml
```

說明：`pip freeze` 會列出目前 Python 環境中所有安裝的套件；`conda env export` 則會包含 conda 與 pip 安裝的套件，以及平台資訊，較適合完全重現環境。

4) 準備資料

把原始資料放到 `Dataset/raw/`，執行前處理腳本以產生 `Dataset/processed/`。主程式會嘗試載入已處理資料：

```powershell
cd src
python data_preproccess.py
```

5) 訓練模型（互動式）

執行 `main.py`，依選單選擇訓練模式：

- Tree models（XGBoost / LightGBM / CatBoost）
- Linear models（Ridge / LinearRegression / Lasso）
- Deep Learning（使用 TensorFlow / Keras）
- Stacking（DNN + Trees，或使用 Ridge 代替 DNN）

範例：快速訓練（只訓練樹模型）

```powershell
cd src
python main.py
# 透過輸入數字選項，選擇 train -> tree
```

6) 測試與產出預測

測試流程會尋找 `models/` 裡的模型（`.joblib` 或 Keras 資料夾 `_keras` 與對應 `_scaler.joblib`），要自行輸入想要測試的模型，就會'將預測結果輸出到 `results/predictions_<model>.csv`。

7) 模型儲存格式

- 傳統 scikit-learn 模型：`models/<model_name>.joblib`
- 深度學習模型（Keras）：`models/<name>_keras/`（Keras 原生儲存）+ `models/<name>_scaler.joblib` + `models/<name>_info.txt`
- Stacking：會儲存為 `.joblib`（現在已相容，因為 DNN 包裝器在模組頂層）
