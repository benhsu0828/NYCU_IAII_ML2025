# 房地產價格預測系統使用指南

## 功能概述

本系統提供房地產價格預測的完整流程，包括：
- 資料預處理與特徵工程
- 多種機器學習模型訓練
- 模型評估與比較
- 測試集預測
- 特徵重要性分析

## 目錄結構

```
Ass1-Regression/
├── Dataset/
│   ├── raw/                 # 原始資料
│   └── processed/           # 預處理後資料
├── models/                  # 訓練好的模型
├── results/                 # 預測結果與分析
├── src/
│   ├── main.py             # 主程式
│   ├── model.py            # 模型定義
│   ├── data_preprocess.py  # 資料預處理
│   ├── system_test.py      # 系統測試
│   └── test.py             # 原有測試檔
└── requirements.txt        # 依賴套件
```

## 安裝與設定

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 資料預處理（如果需要）

如果 `Dataset/processed/` 目錄為空，需要先執行資料預處理：

```python
from src.data_preprocess import preprocess_data
preprocess_data()
```

### 3. 系統測試（可選）

執行系統測試確認環境：

```bash
cd src
python system_test.py
```

## 使用方式

### 執行主程式

```bash
cd src
python main.py
```

### 執行模式選擇

程式會提示選擇執行模式：

```
=== 請選擇執行模式 ===
1. 訓練模式 (train) - 只訓練模型
2. 測試模式 (test) - 使用已訓練模型進行預測
3. 訓練並測試 (both) - 先訓練再測試
4. 快速測試 (quick) - 快速基準測試
5. 退出 (exit)
```

#### 1. 訓練模式 (train)
- 載入預處理資料
- 訓練多種機器學習模型
- 評估模型性能
- 儲存最佳模型
- 輸出特徵重要性分析

#### 2. 測試模式 (test)
- 載入已訓練的模型
- 對測試集進行預測
- 儲存預測結果

#### 3. 訓練並測試 (both)
- 結合訓練和測試流程
- 先訓練模型，再立即進行測試

#### 4. 快速測試 (quick)
- 使用少量資料快速測試流程
- 適合驗證系統是否正常運作

## 模型說明

### 支援的模型類型

1. **樹模型**
   - RandomForest (隨機森林)
   - XGBoost
   - LightGBM
   - CatBoost
   - GradientBoosting

2. **線性模型**
   - LinearRegression
   - Ridge
   - Lasso
   - ElasticNet

3. **深度學習模型**
   - Neural Network (Keras/TensorFlow)

### 模型訓練流程

1. **第一輪訓練**：基礎樹模型
2. **第二輪訓練**：優化參數的樹模型
3. **線性模型**：可選擇是否訓練

## 輸出檔案

### 1. 模型檔案 (`models/`)
- `{model_name}.joblib`：訓練好的模型

### 2. 預測結果 (`results/`)
- `predictions_{model_name}.csv`：測試集預測結果
- `feature_importance_{model_name}.csv`：特徵重要性分析

### 3. 控制台輸出
- 訓練過程監控
- 模型性能指標 (RMSE, R², MAE)
- 特徵重要性排序

## 資料預處理功能

### 自動處理項目

1. **缺失值處理**
2. **類別編碼**
   - One-hot encoding
   - Label encoding
3. **數值特徵處理**
   - 正則表達式提取數字
   - 日期欄位編碼
4. **維度對齊**
   - 確保 train/valid/test 特徵一致

### 特徵工程

1. **日期特徵**
   - 年、月、日分離
   - 天數計算
   - 循環編碼

2. **文字特徵**
   - 正則表達式處理
   - 類別統一化

## 疑難排解

### 常見問題

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **資料檔案找不到**
   - 確認 `Dataset/processed/` 目錄有資料檔案
   - 執行資料預處理

3. **模型載入失敗**
   - 確認 `models/` 目錄有模型檔案
   - 先執行訓練模式

4. **記憶體不足**
   - 使用快速測試模式
   - 減少資料量或模型複雜度

### 效能優化

1. **使用快速測試模式**進行初步驗證
2. **選擇合適的模型**（LightGBM 通常最快）
3. **調整資料量**用於測試

## 擴展功能

### 新增模型

在 `model.py` 中的 `RegressionModels` 類別新增模型：

```python
def get_custom_models(self):
    return {
        'CustomModel': YourModelClass(**params)
    }
```

### 自訂特徵工程

在 `data_preprocess.py` 中修改或新增處理函數。

### 評估指標

可在 `model.py` 中新增其他評估指標。

## 技術細節

- **程式語言**：Python 3.8+
- **主要套件**：pandas, scikit-learn, xgboost, lightgbm, catboost
- **深度學習**：TensorFlow/Keras
- **資料格式**：CSV, Excel
- **模型儲存**：joblib

## 授權與聲明

本系統為教學與研究用途，請確保資料使用符合相關法規。
