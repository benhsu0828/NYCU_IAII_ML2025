# 🎯 RawBoost 資料增強使用指南

## 📋 目錄
- [簡介](#簡介)
- [演算法說明](#演算法說明)
- [使用方式](#使用方式)
- [參數調整](#參數調整)
- [建議設定](#建議設定)

---

## 簡介

RawBoost 是專為語音防偽檢測（Anti-Spoofing）設計的資料增強方法，但也適用於一般語音辨識任務。它通過在音訊波形上添加各種類型的噪音來提升模型的泛化能力。

**特點：**
- ✅ 直接在波形上操作，保留原始特徵
- ✅ 模擬真實環境噪音
- ✅ 提升模型對雜訊的魯棒性

---

## 演算法說明

### 🔹 Algorithm 1: LnL (Linear and Non-linear Convolutive Noise)
**線性與非線性卷積噪音**

- **用途**: 模擬錄音設備的頻率響應和失真
- **效果**: 改變音訊的頻率特性
- **適用場景**: 
  - 不同麥克風錄音
  - 電話語音
  - 壓縮音訊
- **參數**:
  - `N_f`: 非線性濾波器數量 (預設: 5)
  - `nBands`: 頻帶數量 (預設: 5)
  - `minF/maxF`: 頻率範圍 (20-8000 Hz)

---

### 🔹 Algorithm 2: ISD (Impulsive Signal Dependent Noise)
**脈衝訊號相依噪音**

- **用途**: 添加與訊號相關的脈衝噪音
- **效果**: 模擬突發性噪音（如按鍵聲、碰撞聲）
- **適用場景**:
  - 環境突發噪音
  - 設備故障
  - 數位干擾
- **參數**:
  - `P`: 噪音比例 (0-100%, 預設: 10)
  - `g_sd`: 噪音強度 (預設: 2)

---

### 🔹 Algorithm 3: SSI (Stationary Signal Independent Noise)
**平穩訊號獨立噪音**

- **用途**: 添加穩定的背景噪音
- **效果**: 模擬真實環境的背景噪音
- **適用場景**: ✅ **最推薦用於語音辨識！**
  - 街道噪音
  - 辦公室環境
  - 自然背景音
- **參數**:
  - `SNRmin/SNRmax`: 訊噪比範圍 (10-40 dB)

---

### 🔹 組合演算法

| 演算法編號 | 組合方式 | 說明 | 推薦用於語音辨識 |
|-----------|---------|------|----------------|
| **4** | 1+2+3 (串聯) | 所有增強 | ⚠️ 可能過度增強 |
| **5** | 1+2 (串聯) | LnL + ISD | ⚠️ 較激進 |
| **6** | 1+3 (串聯) | LnL + SSI | ✅ **推薦** |
| **7** | 2+3 (串聯) | ISD + SSI | ⚠️ 適中 |

---

## 使用方式

### 方法 1: 使用獨立腳本

```bash
python data_augmentation_rawboost.py \
    --input_dir "./train/preprocessed" \
    --output_dir "./train/rawboost_augmented" \
    --csv_input "./train/trainAgg-toneless.csv" \
    --csv_output "./train/trainAgg-toneless-rawboost.csv" \
    --algo_types 3 6 \
    --num_aug 1 \
    --sr 16000
```

**參數說明：**
- `--input_dir`: 原始音訊目錄
- `--output_dir`: 輸出目錄
- `--csv_input`: 原始 CSV 檔案
- `--csv_output`: 輸出 CSV 檔案
- `--algo_types`: 演算法編號（可多選，如 `3 6`）
- `--num_aug`: 每種演算法生成幾個版本
- `--sr`: 採樣率

---

### 方法 2: 在 Jupyter Notebook 中使用

參考 Notebook 中的 **RawBoost 資料增強** cell，直接執行即可。

**優點：**
- ✅ 即時查看進度
- ✅ 方便調整參數
- ✅ 整合在訓練流程中

---

## 參數調整

### 🔧 RawBoost 參數

```python
class RawBoostConfig:
    def __init__(self):
        # Algorithm 1: LnL
        self.N_f = 5              # 濾波器數量 (↑ 更複雜)
        self.nBands = 5           # 頻帶數量 (↑ 更細緻)
        self.minF = 20            # 最低頻率 (Hz)
        self.maxF = 8000          # 最高頻率 (Hz)
        
        # Algorithm 2: ISD
        self.P = 10               # 噪音比例 (%) (↑ 更多噪音)
        self.g_sd = 2             # 噪音強度 (↑ 更強)
        
        # Algorithm 3: SSI
        self.SNRmin = 10          # 最低訊噪比 (dB) (↓ 更吵)
        self.SNRmax = 40          # 最高訊噪比 (dB) (↑ 更乾淨)
```

### 調整建議

**如果想要更強的增強效果：**
```python
self.SNRmin = 5    # 降低最低訊噪比
self.SNRmax = 30   # 降低最高訊噪比
self.P = 15        # 增加脈衝噪音比例
```

**如果想要更溫和的增強：**
```python
self.SNRmin = 15   # 提高最低訊噪比
self.SNRmax = 40   # 保持最高訊噪比
self.P = 5         # 減少脈衝噪音比例
```

---

## 建議設定

### ✅ 台語語音辨識推薦配置

```python
# 推薦使用的演算法
algo_types = [3, 6]

# 理由:
# - Algorithm 3 (SSI): 添加自然背景噪音
# - Algorithm 6 (LnL+SSI): 同時模擬設備特性和背景噪音
```

### 📊 資料量建議

| 原始資料量 | 建議增強策略 | 說明 |
|-----------|------------|------|
| < 500 筆 | `algo_types=[3, 6]`, `num_aug=2` | 大幅擴增 |
| 500-1000 筆 | `algo_types=[3, 6]`, `num_aug=1` | 適度擴增 ✅ **你的情況** |
| > 1000 筆 | `algo_types=[3]`, `num_aug=1` | 輕度擴增 |

### ⚖️ 增強與原始資料比例

**建議比例：** 增強資料 ≤ 2 倍原始資料

```
原始資料: 1000 筆
增強資料: 2000 筆 (每筆用 2 種演算法)
總計: 3000 筆
```

**過度增強的風險：**
- ❌ 模型學到噪音特徵而非語音特徵
- ❌ 驗證集表現良好但測試集差
- ❌ 泛化能力下降

---

## 整合到訓練流程

### 1️⃣ 資料準備

```python
# 原始資料預處理
python datapreprocess.py  # Audiomentations 增強

# RawBoost 增強
python data_augmentation_rawboost.py --algo_types 3 6
```

### 2️⃣ 更新訓練資料路徑

```python
# 使用 RawBoost 增強後的資料
train_dir = "./train/rawboost_augmented"
train_csv = "./train/trainAgg-toneless-rawboost.csv"
```

### 3️⃣ 訓練模型

使用更新後的 CSV 和音訊目錄進行訓練。

---

## 疑難排解

### ❓ 問題：增強後的音訊太吵

**解決方案：**
```python
# 提高訊噪比範圍
self.SNRmin = 20
self.SNRmax = 40
```

### ❓ 問題：增強效果不明顯

**解決方案：**
```python
# 降低訊噪比範圍
self.SNRmin = 5
self.SNRmax = 30

# 或使用組合演算法
algo_types = [4, 6]  # 使用 1+2+3 和 1+3
```

### ❓ 問題：處理速度太慢

**解決方案：**
- 減少 `num_aug` 數量
- 只使用單一演算法 (如 `algo_types=[3]`)
- 使用多核心處理（需修改程式碼）

---

## 參考資料

- [RawBoost 原始論文](https://arxiv.org/abs/2111.04433)
- [ASVspoof 2021](https://www.asvspoof.org/)

---

## 📝 總結

### ✅ DO (建議做的)
- ✅ 使用 Algorithm 3 或 6
- ✅ 適度增強（1-2倍原始資料）
- ✅ 監控驗證集表現
- ✅ 與其他增強方法（Audiomentations）結合使用

### ❌ DON'T (不建議做的)
- ❌ 過度增強（>3倍原始資料）
- ❌ 使用過強的噪音（SNR < 5）
- ❌ 只依賴單一增強方法
- ❌ 不驗證增強後的音訊品質

---

**祝訓練順利！🚀**
