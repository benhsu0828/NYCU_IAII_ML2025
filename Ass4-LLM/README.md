# Ass4-LLM: 台語語言模型微調專案

本專案使用兩階段訓練策略（CPT + SFT）對台語 LLM 進行微調，以提升其在台語問答任務上的表現。

## 📋 目錄結構

```
Ass4-LLM/
├── src/
│   └── Ass4_LLM_CPT+SFT.ipynb    # 主要訓練腳本
├── data/
│   ├── IMA-Taiwan/                # 台語預訓練語料庫
│   ├── AI_conv.csv                # 監督式微調問答資料
│   └── 1001-question-v3.csv       # 測試資料
├── model/
│   ├── cpt_model/                 # CPT 階段訓練後的模型
│   └── final_model/               # 最終微調完成的模型
└── README.md
```

## 🎯 專案目標

使用 **Bohanlu/Taigi-Llama-2-13B** 作為基礎模型，通過持續預訓練（CPT）和監督式微調（SFT）兩階段策略，提升模型在台語閱讀理解問答任務上的表現。

## 🔧 環境設置

### 必要套件安裝

pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install pandas datasets wandb scipy


```bash
conda create -n Ass4 python=3.11 -y
conda activate Ass4
pip install -r requirements.txt

# 1. 安裝 PyTorch nightly（支援 RTX 5050）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 2. 驗證 CUDA 是否可用
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"無\"}')"
# 3. 安裝 Transformers 和資料處理套件
pip install transformers datasets pandas
# 4. 安裝 Weights & Biases（實驗追蹤）
pip install wandb
# 5. 安裝訓練相關套件
pip install accelerate peft trl
# 6. 安裝 Unsloth（加速訓練）
pip install unsloth
# 7. 安裝量化和優化套件
pip install bitsandbytes
# 8. 安裝 xformers（可選，加速 attention 計算）
pip install xformers
```

### 硬體需求

- GPU: 建議使用至少 16GB VRAM 的 GPU（如 T4、V100、A100）
- 記憶體: 建議至少 32GB RAM
- 儲存空間: 至少 50GB 可用空間

## 📚 訓練流程

### 階段 1: CPT（持續預訓練）

**目的**: 讓模型學習台語領域知識

#### 資料處理

從 `IMA-Taiwan` 資料集中讀取 JSON 格式的台語文本，進行以下預處理：

1. 移除網址連結（http/https/www 開頭）
2. 移除段落編號
3. 統一標點符號格式
4. 文本長度控制（100-2048 字元）
5. 按文章標題合併段落
6. 去除重複文本

#### CPT 訓練配置

| 參數 | 數值 | 說明 |
|------|------|------|
| **LoRA Rank (r)** | 64 | 較大的 rank 學習更多知識 |
| **LoRA Alpha** | 128 | 對應調整的縮放因子 |
| **Learning Rate** | 2e-4 | 較高學習率加速領域適應 |
| **Max Steps** | 1000 | 充分學習台語語料 |
| **Batch Size** | 4 | 每個設備的批次大小 |
| **Gradient Accumulation** | 4 | 有效批次大小為 16 |
| **Max Seq Length** | 2048 | 最大序列長度 |
| **LoRA Dropout** | 0.05 | 較小的 dropout 保留更多資訊 |

#### 訓練目標模組

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### 階段 2: SFT（監督式微調）

**目的**: 針對問答任務進行精準微調

#### 資料格式

使用 `AI_conv.csv` 中的結構化問答資料：

```
根據前文內容回答問題
前文：{文章內容}
問題：{問題內容}
根據問題，從以下四個選項選出正確的選項編號(1-4)
選項1：{選項內容}
選項2：{選項內容}
選項3：{選項內容}
選項4：{選項內容}
答案：{正確答案編號}
```

#### SFT 訓練配置

| 參數 | 數值 | 說明 |
|------|------|------|
| **LoRA Rank (r)** | 16 | 較小的 rank 避免過擬合 |
| **LoRA Alpha** | 32 | 對應的縮放因子 |
| **Learning Rate** | 1e-5 | 較小學習率微調參數 |
| **Max Steps** | 100 | 適量步驟避免過擬合 |
| **Batch Size** | 4 | 每個設備的批次大小 |
| **Gradient Accumulation** | 4 | 有效批次大小為 16 |
| **LoRA Dropout** | 0.1 | 較大的 dropout 提升泛化能力 |

#### 重要步驟

在 SFT 階段前，必須先將 CPT 階段的 LoRA 權重合併到基礎模型：

```python
model = model.merge_and_unload()
```

## 🚀 使用方式

### 1. 掛載 Google Drive（Colab 環境）

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. 設定 Wandb（可選）

```
wandb sync wandb/offline-run-*
```

### 3. 執行訓練

按照 Notebook 中的順序依次執行各個 Cell：

1. 資料預處理與載入
2. CPT 階段訓練
3. SFT 階段訓練

### 4. 推理預測

```python
# 載入最終模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 設定推理模式
FastLanguageModel.for_inference(model)

# 批次預測
batch_size = 4  # 根據 GPU 記憶體調整
```

#### 答案提取策略

使用正則表達式提取預測結果中的數字（1-4）：

```python
import re
match = re.search(r'^[1-4]', predicted_text)
clean_answer = match.group() if match else "1"
```

## 📊 實驗追蹤

使用 **Weights & Biases** 追蹤訓練過程：

- **CPT Run**: `Taigi-CPT-V2`
- **SFT Run**: `Taigi-SFT-V2`

追蹤指標包括：
- Loss
- Learning Rate
- Gradient Norm
- Training Steps

## 💡 技術亮點

### 1. 兩階段訓練策略
- **CPT**: 先讓模型學習台語領域知識
- **SFT**: 再針對特定任務微調

### 2. LoRA 參數配置策略
- CPT 使用較大的 rank (64) 學習更多知識
- SFT 使用較小的 rank (16) 避免過擬合

### 3. 資料預處理
- 智能文本清洗（移除網址、統一標點）
- 文章合併與分段策略
- 長度控制與去重

### 4. 記憶體優化
- 使用 4-bit 量化載入模型
- Gradient Checkpointing 節省記憶體
- 批次推理提升效率

## 📈 輸出結果

預測結果將儲存為 CSV 格式：

| ID | Answer |
|----|--------|
| 1  | 2      |
| 2  | 1      |
| 3  | 4      |
| ... | ...   |

輸出檔案: `./data/output-CPT&SFT_V2.csv`

## 🔍 關鍵參數說明

### CPT vs SFT 參數對比

| 參數 | CPT | SFT | 原因 |
|------|-----|-----|------|
| LoRA Rank | 64 | 16 | CPT 需要學更多，SFT 避免過擬合 |
| Learning Rate | 2e-4 | 1e-5 | CPT 需要較大調整，SFT 微調即可 |
| Max Steps | 1000 | 100 | CPT 學習領域知識需更多步驟 |
| Dropout | 0.05 | 0.1 | SFT 需要更好的泛化能力 |

## ⚠️ 注意事項

1. **模型合併**: SFT 階段前必須執行 `merge_and_unload()`
2. **記憶體管理**: 推理前執行 `gc.collect()` 和 `torch.cuda.empty_cache()`
3. **Padding 設定**: 推理時設定 `tokenizer.padding_side = 'left'`
4. **批次大小**: 根據 GPU 記憶體調整 `batch_size`
5. **隨機種子**: 使用固定種子 (9527) 確保可重現性
