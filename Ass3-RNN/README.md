# 🎙️ Ass3 - 台語語音辨識 (Taiwanese ASR)

使用 Whisper 模型進行台語自動語音辨識（Automatic Speech Recognition）的訓練與測試。

---

## 📋 目錄

- [資料準備](#資料準備)
- [訓練流程](#訓練流程)
- [相關文件](#相關文件)
- [模型版本記錄](#模型版本記錄)

---

## 🛠️ 環境需求

```bash
# 主要套件
- Python 3.8+
- PyTorch
- Transformers
- librosa
- pandas
```

---

## 📊 資料準備

### 步驟 1：下載資料增強素材

執行 `downloadData.py` 從以下三個資料集下載資料增強用的素材：

| 資料集 | 說明 | 檔案數量 |
|--------|------|----------|
| **MS-SNSD** | 微軟背景噪音資料集 | ~900 個檔案 |
| **ESC-50** | 環境聲音分類資料集 | 100 個短暫噪音 |
| **OpenSLR RIR** | 房間脈衝響應資料集 | 50 個 RIR 檔案 |

```bash
python downloadData.py
```

### 步驟 2：準備 Kaggle 資料

將 Kaggle 競賽資料解壓縮到專案目錄，形成以下結構：

```
Ass3-RNN/
├── train/
│   └── train/
│       └── train/*.wav          # 訓練音訊檔案
└── test-random/
    └── test-random/*.wav        # 測試音訊檔案
```

### 步驟 3：音訊預處理

執行 `datapreprocess.py` 進行音訊標準化：

```bash
python datapreprocess.py
```

**預處理內容**：
- ✅ 採樣率統一到 22050 Hz
- ✅ 格式轉換為 16-bit PCM
- ✅ 生成資料夾結構與統計資訊

---

## 🚀 訓練流程

### 步驟 1：資料增強（或是直接在.ipynb執行資料增強也可以）

使用 **RawBoost** 技術進行資料增強，可大幅提升模型效能：

```bash
python data_augmentation_rawboost.py
```

📖 **詳細說明**：請參考 [RawBoost 使用指南](./RAWBOOST_USAGE_GUIDE.md)

**效果**：
- 🎯 將訓練資料量擴增至 **3-4 倍**
- 📈 顯著降低 Mean Levenshtein Distance (MDL)
- 🛡️ 增強模型對噪音的魯棒性

### 步驟 2：開始訓練

運行主要訓練筆記本：

```bash
jupyter notebook NYCU_IAlI_ML2025_RNN.ipynb
```

LoRA 微調版本沒有寫出來不能用...


**訓練選項**：
- 📝 可指定不同版本的 Whisper 模型（tiny/base/small/medium/large）
- ⚙️ 可調整訓練超參數
- 💾 自動儲存最佳模型

---

## 📚 相關文件

| 文件 | 說明 |
|------|------|
| [RawBoost 使用指南](./RAWBOOST_USAGE_GUIDE.md) | 資料增強詳細說明與參數調整 |
| [NYCU_IAlI_ML2025_RNN.ipynb](./NYCU_IAlI_ML2025_RNN.ipynb) | 基礎 Whisper 訓練流程 |

---

## 📈 模型版本記錄

# Whisper-Taiwanese model V0.5 (Tv0.5)
- 第一版:
    ```
    training_args = Seq2SeqTrainingArguments(
        # 輸出設定
        output_dir="./whisper-taiwanese-finetuned",
        
        # 訓練設定
        per_device_train_batch_size=8,      # ✅ 減小批次大小 (更穩定)
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,      # ✅ 增加梯度累積 (有效 batch=16)

        
        # 學習率設定
        learning_rate=1e-5,                 # ✅ 降低學習率 (更穩定)
        warmup_steps=300,                   # ✅ 增加 warmup (更平滑)
        
        # 訓練輪數
        num_train_epochs=5,
        
        # 評估與儲存
        eval_strategy="steps",              # 每 N 步評估一次
        eval_steps=500,                     # 每 500 步評估
        save_strategy="steps",              # 每 N 步儲存
        save_steps=500,                     # 每 500 步儲存
        save_total_limit=3,                 # 只保留最新 3 個 checkpoint
        
        # 記錄設定
        logging_steps=50,                   # ✅ 更頻繁記錄
        logging_dir="./logs",
        
        # 最佳模型
        load_best_model_at_end=True,       # 訓練結束載入最佳模型
        metric_for_best_model="wer",       # 使用 WER 作為評估指標
        greater_is_better=False,           # WER 越小越好
        
        # 硬體設定
        fp16=True,                         # 使用混合精度訓練 (加速+省記憶體)
        dataloader_num_workers=32,          # 資料載入執行緒數
        
        # 其他
        predict_with_generate=True,        # 評估時使用生成模式
        generation_max_length=225,         # 生成最大長度
        push_to_hub=False,                 # 不上傳到 HuggingFace Hub
    )
    ```
    資料為原始資料+原始資料使用老師的Data agg
    MDL = 8.13131
    - 第二版:
    ```
    training_args = Seq2SeqTrainingArguments(
        # 輸出設定
        output_dir="./whisper-taiwanese-finetuned-RawBoost",
        
        # 訓練設定
        per_device_train_batch_size=8,      # ✅ 減小批次大小 (更穩定)
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,      # ✅ 增加梯度累積 (有效 batch=16)
        
        # ✅ 梯度裁剪 (防止梯度爆炸，最重要！)
        max_grad_norm=1.0,
        
        # 學習率設定
        learning_rate=5e-6,                 # ✅ 降低學習率 (更穩定)
        warmup_steps=500,                   # ✅ 增加 warmup (更平滑)
        lr_scheduler_type="cosine",         # ✅ 使用餘弦學習率衰減
        weight_decay=0.01,                  # ✅ L2 正則化
        
        # 訓練輪數
        num_train_epochs=5,
        
        # 評估與儲存
        eval_strategy="steps",              # 每 N 步評估一次
        eval_steps=500,                     # 每 500 步評估
        save_strategy="steps",              # 每 N 步儲存
        save_steps=500,                     # 每 500 步儲存
        save_total_limit=3,                 # 只保留最新 3 個 checkpoint
        
        # 記錄設定
        logging_steps=50,                   # ✅ 更頻繁記錄
        logging_dir="./logs",
        
        # 最佳模型
        load_best_model_at_end=True,       # 訓練結束載入最佳模型
        metric_for_best_model="wer",       # 使用 WER 作為評估指標
        greater_is_better=False,           # WER 越小越好
        
        # 硬體設定
        fp16=True,                         # 使用混合精度訓練 (加速+省記憶體)
        dataloader_num_workers=32,          # 資料載入執行緒數
        
        # 其他
        predict_with_generate=True,        # 評估時使用生成模式
        generation_max_length=225,         # 生成最大長度
        push_to_hub=False,                 # 不上傳到 HuggingFace Hub
    )
    ```
    資料為原始資料+RawBoost 3和6(資料量變成3倍)
    MDL = 11.47474

 - 第三版:
    ```
        training_args = Seq2SeqTrainingArguments(
        # 輸出設定
        output_dir="./whisper-taiwanese-finetuned-RawBoost",
        
        # 訓練設定
        per_device_train_batch_size=8,      # ✅ 減小批次大小 (更穩定)
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,      # ✅ 增加梯度累積 (有效 batch=16)
        
        # ✅ 梯度裁剪 (防止梯度爆炸，最重要！)
        max_grad_norm=1.0,
        
        # 學習率設定
        learning_rate=1e-5,                 # ✅ 提高學習率 (資料多了，可以學更快)
        warmup_steps=1000,                  # ✅ 增加 warmup (步數更多需要更長 warmup)
        lr_scheduler_type="cosine",         # ✅ 使用餘弦學習率衰減
        weight_decay=0.01,                  # ✅ L2 正則化
        
        # 訓練輪數
        num_train_epochs=5,
        
        # 評估與儲存
        eval_strategy="steps",              # 每 N 步評估一次
        eval_steps=300,                     # ✅ 更頻繁評估 (每 300 步，約每半個 epoch)
        save_strategy="steps",              # 每 N 步儲存
        save_steps=300,                     # ✅ 更頻繁儲存 (每 300 步)
        save_total_limit=5,                 # ✅ 保留最新 5 個 checkpoint (訓練更長)
        
        # 記錄設定
        logging_steps=50,                   # ✅ 更頻繁記錄
        logging_dir="./logs",
        
        # 最佳模型
        load_best_model_at_end=True,       # 訓練結束載入最佳模型
        metric_for_best_model="wer",       # 使用 WER 作為評估指標
        greater_is_better=False,           # WER 越小越好
        
        # 硬體設定
        fp16=True,                         # 使用混合精度訓練 (加速+省記憶體)
        dataloader_num_workers=4,          # ✅ 資料載入執行緒數
        
        # 其他
        predict_with_generate=True,        # 評估時使用生成模式
        generation_max_length=225,         # 生成最大長度
        push_to_hub=False,                 # 不上傳到 HuggingFace Hub
    )
    ```
    資料使用老師的+RawBoost3和6，總共資料量為4倍
    MDL = 7.34343