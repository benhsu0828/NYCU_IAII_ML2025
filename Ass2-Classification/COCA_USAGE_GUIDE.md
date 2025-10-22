# 🔮 CoCa 辛普森角色分類器 - 完整指南

這是一個使用 CoCa (Contrastive Captioners) 作為特徵提取器的辛普森角色分類系統。

## 🚀 快速開始

### 1. 環境準備

首先確保你的系統已安裝必要套件：

```bash
# 安裝基本 PyTorch 套件 (Mac MPS 支援版本)
pip install torch torchvision torchaudio

# 安裝 CoCa 相關套件
pip install open-clip-torch>=2.20.0

# 安裝其他必要套件
pip install pandas numpy pillow tqdm matplotlib seaborn scikit-learn

# 檢查安裝
python src/test_coca_setup.py
```

### 2. 系統測試

運行測試腳本確保環境正確：

```bash
cd /Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification
python src/test_coca_setup.py
```

應該看到類似輸出：
```
🧪 CoCa 系統測試
==================================================
✅ PyTorch: 2.x.x
✅ OpenCLIP: 2.x.x
✅ MPS (Metal) 可用
📋 找到 X 個 CoCa 模型
🎉 所有測試通過！CoCa 系統準備就緒
```

## 🏋️ 訓練 CoCa 分類器

### 資料準備

確保你的資料結構如下：
```
Dataset/
├── train/
│   ├── abraham_grampa_simpson/
│   ├── apu_nahasapeemapetilon/
│   ├── bart_simpson/
│   ├── charles_montgomery_burns/
│   └── ... (其他角色資料夾)
├── val/
│   ├── abraham_grampa_simpson/
│   ├── apu_nahasapeemapetilon/
│   └── ... (驗證資料)
└── test/
    ├── 0.jpg
    ├── 1.jpg
    └── ... (測試圖片，純數字檔名)
```

### 開始訓練

```bash
# 訓練 CoCa 分類器
python src/CoCa_character_classifier.py

# 或使用自訂參數
python src/CoCa_character_classifier.py \
    --data-dir Dataset \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

訓練過程中會看到：
```
🔮 CoCa 辛普森角色分類器訓練
🖥️ 使用設備: mps
🤖 載入 CoCa 模型: coca_ViT-B-32
📐 特徵維度: 512
📝 類別數: 20
⭐ 訓練開始！

Epoch 1/50:
訓練: 100%|██████████| 156/156 [02:30<00:00, 1.04batch/s]
驗證: 100%|██████████| 39/39 [00:15<00:00, 2.56batch/s]
📊 Epoch 1 - 訓練損失: 2.845, 訓練準確率: 15.2%, 驗證損失: 2.234, 驗證準確率: 28.5%
```

### 訓練完成

訓練完成後會產生：
- `coca_character_classifier_YYYYMMDD_HHMM.pth` - 訓練好的模型
- `training_plots_coca_YYYYMMDD_HHMM.png` - 訓練過程圖表

## 🔍 使用 CoCa 模型推理

### 基本推理

```bash
# 使用互動模式（推薦新手）
python src/coca_inference.py

# 命令列模式
python src/coca_inference.py \
    --model coca_character_classifier_20241205_1234.pth \
    --test-dir Dataset/test \
    --output predictions.csv
```

### 推理結果

推理完成後會產生 `predictions.csv`：
```csv
id,character
0,homer_simpson
1,marge_simpson
2,bart_simpson
3,lisa_simpson
...
```

## 🔧 高級使用

### 1. 模型比較

你可以同時訓練多個版本並比較效果：

```bash
# 訓練不同的 CoCa 模型
python src/CoCa_character_classifier.py --coca-model coca_ViT-B-32
python src/CoCa_character_classifier.py --coca-model coca_ViT-L-14

# 比較推理結果
python src/coca_inference.py -m model1.pth -o predictions1.csv
python src/coca_inference.py -m model2.pth -o predictions2.csv
```

### 2. 自訂訓練參數

```python
# 編輯 CoCa_character_classifier.py 中的參數
COCA_MODEL_NAME = "coca_ViT-L-14"  # 使用更大的模型
BATCH_SIZE = 16  # 調整批次大小
LEARNING_RATE = 0.0005  # 調整學習率
NUM_EPOCHS = 100  # 增加訓練輪數
```

### 3. 使用已有的 EfficientNet 推理工具

如果你還想使用原來的 EfficientNet 模型：

```bash
# 使用原有的 Mac 推理工具
python src/mac_inference.py
```

## 📊 效能監控

### 訓練過程監控

訓練過程中會自動：
1. 📈 即時顯示損失和準確率
2. 📊 每 5 個 epoch 生成圖表
3. 💾 自動保存最佳模型
4. ⏰ 早停機制避免過擬合

### 推理效能

在 Apple Silicon Mac 上：
- **M1/M2 Mac**: ~50-100 張圖片/秒
- **Intel Mac**: ~20-50 張圖片/秒
- **記憶體使用**: 約 2-4GB

## 🔍 故障排除

### 常見問題

#### 1. MPS 設備錯誤
```bash
# 如果 MPS 有問題，強制使用 CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
python src/CoCa_character_classifier.py --device cpu
```

#### 2. 記憶體不足
```python
# 減少批次大小
BATCH_SIZE = 16  # 或更小
```

#### 3. CoCa 模型載入失敗
```bash
# 更新 open-clip-torch
pip install --upgrade open-clip-torch

# 檢查網路連線（首次下載模型需要網路）
```

#### 4. 找不到 Dataset 目錄
```bash
# 確保在正確目錄
cd /Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification
ls Dataset/  # 應該看到 train, val, test 目錄
```

### 檢查清單

使用前請確認：
- [ ] ✅ PyTorch 已安裝且支援 MPS
- [ ] ✅ open-clip-torch >= 2.20.0 已安裝
- [ ] ✅ Dataset 目錄結構正確
- [ ] ✅ 測試腳本通過
- [ ] ✅ 有足夠的磁碟空間（約 5GB）

## 🎯 與 EfficientNet 的比較

| 特徵 | EfficientNet | CoCa |
|------|-------------|------|
| **特徵提取能力** | 強 | 更強（多模態訓練） |
| **訓練資料需求** | 中等 | 較少（遷移學習效果好） |
| **推理速度** | 快 | 中等 |
| **模型大小** | 小 | 較大 |
| **準確率潜力** | 高 | 更高 |

## 💡 最佳實踐

1. **首次使用**: 先運行測試腳本確保環境正確
2. **資料準備**: 確保訓練/驗證資料平衡
3. **訓練監控**: 關注驗證準確率，避免過擬合
4. **模型選擇**: 小資料集用 `coca_ViT-B-32`，大資料集用 `coca_ViT-L-14`
5. **推理優化**: 批量推理比單張推理快很多

## 📞 需要幫助？

如果遇到問題：
1. 先運行 `python src/test_coca_setup.py` 檢查環境
2. 檢查錯誤訊息和故障排除章節
3. 確認 Dataset 目錄結構正確
4. 檢查可用記憶體和磁碟空間

祝你使用愉快！🎉
