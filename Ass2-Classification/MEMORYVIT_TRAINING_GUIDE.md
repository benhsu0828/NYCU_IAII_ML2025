# MemoryViT 模型架構與訓練指南

## 🏗️ 模型架構詳解

### 1. **整體架構**

```
📊 MemoryViT 50類角色分類系統
│
├── 🧠 基礎 ViT (Vision Transformer)
│   ├── 輸入: 224×224×3 RGB 圖片
│   ├── Patch 分割: 16×16 patches (196 patches)
│   ├── 位置編碼: Learnable position embeddings  
│   ├── Transformer Encoder: 12 層
│   │   ├── Multi-Head Attention (12 heads)
│   │   ├── MLP (3072 hidden units)
│   │   └── Layer Normalization
│   └── 特徵維度: 768
│
└── 🎯 角色分類 Adapter
    ├── 可學習記憶: 20 memories per layer × 12 layers = 240 memories
    ├── 任務特定頭: Linear(768 → 50)
    └── 輸出: 50 類角色分類概率
```

### 2. **ViT 基礎模型詳細參數**

```python
基礎 ViT 配置:
├── image_size: 224×224
├── patch_size: 16×16
├── num_patches: 196 (14×14)
├── embed_dim: 768
├── num_layers: 12
├── num_heads: 12
├── mlp_dim: 3072
├── dropout: 0.1
└── 總參數: ~86M (凍結，不訓練)
```

### 3. **Adapter 模組詳細**

```python
Adapter 配置:
├── 基於基礎 ViT 的特徵
├── 可學習記憶:
│   ├── 每層 20 個記憶 tokens
│   ├── 總共 240 個記憶 (20×12層)
│   └── 記憶維度: 768
├── 分類頭: Linear(768 → 50)
└── 可訓練參數: ~1.5M (只訓練這部分)
```

## 🔥 訓練流程

### 1. **資料準備階段**

```python
# 自動偵測資料路徑
data_path = "E:/NYCU/.../Dataset/augmented/train"  # 使用增強後的資料

# 資料分割
├── 訓練集: 70% (約 7000 張，增強後)
├── 驗證集: 15% (約 1500 張)
└── 測試集: 15% (約 1500 張)

# 資料增強 (訓練時)
├── RandomCrop(224)
├── RandomHorizontalFlip(0.5)
├── RandomRotation(15°)
├── ColorJitter
├── RandomAffine
└── 標準化 (ImageNet 統計)
```

### 2. **訓練策略**

```python
訓練配置:
├── 優化器: AdamW (weight_decay=0.05)
├── 學習率: 1e-4 (cosine decay + warmup)
├── 批次大小: 16 (GPU 記憶體允許)
├── 總輪數: 100
├── Warmup: 10 epochs
├── 損失函數: CrossEntropyLoss + Label Smoothing(0.1)
└── 早停: patience=15 (驗證準確率)
```

### 3. **記憶體和參數效率**

```
參數對比:
├── 基礎 ViT: ~86M 參數 (凍結)
├── Adapter: ~1.5M 參數 (訓練)
├── 總訓練參數: 僅 1.7% 的完整 ViT
└── 記憶體需求: ~4-6GB VRAM (batch_size=16)
```

## 🚀 如何開始訓練

### **方法 1: 快速訓練 (推薦)**

```bash
cd src
python MemoryViT_character_classifier.py
```

### **方法 2: 自訂參數訓練**

```python
# 修改 main() 函數中的參數
classifier = MemoryViTCharacterClassifier(
    num_classes=50,
    image_size=224,
    device='cuda'
)

# 準備資料 (自動使用增強後資料)
data_path = "你的資料路徑"
train_dataset, val_dataset, test_dataset = classifier.prepare_data(data_path)

# 開始訓練
classifier.train(
    batch_size=16,        # 可調整 (8, 16, 32)
    epochs=100,          # 總訓練輪數
    lr=1e-4,            # 學習率
    warmup_epochs=10    # 熱身輪數
)
```

### **方法 3: 分階段訓練**

```python
# 階段 1: 快速收斂 (高學習率)
classifier.train(batch_size=16, epochs=50, lr=2e-4)

# 階段 2: 精細調整 (低學習率)  
classifier.train(batch_size=16, epochs=50, lr=5e-5)
```

## 📊 訓練監控

### **即時監控指標**

```
每個 Epoch 顯示:
├── 訓練損失 & 準確率
├── 驗證損失 & 準確率  
├── 當前學習率
├── 最佳驗證準確率
└── 早停計數器
```

### **自動保存**

```
訓練過程自動保存:
├── best_character_model.pth (最佳模型)
├── training_log.csv (訓練記錄)
├── character_class_mapping.json (類別映射)
├── confusion_matrix.png (混淆矩陣)
└── training_curves.png (訓練曲線)
```

## 🎯 預期效果

### **性能指標**

```
預期結果:
├── 訓練準確率: >95%
├── 驗證準確率: 85-90%
├── 測試準確率: 80-85%
├── 訓練時間: 2-4 小時 (100 epochs)
└── 推理速度: ~50ms/image
```

### **優勢分析**

```
MemoryViT 優勢:
├── 🧠 記憶機制: 每層可學習記憶提升特徵表達
├── 💾 參數效率: 只訓練 1.7% 參數，避免過擬合
├── 🔄 多任務能力: 可輕鬆擴展到其他分類任務
├── ⚡ 訓練速度: 比完整 ViT 快 5-10 倍
└── 🎯 高準確率: 在小數據集上表現優異
```

## 🛠️ 故障排除

### **常見問題**

1. **GPU 記憶體不足**
   ```python
   # 降低批次大小
   classifier.train(batch_size=8)  # 或 batch_size=4
   ```

2. **訓練過慢**
   ```python
   # 減少 workers 數量
   num_workers=2  # 在 DataLoader 中
   ```

3. **準確率不收斂**
   ```python
   # 調整學習率
   classifier.train(lr=5e-5)  # 降低學習率
   ```

現在你可以直接運行訓練了！模型會自動處理所有細節。