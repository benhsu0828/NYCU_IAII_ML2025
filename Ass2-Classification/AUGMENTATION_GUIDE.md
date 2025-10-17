# 辛普森角色資料增強指南

## 📁 檔案說明

### 新增的資料增強檔案：

1. **`data_augmentation.py`** - 完整的資料增強腳本
   - 結合 `data_aggV1.py` 和 `data_aggV2.py` 的增強方法
   - 支援批次處理整個資料集
   - 包含多種增強策略

2. **`quick_augment.py`** - 快速啟動腳本
   - 使用預設參數
   - 簡化操作流程

3. **`generate_backgrounds.py`** - 背景圖片生成器
   - 創建純色、漸層、紋理背景
   - 用於背景合成增強

## 🎯 資料增強策略

### 方法一：變換增強 (來自 data_aggV1.py)
- **幾何變換**: 水平翻轉、旋轉、透視變換、仿射變換
- **顏色變換**: 亮度、對比度、飽和度、色調調整
- **噪聲增強**: 高斯噪聲、散斑噪聲、泊松噪聲、椒鹽噪聲
- **模糊效果**: 高斯模糊

### 方法二：背景合成 (來自 data_aggV2.py)
- 移除原始背景（黑色/白色區域）
- 替換為自定義背景圖片
- 添加隨機邊距

## 🚀 使用方法

### 快速開始（推薦）

```bash
# 1. 進入項目目錄
cd E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\src

# 2. 激活環境
conda activate vit_env

# 3. 生成背景圖片（可選）
python generate_backgrounds.py

# 4. 快速增強
python quick_augment.py
```

### 自定義參數

```bash
# 完整參數控制
python data_augmentation.py \
    --input_dir "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/train" \
    --output_dir "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/augmented/train" \
    --backgrounds_dir "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/backgrounds" \
    --augment_per_image 5 \
    --no_background  # 關閉背景合成
```

### 參數說明

- `--input_dir`: 輸入資料夾（包含類別子資料夾）
- `--output_dir`: 輸出資料夾
- `--backgrounds_dir`: 背景圖片資料夾
- `--augment_per_image`: 每張圖片生成的增強版本數量
- `--no_background`: 關閉背景合成增強
- `--no_transform`: 關閉變換增強

## 📊 預期結果

### 原始資料結構：
```
preprocessed/train/
├── abraham_grampa_simpson/  (例：50張)
├── agnes_skinner/          (例：45張)  
└── ...                     (49個類別)
```

### 增強後資料結構：
```
augmented/train/
├── abraham_grampa_simpson/  (例：200張 = 50原始 + 150增強)
├── agnes_skinner/          (例：180張 = 45原始 + 135增強)
└── ...                     (49個類別)
```

### 檔案命名規則：
- 原始圖片：保持原名
- 增強圖片：`原名_aug_方法_序號.jpg`
  - 例：`homer_01_aug_trans_00.jpg`（變換增強）
  - 例：`bart_02_aug_bg_trans_01.jpg`（背景+變換增強）

## 🔧 故障排除

### 1. 導入錯誤
```bash
# 確保環境正確
conda activate vit_env

# 安裝缺少的套件
pip install torch torchvision Pillow numpy tqdm
```

### 2. 記憶體不足
- 減少 `--augment_per_image` 參數
- 分批處理類別

### 3. 背景圖片問題
- 如果沒有背景圖片，使用 `--no_background` 參數
- 運行 `generate_backgrounds.py` 創建背景

### 4. 路徑問題
- 確保使用正確的絕對路徑
- 檢查資料夾是否存在

## 📈 增強效果

### 資料量倍增
- 原始：~2500張圖片（50類 × 50張平均）
- 增強後：~10000張圖片（4倍增長）

### 多樣性提升
- **幾何多樣性**: 不同角度、變形
- **顏色多樣性**: 不同光照、對比度
- **背景多樣性**: 不同場景背景
- **噪聲魯棒性**: 抗噪聲能力

## 🎯 下一步

1. **檢查增強結果**：
   ```bash
   # 查看增強後的類別數量
   ls "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/augmented/train"
   ```

2. **更新訓練腳本**：
   修改 `MemoryViT_character_classifier.py` 中的資料路徑：
   ```python
   train_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/augmented/train"
   ```

3. **開始訓練**：
   ```bash
   python MemoryViT_character_classifier.py
   ```

## 💡 最佳實踐

1. **漸進式增強**: 先用少量增強測試，再逐步增加
2. **質量檢查**: 隨機查看增強結果，確保質量
3. **平衡策略**: 根據類別數據量調整增強強度
4. **儲存空間**: 確保有足夠的硬碟空間（增強後約4倍大小）
