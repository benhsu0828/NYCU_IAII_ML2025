# 🎨 第一層權重與特徵圖可視化工具

## 📋 功能介紹

這個工具可以幫你：
1. **可視化模型第一層的權重** - 看到模型學到的基礎特徵檢測器
2. **分析圖片的特徵圖** - 看到圖片經過第一層後產生的反應
3. **創建注意力熱力圖** - 看到模型對圖片不同區域的關注程度
4. **分析通道響應強度** - 量化各個特徵通道的激活程度

## 🚀 快速開始

### 基本使用
```bash
python model_first_layer_visualizer.py
```

### 程式會自動：
1. 尋找可用的 `.pth` 模型檔案
2. 讓你選擇要分析的模型
3. 尋找範例圖片
4. 讓你選擇要分析的圖片
5. 生成所有分析結果

## 📊 輸出結果

### 1. 第一層權重可視化
- **檔案**: `{model_name}_first_layer_weights.png`
- **內容**: 顯示模型第一層的 filters/kernels
- **說明**: 這些是模型學到的基礎特徵檢測器（邊緣、紋理等）

### 2. 特徵圖可視化
- **檔案**: `{model_name}_feature_maps.png`
- **內容**: 顯示圖片經過第一層後各通道的響應
- **說明**: 不同通道對圖片中不同特徵的反應強度

### 3. 注意力熱力圖
- **檔案**: `{model_name}_attention_heatmap.png`
- **內容**: 
  - 原始圖片
  - 平均注意力熱力圖（所有通道平均）
  - 最大響應通道的熱力圖
- **說明**: 紅色區域表示模型關注度高的地方

### 4. 通道響應分析
- **檔案**: 
  - `{model_name}_channel_analysis.png` (圖表)
  - `{model_name}_channel_analysis.csv` (數據)
- **內容**: 各通道的統計分析（最大值、平均值、總和、正值比例）

## 🔍 分析解讀

### 權重可視化的意義
- **邊緣檢測器**: 看起來像線條的 filter
- **斑點檢測器**: 看起來像圓形的 filter  
- **紋理檢測器**: 複雜模式的 filter
- **顏色檢測器**: 不同顏色響應的 filter

### 特徵圖的意義
- **亮的區域**: 該 filter 對圖片該區域有強烈反應
- **暗的區域**: 該 filter 對圖片該區域反應微弱
- **不同通道**: 檢測不同類型的特徵

### 注意力熱力圖的意義
- **紅色區域**: 模型認為重要的區域
- **藍色區域**: 模型認為不重要的區域
- **可以看出模型是否關注正確的物體部位**

## 📝 程式碼範例

### 單獨使用某個功能

```python
from model_first_layer_visualizer import FirstLayerVisualizer

# 初始化
visualizer = FirstLayerVisualizer("your_model.pth")

# 載入圖片
original_img, processed_img = visualizer.load_image("your_image.jpg")

# 只看權重
weights = visualizer.visualize_first_layer_weights()

# 只看特徵圖
feature_maps = visualizer.visualize_feature_maps(processed_img)

# 只看注意力熱力圖
avg_heatmap, max_heatmap, max_channel = visualizer.create_attention_heatmap(
    processed_img, original_img
)

# 只分析通道響應
channel_df = visualizer.analyze_channel_responses(processed_img)
```

## 🛠️ 技術細節

### 支援的模型
- EfficientNet 系列
- ConvNeXt 系列  
- 任何使用 timm 創建的模型

### 圖片格式
- JPG, PNG, JPEG
- 會自動調整到 224x224
- 會自動進行標準化處理

### 依賴套件
- torch, torchvision
- timm
- opencv-python
- matplotlib, seaborn
- numpy, pandas
- PIL

## 🔧 疑難排解

### 找不到模型檔案
確認當前目錄有 `.pth` 檔案，或修改程式中的搜尋路徑

### 找不到圖片
確認有圖片檔案，或修改 `get_sample_images()` 函數中的搜尋路徑

### 記憶體不足
- 減少 `max_channels` 參數
- 使用較小的 `figsize`
- 確保有足夠的 GPU/CPU 記憶體

### 模型載入失敗
確認模型檔案包含必要的資訊：
- `model_name`
- `model_state_dict`
- `num_classes`

## 💡 實用技巧

### 1. 批量分析多張圖片
修改 main() 函數，對多張圖片進行迴圈分析

### 2. 比較不同模型
對同一張圖片使用不同模型，比較注意力差異

### 3. 關注特定通道
找到響應最強的通道，分析其對應的特徵類型

### 4. 診斷模型問題
- 如果注意力集中在背景：可能需要更好的資料清理
- 如果某些通道完全不激活：可能訓練不充分
- 如果注意力分散：可能需要更多正則化

## 🎯 實際應用

### 模型診斷
- 檢查模型是否學到正確特徵
- 發現過擬合或欠擬合問題
- 驗證注意力機制是否合理

### 資料集改進
- 找出模型關注的錯誤區域
- 識別需要更多樣本的特徵
- 發現標籤錯誤

### 模型優化
- 根據特徵圖調整網路架構
- 優化資料增強策略
- 改進預處理流程