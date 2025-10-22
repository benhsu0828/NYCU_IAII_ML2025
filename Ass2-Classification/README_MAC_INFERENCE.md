# Mac EfficientNet 推理工具

這個工具專為 Mac 系統優化，可以從 `Dataset/test` 目錄讀取測試圖片，使用訓練好的 EfficientNet 模型進行預測，並輸出 CSV 格式的結果。

## 功能特色

- 🍎 **Mac 優化**: 支援 MPS (Metal Performance Shaders) 加速
- 📁 **自動讀取**: 自動讀取 `Dataset/test` 目錄下的所有圖片
- 📊 **CSV 輸出**: 輸出格式為兩欄：`filename` 和 `prediction`
- 🚀 **批量處理**: 支援大量圖片的批量推理
- 📈 **進度顯示**: 使用進度條顯示處理進度

## 環境設定

### 1. 安裝依賴套件

```bash
# 切換到項目目錄
cd /Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification

# 安裝依賴
pip install -r requirements.txt
```

### 2. 確認 PyTorch 支援 MPS

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## 使用方法

### 命令列模式

```bash
# 基本用法
python src/mac_inference.py --model /Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification/convnext_tiny_epoch_013_acc_99.91.pth --test-dir /Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification/Dataset/test --output my_predictions.csv

# 指定輸出檔案
python src/mac_inference.py --model model.pth --output my_predictions.csv

# 指定測試目錄
python src/mac_inference.py --model model.pth --test-dir custom/test/dir

# 強制使用 CPU
python src/mac_inference.py --model model.pth --device cpu
```

### 互動模式

直接執行程式，會進入互動模式：

```bash
python src/mac_inference.py
```

程式會引導您：
1. 選擇模型檔案
2. 設定測試目錄
3. 設定輸出檔案名稱

## 輸出格式

輸出的 CSV 檔案包含兩欄：

```csv
filename,prediction
1.jpg,class_name_1
2.jpg,class_name_2
3.jpg,class_name_1
...
```

## 目錄結構

確保您的目錄結構如下：

```
Ass2-Classification/
├── Dataset/
│   └── test/           # 測試圖片目錄
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
├── src/
│   └── mac_inference.py
├── requirements.txt
└── README_MAC_INFERENCE.md
```

## 範例使用

假設您有一個名為 `best_model.pth` 的模型檔案：

```bash
# 1. 進入項目目錄
cd /Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification

# 2. 執行推理
python src/mac_inference.py --model best_model.pth

# 3. 查看結果
cat predictions.csv
```

## 故障排除

### 1. 模型載入失敗
- 確認模型檔案路徑正確
- 確認模型檔案是完整的 PyTorch checkpoint

### 2. 找不到測試圖片
- 確認 `Dataset/test` 目錄存在
- 確認目錄中有 `.jpg`、`.png` 等圖片檔案

### 3. MPS 不可用
- 確保使用 macOS 12.3 或更新版本
- 確保安裝了支援 MPS 的 PyTorch 版本

### 4. 記憶體不足
- 使用 `--device cpu` 強制使用 CPU
- 考慮分批處理大量圖片

## 效能優化

- **MPS 加速**: 在支援的 Mac 上會自動使用 MPS 加速
- **批量處理**: 程式會顯示處理進度，大量圖片也能高效處理
- **記憶體管理**: 每張圖片處理後會釋放記憶體，避免記憶體洩漏

## 支援的圖片格式

- `.jpg` / `.jpeg`
- `.png`
- `.bmp`
- `.gif`
- `.webp`
