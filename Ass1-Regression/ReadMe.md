# 作業連結
https://www.kaggle.com/competitions/nycu-iaii-ml-2025-regression/overview

Ass1-Regression/
│
├── data/                # 原始資料、處理後資料（如 Dataset/ 可放這裡）
│   ├── raw/             # 原始檔案
│   └── processed/       # 前處理後的檔案
│
├── src/                 # 所有 Python 程式
│   ├── data_preprocess.py   # 資料前處理
│   ├── model.py             # 模型定義與訓練
│   ├── main.py             # 主程式
│
│
├── models/              # 儲存訓練好的模型檔案
│
├── results/             # 儲存預測結果、圖表、報告
│
├── requirements.txt     # Python 套件需求
├── .gitignore           # 忽略不需上傳的檔案
└── README.md            # 專案說明

# 建立環境
## Windows 建立 ML conda 環境（適用 CUDA 11.6）

1. 建立環境：
```sh
conda create -n NYCUML python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch
安裝 cuDNN 8.x（推薦 
```

> Python 3.8 為 CUDA 11.6 官方推薦版本，若有特殊需求可查詢 [CUDA 對應 Python 版本](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)

2. 啟動環境：

```sh
conda activate NYCUML
```

3. 安裝 ML 套件（以 PyTorch 為例）：

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

# 訓練資料
Submissions are evaluated on the mean absolute error (MAE) between the predicted sale price and the observed one.