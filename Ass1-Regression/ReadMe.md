# 作業連結
https://www.kaggle.com/competitions/nycu-iaii-ml-2025-regression/overview

# 建立環境
## Windows 建立 ML conda 環境（適用 CUDA 11.6）

1. 建立環境：
```sh
conda create -n NYCUML python=3.8
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