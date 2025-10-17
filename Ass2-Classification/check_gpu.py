#!/usr/bin/env python3

import torch

print("=== GPU 可用性檢查 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  記憶體: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # 測試 GPU 運算
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU 運算測試通過！")
        print(f"建議使用: --device cuda")
    except Exception as e:
        print(f"❌ GPU 運算測試失敗: {e}")
        print(f"建議使用: --device cpu")
else:
    print("❌ CUDA 不可用")
    print(f"建議使用: --device cpu")

print("\n=== 推薦設定 ===")
if torch.cuda.is_available():
    print("🚀 你的系統支援 GPU 加速！")
    print("執行指令：python main_finetune.py --device cuda")
else:
    print("💻 使用 CPU 訓練（速度較慢）")
    print("執行指令：python main_finetune.py --device cpu")