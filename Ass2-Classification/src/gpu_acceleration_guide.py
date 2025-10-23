#!/usr/bin/env python3
"""
GPU 加速資料增強使用範例
"""

print("🚀 GPU 加速資料增強使用指南")
print("=" * 60)

print("""
🎯 基本使用 (自動使用 GPU)：
   python data_augmentation.py

🎯 自訂 GPU 批量大小：
   python data_augmentation.py --batch_size 64

🎯 強制使用 CPU (不用 GPU)：
   python data_augmentation.py --no_gpu

🎯 完整自訂範例：
   python data_augmentation.py \\
       --augment_per_image 5 \\
       --batch_size 16 \\
       --max_bg_per_category 20 \\
       --background_prob 0.5

💡 GPU 加速優勢：
   ✅ 速度提升：2-5倍加速 (取決於 GPU 型號)
   ✅ 批量處理：同時處理多張圖片
   ✅ 記憶體優化：自動清理 GPU 記憶體
   ✅ 自動回退：GPU 不可用時自動用 CPU

⚠️ 注意事項：
   - GPU 記憶體不足時會自動降低批量大小
   - 某些增強操作仍需在 CPU 上完成
   - 背景合成因為涉及複雜邏輯，仍在 CPU 上處理

📊 性能比較 (參考數值)：
   CPU 模式：     ~50-100 圖片/分鐘
   GPU 模式：     ~200-500 圖片/分鐘 (取決於 GPU)
   
🎮 推薦 GPU 設定：
   RTX 3060/4060：   batch_size=16-32
   RTX 3070/4070：   batch_size=32-64  
   RTX 3080/4080：   batch_size=64-128
   RTX 3090/4090：   batch_size=128+
""")

# 檢查 GPU 可用性
import torch

if torch.cuda.is_available():
    print(f"✅ 檢測到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 推薦批量大小
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_memory_gb >= 24:
        recommended_batch = 128
    elif gpu_memory_gb >= 12:
        recommended_batch = 64
    elif gpu_memory_gb >= 8:
        recommended_batch = 32
    elif gpu_memory_gb >= 6:
        recommended_batch = 16
    else:
        recommended_batch = 8
    
    print(f"💡 推薦批量大小: {recommended_batch}")
    print(f"\n🚀 立即開始 GPU 加速增強：")
    print(f"   python data_augmentation.py --batch_size {recommended_batch}")
else:
    print("❌ 未檢測到 GPU，將使用 CPU 模式")
    print("💡 確認事項：")
    print("   1. 已安裝 CUDA 版本的 PyTorch")
    print("   2. NVIDIA 驅動程式已更新")
    print("   3. CUDA 工具包已安裝")