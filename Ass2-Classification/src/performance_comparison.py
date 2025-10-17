#!/usr/bin/env python
"""
資料增強性能比較工具
"""

import os
import time
from pathlib import Path
import platform

def benchmark_cpu_vs_gpu():
    """比較 CPU 和 GPU 增強性能"""
    
    print("⚡ 資料增強性能比較")
    print("=" * 50)
    
    # 檢查 GPU 可用性
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ GPU 不可用")
    except:
        print("❌ PyTorch 未安裝")
    
    # CPU 信息
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"💻 CPU: {cpu_count} 核心")
    except:
        cpu_count = "未知"
    
    print("\n📊 性能預估 (1000張圖片):")
    print("-" * 50)
    
    methods = [
        {
            "name": "原始 CPU (逐張)",
            "file": "data_augmentation.py", 
            "estimated_time": "15-20 分鐘",
            "memory": "2-4 GB",
            "pros": ["低記憶體需求", "穩定可靠"],
            "cons": ["速度較慢", "CPU 利用率低"]
        },
        {
            "name": "GPU 加速 (批次)",
            "file": "gpu_augmentation.py",
            "estimated_time": "3-5 分鐘" if gpu_available else "不可用",
            "memory": "4-8 GB",
            "pros": ["速度快 3-5倍", "批次處理", "並行計算"],
            "cons": ["需要 GPU", "記憶體需求高"]
        },
        {
            "name": "CPU 多核 (並行)",
            "file": "準備開發中...",
            "estimated_time": "8-12 分鐘",
            "memory": "3-6 GB", 
            "pros": ["中等速度", "充分利用多核"],
            "cons": ["複雜度高", "記憶體需求中等"]
        }
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n{i}. {method['name']}")
        print(f"   📝 腳本: {method['file']}")
        print(f"   ⏱️  預估時間: {method['estimated_time']}")
        print(f"   💾 記憶體: {method['memory']}")
        print(f"   ✅ 優點: {', '.join(method['pros'])}")
        print(f"   ❌ 缺點: {', '.join(method['cons'])}")

def recommend_method():
    """推薦最適合的方法"""
    print(f"\n🎯 方法推薦:")
    print("-" * 30)
    
    # 檢查 GPU
    has_gpu = False
    gpu_memory = 0
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except:
        pass
    
    if has_gpu and gpu_memory >= 4:
        print("🥇 推薦: GPU 加速方法")
        print("   理由: 你有足夠的 GPU 記憶體")
        print("   腳本: python quick_gpu_augment.py")
        print("   預期效果: 速度提升 3-5 倍")
    elif has_gpu and gpu_memory < 4:
        print("🥈 推薦: GPU 加速方法 (小批次)")
        print("   理由: GPU 可用但記憶體有限")
        print("   腳本: python quick_gpu_augment.py")
        print("   注意: 會自動調整批次大小")
    else:
        print("🥉 推薦: 原始 CPU 方法")
        print("   理由: GPU 不可用，使用穩定的 CPU 方法")
        print("   腳本: python quick_augment.py") 
        print("   特點: 穩定可靠，記憶體需求低")

def optimization_tips():
    """優化建議"""
    print(f"\n💡 優化建議:")
    print("-" * 30)
    
    tips = [
        "🔧 如果 GPU 記憶體不足，降低批次大小 (--batch_size 4)",
        "⚡ 關閉不必要的程序釋放 VRAM",
        "💾 確保有足夠的硬碟空間 (增強後約 4 倍大小)",
        "🌡️  長時間運行注意 GPU 溫度",
        "🔄 可以分批處理類別避免記憶體問題",
        "📊 先用小量資料測試確定最佳參數"
    ]
    
    for tip in tips:
        print(f"   {tip}")

def speed_calculator():
    """速度計算器"""
    print(f"\n🧮 速度計算器:")
    print("-" * 30)
    
    try:
        # 獲取資料量
        is_wsl = "microsoft" in platform.uname().release.lower()
        if is_wsl:
            input_dir = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/train"
        else:
            input_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\Dataset\preprocessed\train"
        
        if os.path.exists(input_dir):
            class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
            total_images = 0
            for class_dir in class_dirs:
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(list(class_dir.glob(ext)))
                total_images += len(image_files)
            
            print(f"📊 你的資料量: {total_images} 張原始圖片")
            print(f"📈 增強後數量: ~{total_images * 4} 張")
            
            # 時間預估
            cpu_time = total_images * 0.8  # 每張約 0.8 秒
            gpu_time = total_images * 0.2  # 每張約 0.2 秒 (GPU加速)
            
            print(f"\n⏱️  預估處理時間:")
            print(f"   CPU 方法: {cpu_time/60:.1f} 分鐘")
            print(f"   GPU 方法: {gpu_time/60:.1f} 分鐘 (節省 {(cpu_time-gpu_time)/60:.1f} 分鐘)")
            
        else:
            print("📁 找不到預處理資料，無法計算")
            
    except Exception as e:
        print(f"❌ 計算失敗: {e}")

def main():
    """主函數"""
    benchmark_cpu_vs_gpu()
    recommend_method()
    optimization_tips()
    speed_calculator()
    
    print(f"\n🚀 準備開始增強:")
    print("1. GPU 加速 (推薦): python quick_gpu_augment.py")
    print("2. CPU 穩定版: python quick_augment.py")
    print("3. 手動參數版: python gpu_augmentation.py --help")

if __name__ == "__main__":
    main()