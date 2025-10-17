#!/usr/bin/env python
"""
快速訓練 MemoryViT 腳本
"""

import os
import platform
import sys

def check_environment():
    """檢查訓練環境"""
    print("🔍 檢查訓練環境...")
    
    # 檢查必要套件
    required_packages = [
        'torch', 'torchvision', 'vit_pytorch', 
        'sklearn', 'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  缺少套件: {', '.join(missing)}")
        print("請在 vit_env 環境中安裝:")
        for pkg in missing:
            if pkg == 'vit_pytorch':
                print("pip install vit-pytorch")
            else:
                print(f"pip install {pkg}")
        return False
    
    return True

def check_data():
    """檢查訓練資料"""
    print("\n📁 檢查訓練資料...")
    
    # 檢測環境
    is_wsl = "microsoft" in platform.uname().release.lower()
    
    if is_wsl:
        base_path = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset"
    else:
        base_path = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset"
    
    # 檢查資料路徑
    augmented_path = f"{base_path}/augmented/train"
    preprocessed_path = f"{base_path}/preprocessed/train"
    
    if os.path.exists(augmented_path):
        print(f"  ✅ 找到增強資料: {augmented_path}")
        return augmented_path, "增強資料 (推薦)"
    elif os.path.exists(preprocessed_path):
        print(f"  ⚠️  只找到預處理資料: {preprocessed_path}")
        print("  💡 建議先執行資料增強以獲得更好效果:")
        print("     python quick_augment.py")
        return preprocessed_path, "預處理資料"
    else:
        print("  ❌ 找不到訓練資料！")
        print("  請確認資料是否在正確位置")
        return None, None

def check_gpu():
    """檢查 GPU 狀態"""
    print("\n🎮 檢查 GPU 狀態...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  💾 VRAM: {gpu_memory:.1f} GB")
            
            # 記憶體建議
            if gpu_memory >= 8:
                batch_size = 32
                print(f"  🚀 建議批次大小: {batch_size}")
            elif gpu_memory >= 4:
                batch_size = 16  
                print(f"  🚀 建議批次大小: {batch_size}")
            else:
                batch_size = 8
                print(f"  ⚠️  建議批次大小: {batch_size} (VRAM 較小)")
            
            return True, batch_size
        else:
            print("  ⚠️  未檢測到 GPU，將使用 CPU")
            return False, 8
            
    except ImportError:
        print("  ❌ PyTorch 未安裝")
        return False, 8

def estimate_training_time(data_path, batch_size, has_gpu):
    """預估訓練時間"""
    print(f"\n⏱️  預估訓練時間...")
    
    try:
        import glob
        
        # 統計圖片數量
        total_images = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            total_images += len(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
        
        print(f"  📊 總圖片數: {total_images:,}")
        
        # 預估時間
        if has_gpu:
            time_per_epoch = total_images / batch_size * 0.5  # 秒
            speed_note = "GPU 加速"
        else:
            time_per_epoch = total_images / batch_size * 2.0  # 秒
            speed_note = "CPU 模式"
        
        total_time = time_per_epoch * 100 / 60  # 100 epochs in minutes
        
        print(f"  🔄 每 epoch: ~{time_per_epoch/60:.1f} 分鐘 ({speed_note})")
        print(f"  🎯 100 epochs: ~{total_time:.1f} 分鐘 ({total_time/60:.1f} 小時)")
        
        return total_images
        
    except Exception as e:
        print(f"  ❌ 預估失敗: {e}")
        return 0

def start_training():
    """開始訓練"""
    print(f"\n🚀 開始 MemoryViT 訓練...")
    
    try:
        # 直接導入並執行主函數
        from MemoryViT_character_classifier import main
        main()
        
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()



def main():
    """主函數"""
    print("🎭 MemoryViT 快速訓練啟動器")
    print("=" * 50)
    
    # 1. 檢查環境
    if not check_environment():
        print("\n❌ 環境檢查失敗，請先安裝必要套件")
        return
    
    # 2. 檢查資料
    data_path, data_type = check_data()
    if data_path is None:
        print("\n❌ 資料檢查失敗，請準備訓練資料")
        return
    
    # 3. 檢查 GPU
    has_gpu, batch_size = check_gpu()
    
    # 4. 預估訓練時間
    total_images = estimate_training_time(data_path, batch_size, has_gpu)
    
    # 5. 訓練配置總結
    print(f"\n📋 訓練配置總結:")
    print(f"  📂 資料類型: {data_type}")
    print(f"  📊 圖片數量: {total_images:,}")
    print(f"  🎮 運算設備: {'GPU' if has_gpu else 'CPU'}")
    print(f"  📦 批次大小: {batch_size}")
    print(f"  🎯 目標類別: 50 個角色")
    print(f"  🔄 訓練輪數: 100 epochs")
    
    # 6. 詢問用戶
    print(f"\n💡 MemoryViT 特色:")
    print(f"  🧠 只訓練 1.7% 參數 (高效率)")
    print(f"  💾 記憶機制增強特徵學習")
    print(f"  🚀 比完整 ViT 快 5-10 倍")
    
    choice = input(f"\n是否開始訓練? (y/n): ")
    if choice.lower() in ['y', 'yes', '是']:
        start_training()
    else:
        print("訓練已取消")
        print("\n手動開始訓練:")
        print("python MemoryViT_character_classifier.py")

if __name__ == "__main__":
    main()