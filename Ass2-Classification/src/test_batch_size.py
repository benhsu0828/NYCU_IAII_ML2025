#!/usr/bin/env python3
"""
Batch Size 檢測器
快速檢測您的 GPU 可以支援的最大 batch size
"""

import torch
import sys
import os

# 添加當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MemoryViT_character_classifier import MemoryViTCharacterClassifier, get_best_data_path

def test_batch_size():
    """快速測試最佳 batch size"""
    print("🔍 MemoryViT Batch Size 檢測器")
    print("=" * 50)
    
    # 檢查 GPU
    if not torch.cuda.is_available():
        print("❌ 未檢測到 GPU，無法進行 batch size 檢測")
        print("💡 建議在 CPU 模式下使用 batch_size=4")
        return
    
    device = torch.device('cuda')
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # 獲取資料路徑
        data_paths, data_type = get_best_data_path()
        print(f"📂 使用資料: {data_type}")
        
        # 初始化分類器
        print("\n🏗️ 初始化 MemoryViT 模型...")
        classifier = MemoryViTCharacterClassifier(
            num_classes=50,
            image_size=224,
            device=device
        )
        
        # 準備資料（只需要少量資料來創建模型）
        print("📊 準備資料...")
        train_dataset, val_dataset, test_dataset = classifier.prepare_data(data_paths)
        
        # 執行 batch size 檢測
        print("\n" + "="*50)
        optimal_batch_size = classifier.find_optimal_batch_size(
            max_batch_size=256,  # 測試更大的範圍
            start_batch_size=16
        )
        print("="*50)
        
        # 顯示建議
        print(f"\n🎯 檢測完成！")
        print(f"📝 建議在訓練時使用: batch_size={optimal_batch_size}")
        
        # 提供不同場景的建議
        print(f"\n💡 使用建議:")
        print(f"   🚀 快速測試: batch_size={min(optimal_batch_size, 32)}")
        print(f"   ⚡ 最佳效能: batch_size={optimal_batch_size}")
        print(f"   🛡️ 保守安全: batch_size={max(16, optimal_batch_size // 2)}")
        
        # 保存結果
        with open('optimal_batch_size.txt', 'w') as f:
            f.write(f"Optimal Batch Size: {optimal_batch_size}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        
        print(f"\n💾 結果已保存至: optimal_batch_size.txt")
        
    except FileNotFoundError as e:
        print(f"❌ 資料路徑錯誤: {e}")
        print("請確認資料夾結構正確")
    except Exception as e:
        print(f"❌ 檢測過程出錯: {e}")
        import traceback
        traceback.print_exc()

def quick_recommendation():
    """基於 GPU 記憶體給出快速建議"""
    print("\n🚀 快速 GPU 記憶體建議:")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory >= 16:
            recommended = 64
            tier = "🔥 高端"
        elif gpu_memory >= 12:
            recommended = 48
            tier = "💪 高性能"
        elif gpu_memory >= 8:
            recommended = 32
            tier = "⚡ 中高端"
        elif gpu_memory >= 6:
            recommended = 24
            tier = "👍 主流"
        else:
            recommended = 16
            tier = "💼 入門"
        
        print(f"等級: {tier}")
        print(f"建議 batch size: {recommended}")
        
        return recommended
    else:
        print("未檢測到 GPU")
        return 4

if __name__ == "__main__":
    print("選擇檢測模式:")
    print("1. 完整檢測 (精確，需要約 2-3 分鐘)")
    print("2. 快速建議 (基於 GPU 記憶體，立即完成)")
    
    choice = input("請選擇 (1/2): ").strip()
    
    if choice == "2":
        quick_recommendation()
    else:
        test_batch_size()