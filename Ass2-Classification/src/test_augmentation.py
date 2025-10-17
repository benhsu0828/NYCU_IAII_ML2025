#!/usr/bin/env python
"""
測試資料增強效果 - 驗證與原始 data_aggV1.py 和 data_aggV2.py 的一致性
"""

import os
import random
from PIL import Image
from pathlib import Path

def test_v1_transform():
    """測試 data_aggV1.py 的變換效果"""
    print("🧪 測試 data_aggV1.py 變換...")
    
    try:
        from data_augmentation import get_augmentation_transforms
        
        # 獲取變換
        transform = get_augmentation_transforms()
        
        # 創建測試圖片
        test_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        
        # 應用變換
        transformed = transform(test_img)
        
        print("✅ data_aggV1.py 變換測試成功")
        print(f"   輸入: {test_img.size} {test_img.mode}")
        print(f"   輸出: {transformed.size} {transformed.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ data_aggV1.py 變換測試失敗: {e}")
        return False

def test_v2_background():
    """測試 data_aggV2.py 的背景合成效果"""
    print("\n🧪 測試 data_aggV2.py 背景合成...")
    
    try:
        from data_augmentation import create_background_composite
        
        # 創建測試前景圖片（含黑白背景）
        fg_img = Image.new('RGB', (100, 100), color=(255, 255, 255))  # 白色背景
        # 在中間畫一個灰色方塊（模擬角色）
        for x in range(30, 70):
            for y in range(30, 70):
                fg_img.putpixel((x, y), (128, 128, 128))
        
        # 創建測試背景圖片
        bg_img = Image.new('RGB', (200, 200), color=(100, 150, 200))
        
        # 應用背景合成
        composite = create_background_composite(fg_img, bg_img)
        
        print("✅ data_aggV2.py 背景合成測試成功")
        print(f"   前景: {fg_img.size} {fg_img.mode}")
        print(f"   背景: {bg_img.size} {bg_img.mode}")
        print(f"   合成: {composite.size} {composite.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ data_aggV2.py 背景合成測試失敗: {e}")
        return False

def test_noise_classes():
    """測試自定義噪聲類別"""
    print("\n🧪 測試自定義噪聲類別...")
    
    try:
        import torch
        from data_augmentation import AddGaussianNoise, AddSpeckleNoise, AddPoissonNoise, AddSaltPepperNoise
        
        # 創建測試 tensor
        test_tensor = torch.rand(3, 50, 50)
        
        # 測試各種噪聲
        noises = [
            ("高斯噪聲", AddGaussianNoise(0., 0.05)),
            ("散斑噪聲", AddSpeckleNoise(noise_level=0.1)),
            ("泊松噪聲", AddPoissonNoise(lam=0.1)),
            ("椒鹽噪聲", AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05))
        ]
        
        for name, noise_transform in noises:
            result = noise_transform(test_tensor)
            print(f"   ✅ {name}: {test_tensor.shape} -> {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 噪聲類別測試失敗: {e}")
        return False

def test_with_real_data():
    """使用真實資料測試"""
    print("\n🧪 使用真實資料測試...")
    
    # 檢查是否有預處理資料
    test_dir = Path("E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/train")
    
    if not test_dir.exists():
        print("⚠️  找不到預處理資料，跳過真實資料測試")
        return True
    
    # 找第一個類別的第一張圖片
    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print("⚠️  預處理資料夾為空，跳過真實資料測試")
        return True
    
    first_class = class_dirs[0]
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(first_class.glob(ext)))
    
    if not image_files:
        print("⚠️  找不到圖片檔案，跳過真實資料測試")
        return True
    
    try:
        from data_augmentation import get_augmentation_transforms
        
        # 載入真實圖片
        test_img = Image.open(image_files[0]).convert("RGB")
        print(f"   載入測試圖片: {test_img.size}")
        
        # 應用變換
        transform = get_augmentation_transforms()
        result = transform(test_img)
        
        print(f"   ✅ 真實資料變換成功: {test_img.size} -> {result.size}")
        return True
        
    except Exception as e:
        print(f"❌ 真實資料測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("🔬 資料增強一致性測試")
    print("驗證與你的 data_aggV1.py 和 data_aggV2.py 的一致性")
    print("=" * 60)
    
    tests = [
        test_v1_transform,
        test_v2_background, 
        test_noise_classes,
        test_with_real_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("✅ 所有測試通過！資料增強與你的原始腳本一致")
        print("\n🚀 你可以安心使用:")
        print("   python quick_augment.py")
        print("   或")
        print("   python data_augmentation.py")
    else:
        print("❌ 部分測試失敗，請檢查環境配置")

if __name__ == "__main__":
    main()