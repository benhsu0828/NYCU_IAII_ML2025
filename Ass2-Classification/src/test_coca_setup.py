#!/usr/bin/env python3
"""
🧪 CoCa 測試腳本 - 驗證 open-clip 安裝和 CoCa 模型載入

測試內容：
1. 檢查 open-clip-torch 安裝
2. 測試 CoCa 模型載入
3. 測試 Mac MPS 支援
4. 驗證基本推理功能
"""

import torch
import sys
import os

def test_basic_imports():
    """測試基本套件匯入"""
    print("🧪 測試基本套件...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch 匯入失敗: {e}")
        return False
        
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision 匯入失敗: {e}")
        return False
        
    try:
        import open_clip
        print(f"✅ OpenCLIP: {open_clip.__version__}")
    except ImportError as e:
        print(f"❌ OpenCLIP 匯入失敗: {e}")
        print("   請安裝: pip install open-clip-torch")
        return False
        
    return True

def test_device_support():
    """測試設備支援"""
    print("\n🖥️ 測試設備支援...")
    
    # CPU 支援
    print(f"✅ CPU 可用")
    
    # MPS 支援 (Mac)
    if torch.backends.mps.is_available():
        print(f"✅ MPS (Metal) 可用")
        try:
            # 測試 MPS 基本操作
            x = torch.randn(2, 3).to('mps')
            y = x * 2
            print(f"   MPS 測試運算: {y.shape}")
        except Exception as e:
            print(f"   ⚠️ MPS 測試失敗: {e}")
    else:
        print(f"ℹ️ MPS 不可用（非 Apple Silicon Mac）")
    
    # CUDA 支援 (通常 Mac 不會有)
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用: {torch.cuda.get_device_name()}")
    else:
        print(f"ℹ️ CUDA 不可用")

def test_coca_models():
    """測試 CoCa 模型載入"""
    print("\n🤖 測試 CoCa 模型...")
    
    try:
        import open_clip
        
        # 列出可用的模型
        available_models = open_clip.list_models()
        coca_models = [m for m in available_models if 'coca' in m.lower()]
        
        print(f"📋 找到 {len(coca_models)} 個 CoCa 模型:")
        for model in coca_models[:5]:  # 只顯示前5個
            print(f"   - {model}")
            
        if not coca_models:
            print("⚠️ 沒有找到 CoCa 模型")
            return False
        
        # 測試載入一個 CoCa 模型
        test_model = coca_models[0]
        print(f"\n🔄 測試載入模型: {test_model}")
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                test_model,
                pretrained='laion2b_s13b_b90k'
            )
            
            # 測試模型基本資訊
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ 模型載入成功")
            print(f"   參數量: {total_params/1e6:.1f}M")
            print(f"   預處理: {type(preprocess)}")
            
            # 測試編碼功能
            dummy_image = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                features = model.encode_image(dummy_image)
                print(f"   圖片特徵維度: {features.shape}")
                
                # 如果有文字編碼功能也測試一下
                try:
                    dummy_text = open_clip.tokenize(["a photo of simpson character"])
                    text_features = model.encode_text(dummy_text)
                    print(f"   文字特徵維度: {text_features.shape}")
                except:
                    print(f"   (沒有文字編碼功能)")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ OpenCLIP 不可用: {e}")
        return False

def test_image_processing():
    """測試圖片處理功能"""
    print("\n🖼️ 測試圖片處理...")
    
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        import numpy as np
        
        # 創建測試圖片
        test_image = Image.new('RGB', (224, 224), color='red')
        print("✅ PIL 圖片創建成功")
        
        # 測試變換
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        tensor = transform(test_image)
        print(f"✅ 圖片變換成功: {tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 圖片處理測試失敗: {e}")
        return False

def test_file_operations():
    """測試檔案操作"""
    print("\n📁 測試檔案操作...")
    
    try:
        import glob
        import pandas as pd
        
        # 測試目錄
        test_dirs = [
            "Dataset/test",
            "src",
            "."
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                files = glob.glob(os.path.join(test_dir, "*"))
                print(f"✅ {test_dir}: {len(files)} 個檔案")
                break
        else:
            print("⚠️ 測試目錄都不存在")
        
        # 測試 pandas
        df = pd.DataFrame({'id': [1, 2, 3], 'character': ['homer', 'marge', 'bart']})
        print(f"✅ Pandas DataFrame: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 檔案操作測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 CoCa 系統測試")
    print("=" * 50)
    
    # 測試結果記錄
    tests = [
        ("基本套件匯入", test_basic_imports),
        ("設備支援", test_device_support),
        ("CoCa 模型", test_coca_models),
        ("圖片處理", test_image_processing),
        ("檔案操作", test_file_operations),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 測試異常: {e}")
            results[test_name] = False
    
    # 總結
    print(f"\n{'='*50}")
    print("📊 測試結果總結:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 總計: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！CoCa 系統準備就緒")
    elif passed >= total * 0.7:
        print("⚠️ 大部分測試通過，系統基本可用")
    else:
        print("❌ 多項測試失敗，請檢查環境設定")
        
    # 給出建議
    print(f"\n💡 建議:")
    if not results.get("基本套件匯入", True):
        print("   1. 安裝必要套件: pip install torch torchvision open-clip-torch")
    if not results.get("CoCa 模型", True):
        print("   2. 檢查 open-clip-torch 版本: pip install --upgrade open-clip-torch")
    if not results.get("圖片處理", True):
        print("   3. 安裝圖片處理套件: pip install Pillow pandas tqdm")
    
    print(f"\n🚀 如果測試通過，可以開始使用 CoCa 分類器了！")

if __name__ == "__main__":
    main()
