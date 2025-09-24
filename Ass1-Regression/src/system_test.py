#!/usr/bin/env python3
"""
系統測試：檢查主程式各功能是否正常
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 加入 src 目錄到路徑
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """測試資料載入"""
    print("=== 測試資料載入 ===")
    
    try:
        from data_preprocess import load_processed_data
        train_df, valid_df, test_df = load_processed_data("processed")
        
        print(f"✅ 資料載入成功")
        print(f"Train: {train_df.shape}")
        print(f"Valid: {valid_df.shape}")
        print(f"Test: {test_df.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 資料載入失敗: {e}")
        return False

def test_model_import():
    """測試模型導入"""
    print("\n=== 測試模型導入 ===")
    
    try:
        from model import RegressionModels
        models = RegressionModels(random_state=42)
        
        print("✅ 模型類別導入成功")
        
        # 測試模型配置
        tree_models = models.get_tree_models()
        print(f"可用樹模型: {list(tree_models.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ 模型導入失敗: {e}")
        return False

def test_main_import():
    """測試主程式導入"""
    print("\n=== 測試主程式導入 ===")
    
    try:
        import main
        print("✅ 主程式導入成功")
        
        # 檢查主要函數是否存在
        functions = ['load_data_and_prepare', 'train_models', 'test_model', 'get_user_choice']
        for func in functions:
            if hasattr(main, func):
                print(f"  ✅ {func} 函數存在")
            else:
                print(f"  ❌ {func} 函數不存在")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 主程式導入失敗: {e}")
        return False

def test_requirements():
    """測試套件依賴"""
    print("\n=== 測試套件依賴 ===")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'lightgbm', 'catboost', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'xgboost':
                import xgboost
            elif package == 'lightgbm':
                import lightgbm
            elif package == 'catboost':
                import catboost
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'seaborn':
                import seaborn
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
            
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少套件: {missing_packages}")
        print("請執行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有必要套件已安裝")
        return True

def test_directory_structure():
    """測試目錄結構"""
    print("\n=== 測試目錄結構 ===")
    
    base_dir = Path("..").resolve()
    required_dirs = ['Dataset/processed', 'models', 'results', 'src']
    required_files = ['src/main.py', 'src/model.py', 'src/data_preprocess.py', 'requirements.txt']
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/")
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        if missing_dirs:
            print(f"\n缺少目錄: {missing_dirs}")
        if missing_files:
            print(f"缺少檔案: {missing_files}")
        return False
    else:
        print("✅ 目錄結構完整")
        return True

def run_all_tests():
    """執行所有測試"""
    print("🧪 系統測試開始")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_requirements,
        test_data_loading,
        test_model_import,
        test_main_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 測試 {test_func.__name__} 發生錯誤: {e}")
    
    print("\n" + "=" * 50)
    print(f"測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("✅ 系統準備就緒！可以執行 main.py")
        print("\n使用方法:")
        print("  python main.py")
        print("  然後選擇執行模式 (train/test/both/quick)")
    else:
        print("❌ 部分測試失敗，請修正後再執行")
        print("\n常見問題:")
        print("1. 缺少套件 → pip install -r requirements.txt")
        print("2. 缺少資料 → 執行資料預處理")
        print("3. 路徑問題 → 確認在正確目錄執行")

if __name__ == "__main__":
    run_all_tests()
