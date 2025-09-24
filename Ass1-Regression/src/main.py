#!/usr/bin/env python3
"""
主程式：房地產價格預測
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# 加入 src 目錄到路徑
sys.path.append(str(Path(__file__).parent))

from data_preprocess import preprocess_data, load_processed_data, validate_model_ready_data
from model import RegressionModels
import os

def get_timestamp():
    """生成時間戳記 - 格式: MMDD_HHMM
    
    例如:
    - 12月25日 14:30 -> "1225_1430"
    - 01月05日 09:15 -> "0105_0915"
    """
    now = datetime.now()
    return now.strftime("%m%d_%H%M")

class DLModelWrapper:
    """深度學習模型包裝器 - 直接包裝訓練好的模型"""
    
    def __init__(self, trained_model):
        self.trained_model = trained_model  # 完整的訓練好的模型
        
    def predict(self, X):
        """預測函數 - 直接使用訓練好的模型"""
        return self.trained_model.predict(X)

def create_dl_model_wrapper(model, model_name, model_dir):
    """創建可序列化的深度學習模型包裝器"""
    
    try:
        print("   🔄 創建深度學習模型包裝器...")
        
        # 🎯 直接包裝整個訓練好的模型（包含 scaler 和 keras 模型）
        wrapper = DLModelWrapper(trained_model=model)
        
        print("   ✅ 深度學習模型包裝器創建成功")
        return wrapper
        
    except Exception as e:
        print(f"   ❌ 包裝器創建失敗: {e}")
        return None

def load_data_and_prepare():
    """載入並準備資料"""
    print("=== 載入資料 ===")
    
    # 1. 資料讀取
    train_df, valid_df, test_df = load_processed_data("processed")
    print(f"資料載入完成 - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    
    # 2. 準備訓練資料
    print("\n2. 準備資料...")
    
    # 確定目標變數名稱
    target_column = '總價元'
    print(f"使用目標變數: {target_column}")
    
    # 分離特徵和目標變數
    X_train = train_df.drop([target_column], axis=1, errors='ignore')
    y_train = train_df[target_column]
    
    X_valid = valid_df.drop([target_column], axis=1, errors='ignore')
    y_valid = valid_df[target_column]
    
    X_test = test_df.drop(["編號"], axis=1, errors='ignore')
    
    # 驗證資料是否準備好
    is_ready = validate_model_ready_data(train_df, valid_df, test_df, target_column)
    if not is_ready:
        print("❌ 資料未準備好，請檢查預處理步驟")
        return None, None, None, None, None, None
    
    print(f"特徵數量: {X_train.shape[1]}")
    print(f"訓練樣本: {X_train.shape[0]}")
    print(f"驗證樣本: {X_valid.shape[0]}")
    print(f"測試樣本: {X_test.shape[0]}")
    
    return X_train, y_train, X_valid, y_valid, X_test, test_df

def train_models(X_train, y_train, X_valid, y_valid, include_linear=False, include_deep_learning=False, tree_only=False):
    """訓練模型
    
    Args:
        X_train, y_train: 訓練資料
        X_valid, y_valid: 驗證資料
        include_linear: 是否包含線性模型
        include_deep_learning: 是否包含深度學習模型
        fast_tree_only: 是否只訓練快速樹模型（精選版本）
    """
    print("\n=== 開始模型訓練 ===")
    
    # 初始化模型
    print("1. 初始化模型...")
    models = RegressionModels(random_state=42)
    
    if tree_only:
        # 完整版：所有樹模型分階段訓練
        print("\n2. 第一階段：快速基礎模型...")
        basic_models = models.get_tree_models()
        models.train_multiple_models(basic_models, X_train, y_train, X_valid, y_valid)
        
        # 顯示第一階段結果
        print("\n=== 第一階段結果 ===")
        results_stage1 = models.get_results_summary()
        print(results_stage1)
        
        # 第二階段：優化樹模型
        print("\n3. 第二階段：優化樹模型...")
        optimized_models = models.get_optimized_tree_models()
        models.train_multiple_models(optimized_models, X_train, y_train, X_valid, y_valid)
    
    # 第三階段：線性模型（可選）
    if include_linear:
        print(f"線性模型...")
        linear_models = models.get_linear_models()
        models.train_multiple_models(linear_models, X_train, y_train, X_valid, y_valid)
    
    # 第四階段：深度學習模型（可選）
    if include_deep_learning:
        print(f"深度學習模型...")
        # 注意：深度學習需要更長時間訓練
        try:
            dl_models = models.get_deep_learning_models()
            models.train_multiple_models(dl_models, X_train, y_train, X_valid, y_valid)
        except Exception as e:
            print(f"⚠️ 深度學習模型訓練失敗: {e}")
    
    # 顯示最終結果
    print("\n=== 最終訓練結果 ===")
    final_results = models.get_results_summary()
    print(final_results)
    
    # 分類別儲存最佳模型
    print(f"\n{6 if not include_deep_learning else 7}. 分類別儲存最佳模型...")
    
    # 🕐 生成統一的時間戳記
    timestamp = get_timestamp()
    best_models_by_type = models.save_best_models_by_type("../models", timestamp)
    
    # 取得整體最佳模型
    if best_models_by_type and 'overall_best' in best_models_by_type:
        overall_best = best_models_by_type['overall_best']
        best_model_name = overall_best['name']
        best_model = overall_best['model']
        
        # 特徵重要性分析（如果是樹模型）
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n{7 if not include_deep_learning else 8}. 特徵重要性分析...")
            analyze_feature_importance(best_model, X_train.columns, best_model_name)
        
        print("\n=== 訓練完成！===")
        print(f"整體最佳模型: {best_model_name}")
        print(f"驗證集 RMSE: {overall_best['rmse']:.2f}")
        print(f"驗證集 R²: {overall_best['r2']:.4f}")
        
        # 顯示各類別最佳模型摘要
        print(f"\n=== 各類別最佳模型摘要 ===")
        for category, info in best_models_by_type.items():
            if category != 'overall_best':
                print(f"{category.upper():15s}: {info['name']:20s} (RMSE: {info['rmse']:.2f}, R²: {info['r2']:.4f})")
        
        return models, best_model_name, best_models_by_type
    else:
        print("❌ 沒有成功訓練的模型")
        return models, None, {}

def test_model(X_test, test_df, model_name=None):
    """使用訓練好的模型進行測試"""
    print("\n=== 開始模型測試 ===")
    
    # 導入必要套件
    try:
        import joblib
    except ImportError:
        print("❌ joblib 套件未安裝，無法載入模型")
        return
    
    model_dir = Path("../models")
    
    if model_name:
        # 指定模型名稱
        print(f"🎯 指定測試模型: {model_name}")
        
        # 🎯 檢查是否為深度學習模型
        if model_name.startswith('DeepLearning'):
            keras_path = model_dir / f"{model_name}_keras"
            scaler_path = model_dir / f"{model_name}_scaler.joblib"
            
            if keras_path.exists() and scaler_path.exists():
                print(f"🧠 檢測到深度學習模型檔案")
                test_single_model_file(X_test, test_df, keras_path, model_name)
                return
            else:
                print(f"❌ 深度學習模型檔案不完整:")
                print(f"   Keras 模型: {keras_path} ({'存在' if keras_path.exists() else '不存在'})")
                print(f"   Scaler: {scaler_path} ({'存在' if scaler_path.exists() else '不存在'})")
                return
        else:
            # 傳統模型
            model_file = model_dir / f"{model_name}.joblib"
            if model_file.exists():
                test_single_model_file(X_test, test_df, model_file, model_name)
                return
            else:
                print(f"❌ 找不到模型檔案: {model_file}")
                return
    else:
        # 自動尋找模型
        print("🔍 自動尋找可用模型...")
        
        # 尋找傳統模型
        joblib_files = list(model_dir.glob("*.joblib"))
        joblib_models = [f for f in joblib_files if not f.name.endswith('_scaler.joblib')]
        
        # 尋找深度學習模型
        keras_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.endswith('_keras')]
        dl_models = []
        for keras_dir in keras_dirs:
            base_name = keras_dir.name.replace('_keras', '')
            scaler_file = model_dir / f"{base_name}_scaler.joblib"
            if scaler_file.exists():
                dl_models.append((keras_dir, base_name))
        
        all_models = []
        
        # 加入傳統模型
        for f in joblib_models:
            all_models.append(('traditional', f, f.stem))
        
        # 加入深度學習模型
        for keras_dir, base_name in dl_models:
            all_models.append(('deeplearning', keras_dir, base_name))
        
        if not all_models:
            print("❌ 沒有找到已訓練的模型檔案")
            print("提示：請先執行訓練模式")
            return
        
        # 選擇最新的模型（以修改時間為準）
        def get_model_time(model_info):
            model_type, model_path, _ = model_info
            if model_type == 'traditional':
                return os.path.getctime(model_path)
            else:  # deeplearning
                return os.path.getctime(model_path)
        
        latest_model = max(all_models, key=get_model_time)
        model_type, model_path, model_name = latest_model
        
        print(f"📊 使用最新模型: {model_name} ({'深度學習' if model_type == 'deeplearning' else '傳統模型'})")
        test_single_model_file(X_test, test_df, model_path, model_name)

    print(f"\n=== 測試完成！===")

def main():
    """主函數 - 訓練模式"""
    print("=== 房地產價格預測模型訓練 ===")
    
    # 載入並準備資料
    data = load_data_and_prepare()
    if data[0] is None:
        return
    
    X_train, y_train, X_valid, y_valid, X_test, test_df = data
    
    # 選擇要訓練的模型類型
    print("\n=== 選擇訓練模型類型 ===")
    print("1. 訓練樹模型")
    print("2. 訓練線性模型")
    print("3. 訓練DL")
    
    while True:
        try:
            choice = input("請選擇 (1-3): ").strip()
            if choice == '1':
                include_linear = False
                include_dl = False
                tree_only = True
                break
            elif choice == '2':
                include_linear = True
                include_dl = False
                tree_only = False
                break
            elif choice == '3':
                include_linear = False
                include_dl = True
                tree_only = False
                break
            else:
                print("請輸入 1、2、3 或 4")
        except KeyboardInterrupt:
            print("\n程序已中止")
            return
    
    # 訓練模型
    result = train_models(
        X_train, y_train, X_valid, y_valid, 
        include_linear=include_linear, 
        include_deep_learning=include_dl,
        fast_tree_only=tree_only
    )
    
    if len(result) == 3:
        models, best_model_name, best_models_by_type = result
    else:
        models, best_model_name = result
        best_models_by_type = {}

def test_only():
    """只進行測試模式"""
    print("=== 房地產價格預測 - 測試模式 ===")
    
    # 載入並準備資料
    data = load_data_and_prepare()
    if data[0] is None:
        return
    
    X_train, y_train, X_valid, y_valid, X_test, test_df = data
    
    # 檢查是否有分類別的模型檔案
    model_dir = Path("../models")
    category_model_files = list(model_dir.glob("best_*_model_*.joblib"))
    
    if category_model_files:
        print("發現分類別訓練的模型，啟用進階測試選項")
        test_model_with_choice(X_test, test_df)
    else:
        print("使用一般測試模式")
        test_model(X_test, test_df)

def train_and_test():
    """訓練並測試模式"""
    print("=== 房地產價格預測 - 訓練並測試模式 ===")
    
    # 載入並準備資料
    data = load_data_and_prepare()
    if data[0] is None:
        return
    
    X_train, y_train, X_valid, y_valid, X_test, test_df = data
    
    # 選擇要訓練的模型類型（快速模式）
    print("\n快速訓練模式：只訓練基礎樹模型")
    
    # 訓練模型（快速模式）
    result = train_models(
        X_train, y_train, X_valid, y_valid, 
        include_linear=False, 
        include_deep_learning=False,
        fast_tree_only=True  # 使用快速模式
    )
    
    if len(result) == 3:
        models, best_model_name, best_models_by_type = result
    else:
        models, best_model_name = result
        best_models_by_type = {}
    
    # 進行測試
    print("\n" + "="*50)
    test_model(X_test, test_df, best_model_name)

def save_predictions(predictions, test_df, model_name):
    """儲存預測結果，編號對應 test_df 的編號欄位"""
    
    # 🎯 直接從 test_df 獲取編號
    if '編號' in test_df.columns:
        test_ids = test_df['編號']
        print(f"✅ 使用測試資料的原始編號欄位")
    else:
        # 如果沒有編號欄位，創建連續編號
        test_ids = pd.Series(range(1, len(test_df) + 1))
        print(f"✅ 創建連續編號 (1 到 {len(test_df)})")
    
    # 確保預測結果與測試資料數量一致
    assert len(predictions) == len(test_df), \
        f"預測結果數量 ({len(predictions)}) 與測試資料數量 ({len(test_df)}) 不匹配"
    
    # 建立預測結果 DataFrame
    results_df = pd.DataFrame({
        '編號': test_ids.values,
        '總價元': predictions
    })
    
    # 儲存路徑
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / f"predictions_{model_name}.csv"
    results_df.to_csv(result_file, index=False, encoding='utf-8-sig')
    
    print(f"預測結果已儲存到: {result_file}")
    
    # 顯示預測統計
    print(f"預測價格統計:")
    print(f"  平均值: {predictions.mean():.2f}")
    print(f"  中位數: {np.median(predictions):.2f}")
    print(f"  最小值: {predictions.min():.2f}")
    print(f"  最大值: {predictions.max():.2f}")

def analyze_feature_importance(model, feature_names, model_name, top_n=20):
    """分析特徵重要性"""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\\n{model_name} - Top {top_n} 重要特徵:")
    print(importance_df.head(top_n))
    
    # 儲存特徵重要性
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    importance_file = results_dir / f"feature_importance_{model_name}.csv"
    importance_df.to_csv(importance_file, index=False, encoding='utf-8-sig')
    
    print(f"特徵重要性已儲存到: {importance_file}")

def get_user_choice():
    """取得用戶選擇的執行模式"""
    print("\n=== 請選擇執行模式 ===")
    print("1. 訓練模式 (train) - 訓練並分類別儲存最佳模型")
    print("2. 測試模式 (test) - 選擇模型類型進行預測")
    print("3. 訓練並測試 (both) - 快速訓練後立即測試")
    print("4. 退出 (exit)")
    
    print("\n💡 新功能說明:")
    print("  - 訓練模式會分別儲存樹模型、線性模型、深度學習模型的最佳版本")
    print("  - 測試模式可以選擇使用哪種類型的模型進行預測")
    print("  - 支援比較不同類型模型的預測結果")
    
    while True:
        try:
            choice = input("\n請輸入選項 (1-4 或 train/test/both/quick/exit): ").strip().lower()
            
            if choice in ['1', 'train']:
                return 'train'
            elif choice in ['2', 'test']:
                return 'test'
            elif choice in ['3', 'both']:
                return 'both'
            elif choice in ['4', 'exit']:
                return 'exit'
            else:
                print("❌ 無效選項，請重新選擇")
        except KeyboardInterrupt:
            print("\n\n程序已中止")
            return 'exit'
        except Exception as e:
            print(f"❌ 輸入錯誤: {e}")

def get_available_models():
    """獲取所有可用的模型資訊"""
    model_dir = Path("../models")
    available_models = {}
    
    # 🔍 尋找傳統模型 (.joblib 檔案，但排除 scaler 檔案)
    joblib_files = list(model_dir.glob("*.joblib"))
    traditional_models = [f for f in joblib_files if not f.name.endswith('_scaler.joblib')]
    
    for model_file in traditional_models:
        model_name = model_file.stem
        
        # 根據檔案名稱判斷類型
        if 'linear' in model_name.lower() or 'ridge' in model_name.lower() or 'lasso' in model_name.lower():
            category = 'linear'
        elif any(tree_type in model_name.lower() for tree_type in ['tree', 'xgboost', 'lightgbm', 'catboost', 'gradient']):
            category = 'tree'
        else:
            category = 'other'
        
        if category not in available_models:
            available_models[category] = []
        
        available_models[category].append({
            'type': 'traditional',
            'name': model_name,
            'file': model_file,
            'display_name': model_name.replace('best_', '').replace('_model_', ' - ')
        })
    
    # 🧠 尋找深度學習模型
    keras_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.endswith('_keras')]
    
    for keras_dir in keras_dirs:
        base_name = keras_dir.name.replace('_keras', '')
        scaler_file = model_dir / f"{base_name}_scaler.joblib"
        info_file = model_dir / f"{base_name}_info.txt"
        
        # 檢查深度學習模型檔案完整性
        if scaler_file.exists():
            if 'deep_learning' not in available_models:
                available_models['deep_learning'] = []
            
            available_models['deep_learning'].append({
                'type': 'deeplearning',
                'name': base_name,
                'keras_dir': keras_dir,
                'scaler_file': scaler_file,
                'info_file': info_file,
                'display_name': base_name.replace('_', ' ')
            })
    
    return available_models

def test_model_with_choice(X_test, test_df):
    """讓使用者選擇要測試的模型"""
    print("\n=== 選擇測試模型 ===")
    
    # 獲取可用模型
    available_models = get_available_models()
    
    if not available_models:
        print("❌ 沒有找到任何已訓練的模型")
        print("提示：請先執行訓練模式")
        return
    
    # 建立選項列表
    all_options = []
    option_counter = 1
    
    # 按類別顯示模型
    category_names = {
        'linear': '📈 線性模型',
        'tree': '🌳 樹模型',
        'deep_learning': '🧠 深度學習模型',
        'other': '📊 其他模型'
    }
    
    print("\n可用的模型:")
    print("0. 🔄 測試所有模型")
    
    for category, models in available_models.items():
        if models:  # 只顯示有模型的類別
            category_display = category_names.get(category, f'📁 {category.upper()}')
            print(f"\n{category_display}:")
            
            for model_info in models:
                print(f"{option_counter}. {model_info['display_name']}")
                all_options.append((category, model_info))
                option_counter += 1
    
    print(f"{option_counter}. 🚪 返回主選單")
    
    # 使用者選擇
    while True:
        try:
            choice = input(f"\n請選擇要測試的模型 (0-{option_counter}): ").strip()
            
            if choice == str(option_counter):  # 返回
                return
            elif choice == '0':  # 測試所有模型
                print("\n🔄 開始測試所有可用模型...")
                for category, model_info in all_options:
                    print(f"\n{'='*50}")
                    print(f"正在測試: {model_info['display_name']}")
                    print('='*50)
                    test_single_model(X_test, test_df, model_info)
                print(f"\n🎉 所有模型測試完成！")
                return
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(all_options):
                    category, model_info = all_options[choice_idx]
                    print(f"\n🎯 測試選定模型: {model_info['display_name']}")
                    test_single_model(X_test, test_df, model_info)
                    return
                else:
                    print("❌ 無效選擇，請重新輸入")
        except ValueError:
            print("❌ 請輸入有效數字")
        except KeyboardInterrupt:
            print("\n👋 操作已取消")
            return

def test_single_model(X_test, test_df, model_info):
    """測試單一模型"""
    try:
        if model_info['type'] == 'traditional':
            # 傳統模型
            print(f"📊 載入傳統模型: {model_info['file']}")
            import joblib
            model = joblib.load(model_info['file'])
            test_predictions = model.predict(X_test)
            
        elif model_info['type'] == 'deeplearning':
            # 深度學習模型
            print(f"🧠 載入深度學習模型:")
            print(f"   Keras 模型: {model_info['keras_dir']}")
            print(f"   Scaler: {model_info['scaler_file']}")
            
            import tensorflow as tf
            import joblib
            
            # 載入模型組件
            keras_model = tf.keras.models.load_model(str(model_info['keras_dir']))
            scaler = joblib.load(model_info['scaler_file'])
            
            # 預測
            X_scaled = scaler.transform(X_test)
            test_predictions = keras_model.predict(X_scaled, verbose=0).flatten()
            
        else:
            print(f"❌ 未知的模型類型: {model_info['type']}")
            return
        
        # 儲存預測結果
        save_predictions(test_predictions, test_df, model_info['name'])
        print(f"✅ 測試完成！預測樣本數: {len(test_predictions)}")
        
    except ImportError as e:
        print(f"❌ 套件載入失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

def test_single_model_file(X_test, test_df, model_file, model_name):
    """測試單一模型檔案 - 支援多種格式"""
    try:
        model_path = Path(model_file)
        
        # 🎯 檢查是否為深度學習模型
        if model_name.startswith('DeepLearning') or '_keras' in str(model_file):
            print(f"🧠 檢測到深度學習模型: {model_name}")
            
            # 尋找對應的檔案
            model_dir = model_path.parent
            base_name = model_name.replace('_keras', '')
            
            keras_path = model_dir / f"{base_name}_keras"
            scaler_path = model_dir / f"{base_name}_scaler.joblib"
            
            if not keras_path.exists():
                print(f"❌ 找不到 Keras 模型: {keras_path}")
                return
            
            if not scaler_path.exists():
                print(f"❌ 找不到 Scaler: {scaler_path}")
                return
            
            try:
                # 載入深度學習模型
                import tensorflow as tf
                import joblib
                
                print(f"   載入 Keras 模型: {keras_path}")
                keras_model = tf.keras.models.load_model(str(keras_path))
                
                print(f"   載入 Scaler: {scaler_path}")
                scaler = joblib.load(scaler_path)
                
                print("   進行預測...")
                X_scaled = scaler.transform(X_test)
                test_predictions = keras_model.predict(X_scaled, verbose=0).flatten()
                
            except ImportError:
                print("❌ TensorFlow 未安裝，無法使用深度學習模型")
                return
            except Exception as e:
                print(f"❌ 深度學習模型載入失敗: {e}")
                return
        
        else:
            # 傳統模型（樹模型、線性模型）
            print(f"📊 載入傳統機器學習模型: {model_file}")
            import joblib
            
            model = joblib.load(model_file)
            print("   進行預測...")
            test_predictions = model.predict(X_test)
        
        print("儲存預測結果...")
        save_predictions(test_predictions, test_df, model_name)
        
        print(f"✅ 完成！預測樣本數: {len(test_predictions)}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

def train_treeModel(X_train, y_train, X_valid, y_valid):
    """訓練樹模型"""
    print("\n=== 開始樹模型訓練 ===")
    
    # 初始化模型
    models = RegressionModels(random_state=42)
    
    # 第一階段：基礎樹模型
    print("\n1. 第一階段：基礎樹模型...")
    basic_models = models.get_tree_models()
    models.train_multiple_models(basic_models, X_train, y_train, X_valid, y_valid)
    
    # 顯示第一階段結果
    print("\n=== 第一階段結果 ===")
    results_stage1 = models.get_results_summary()
    print(results_stage1)
    
    # 第二階段：優化樹模型
    print("\n2. 第二階段：優化樹模型...")
    optimized_models = models.get_optimized_tree_models()
    models.train_multiple_models(optimized_models, X_train, y_train, X_valid, y_valid)
    
    # 顯示最終結果
    print("\n=== 樹模型訓練結果 ===")
    final_results = models.get_results_summary()
    print(final_results)
    
    # 儲存最佳樹模型
    timestamp = get_timestamp()
    best_models = models.save_best_models_by_type("../models", timestamp)
    
    return models, best_models

def train_LinearModel(X_train, y_train, X_valid, y_valid):
    """訓練線性模型"""
    print("\n=== 開始線性模型訓練 ===")
    
    # 初始化模型
    models = RegressionModels(random_state=42)
    
    # 訓練線性模型
    print("\n1. 訓練線性模型...")
    linear_models = models.get_linear_models()
    models.train_multiple_models(linear_models, X_train, y_train, X_valid, y_valid)
    
    # 顯示結果
    print("\n=== 線性模型訓練結果 ===")
    results = models.get_results_summary()
    print(results)
    
    # 儲存最佳線性模型
    timestamp = get_timestamp()
    best_models = models.save_best_models_by_type("../models", timestamp)
    
    return models, best_models

def train_DLModel(X_train, y_train, X_valid, y_valid):
    """訓練深度學習模型"""
    print("\n=== 開始深度學習模型訓練 ===")
    
    # 檢查 TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0
        print(f"🔍 GPU 狀態: {'可用' if gpu_available else '不可用'} ({len(gpus)} 個 GPU)")
    except ImportError:
        print("⚠️ TensorFlow 未安裝，無法訓練深度學習模型")
        return None, {}
    
    # 獲取深度學習模型工廠函數
    models = RegressionModels()
    dl_model_factory = models.get_deep_learning_models()
    
    if not dl_model_factory:
        print("❌ 無法獲取深度學習模型")
        return None, {}
    
    # 使用實際的特徵數量創建模型
    input_dim = X_train.shape[1]
    print(f"📊 訓練數據形狀: {X_train.shape}, 特徵數: {input_dim}")
    
    dl_models = dl_model_factory(input_dim)
    
    print("1. 訓練深度學習模型...")
    print("⚠️  注意：深度學習模型需要較長時間訓練...")
    
    results = []
    best_score = float('inf')
    best_model = None
    best_model_name = None
    
    for name, model in dl_models.items():
        print(f"\n訓練 {name}...")
        
        try:
            # 訓練模型
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 驗證模型
            y_pred = model.predict(X_valid)
            
            # 計算指標
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np
            
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            mae = mean_absolute_error(y_valid, y_pred)
            r2 = r2_score(y_valid, y_pred)
            mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
            
            results.append({
                'Model': name,
                'Valid_RMSE': rmse,
                'Valid_MAE': mae, 
                'Valid_R2': r2,
                'Valid_MAPE': mape,
                'Training_Time(s)': training_time
            })
            
            print(f"✅ {name} 訓練完成")
            print(f"   RMSE: {rmse:,.0f}")
            print(f"   R²: {r2:.4f}")
            print(f"   訓練時間: {training_time:.1f} 秒")
            
            # 記錄最佳模型
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_model_name = name
                
        except Exception as e:
            print(f"❌ {name} 訓練失敗: {str(e)}")
    
    # 顯示結果
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n📊 深度學習模型訓練結果:")
        print(results_df.to_string(index=False))
        
        # 儲存最佳模型
        if best_model is not None:
            # 🕐 生成帶時間戳記的模型名稱
            timestamp = get_timestamp()
            timestamped_model_name = f"{best_model_name}_{timestamp}"
            
            print(f"\n💾 儲存最佳深度學習模型: {timestamped_model_name}")
            try:
                model_dir = Path("../models")
                model_dir.mkdir(exist_ok=True)
                
                # 🎯 使用 TensorFlow 原生格式儲存
                if hasattr(best_model, 'model') and hasattr(best_model, 'scaler'):
                    keras_model = best_model.model
                    scaler = best_model.scaler
                    
                    # 儲存 Keras 模型
                    tf_model_path = model_dir / f"{timestamped_model_name}_keras"
                    keras_model.save(str(tf_model_path))
                    print(f"✅ Keras 模型已儲存: {tf_model_path}")
                    
                    # 儲存 Scaler
                    import joblib
                    scaler_path = model_dir / f"{timestamped_model_name}_scaler.joblib"
                    joblib.dump(scaler, scaler_path)
                    print(f"✅ Scaler 已儲存: {scaler_path}")
                    
                    # 儲存模型資訊
                    info_path = model_dir / f"{timestamped_model_name}_info.txt"
                    with open(info_path, 'w', encoding='utf-8') as f:
                        f.write(f"Model: {timestamped_model_name}\n")
                        f.write(f"Original_Name: {best_model_name}\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"Training_Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Type: DeepLearning\n")
                        f.write(f"Input_dim: {best_model.input_dim}\n")
                        f.write(f"RMSE: {best_score:.2f}\n")
                        f.write(f"Keras_model: {timestamped_model_name}_keras\n")
                        f.write(f"Scaler: {timestamped_model_name}_scaler.joblib\n")
                    print(f"✅ 模型資訊已儲存: {info_path}")
                    
                    print("   格式: TensorFlow 原生格式")
                    print(f"   時間戳記: {timestamp}")
                    print("   測試: 需要特殊載入方法")
                else:
                    print("❌ 模型格式不正確，無法儲存")
                    
            except Exception as e:
                print(f"❌ 深度學習模型儲存失敗: {str(e)}")
        
        return models, {'deep_learning': {'name': best_model_name, 'timestamped_name': timestamped_model_name, 'model': best_model, 'rmse': best_score, 'timestamp': timestamp}}
    else:
        print("❌ 沒有成功訓練的深度學習模型")
        return None, {}

def get_main_mode():
    """取得主要模式選擇"""
    print("\n請選擇模式:")
    print("1. Train (訓練模型)")
    print("2. Test (測試預測)")
    print("0. 退出")
    
    while True:
        choice = input("\n請輸入選擇 (0-2): ").strip()
        if choice == '0':
            return 'exit'
        elif choice == '1':
            return 'train'
        elif choice == '2':
            return 'test'
        else:
            print("❌ 無效選擇，請重新輸入")

def get_train_mode():
    """取得訓練模式選擇"""
    print("\n請選擇訓練模式:")
    print("1. 樹模型 (Tree Models)")
    print("2. 線性模型 (Linear Models)")
    print("3. 深度學習模型 (Deep Learning)")
    print("4. 全部訓練 (All Models)")
    print("0. 返回主選單")
    
    while True:
        choice = input("\n請輸入選擇 (0-4): ").strip()
        if choice == '0':
            return 'back'
        elif choice == '1':
            return 'tree'
        elif choice == '2':
            return 'linear'
        elif choice == '3':
            return 'dl'
        elif choice == '4':
            return 'all'
        else:
            print("❌ 無效選擇，請重新輸入")

def train_all_models(X_train, y_train, X_valid, y_valid):
    """訓練所有類型的模型"""
    print("\n=== 開始訓練所有模型 ===")
    
    all_models = RegressionModels(random_state=42)
    
    # 1. 樹模型
    print("\n🌳 第一階段：樹模型...")
    tree_models = all_models.get_tree_models()
    all_models.train_multiple_models(tree_models, X_train, y_train, X_valid, y_valid)
    
    optimized_tree_models = all_models.get_optimized_tree_models()
    all_models.train_multiple_models(optimized_tree_models, X_train, y_train, X_valid, y_valid)
    
    # 2. 線性模型
    print("\n📊 第二階段：線性模型...")
    linear_models = all_models.get_linear_models()
    all_models.train_multiple_models(linear_models, X_train, y_train, X_valid, y_valid)
    
    # 3. 深度學習模型
    print("\n🧠 第三階段：深度學習模型...")
    try:
        dl_models = all_models.get_deep_learning_models()
        if dl_models:
            all_models.train_multiple_models(dl_models, X_train, y_train, X_valid, y_valid)
        else:
            print("⚠️ 跳過深度學習模型（TensorFlow 未安裝）")
    except Exception as e:
        print(f"⚠️ 深度學習模型訓練失敗: {e}")
    
    # 顯示最終結果
    print("\n=== 所有模型訓練結果 ===")
    results = all_models.get_results_summary()
    print(results)
    
    # 儲存最佳模型
    timestamp = get_timestamp()
    best_models = all_models.save_best_models_by_type("../models", timestamp)
    
    return all_models, best_models

def main_menu():
    """主選單流程"""
    while True:
        mode = get_main_mode()
        
        if mode == 'exit':
            print("感謝使用！")
            break
        
        elif mode == 'train':
            # 載入資料
            data = load_data_and_prepare()
            if data[0] is None:
                print("❌ 資料載入失敗")
                continue
            
            X_train, y_train, X_valid, y_valid, X_test, test_df = data
            
            # 選擇訓練模式
            train_mode = get_train_mode()
            
            if train_mode == 'back':
                continue
            elif train_mode == 'tree':
                train_treeModel(X_train, y_train, X_valid, y_valid)
            elif train_mode == 'linear':
                train_LinearModel(X_train, y_train, X_valid, y_valid)
            elif train_mode == 'dl':
                train_DLModel(X_train, y_train, X_valid, y_valid)
            elif train_mode == 'all':
                train_all_models(X_train, y_train, X_valid, y_valid)
        
        elif mode == 'test':
            # 載入資料
            data = load_data_and_prepare()
            model_name = input("請輸入要測試的模型名稱 (留空表示自動選擇最新模型): ").strip()
            if data[0] is None:
                print("❌ 資料載入失敗")
                continue
            
            X_train, y_train, X_valid, y_valid, X_test, test_df = data
            test_model(X_test, test_df, model_name)

if __name__ == "__main__":
    print("🏠 房地產價格預測系統")
    print("=" * 50)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n程序已中止")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n程序結束")
