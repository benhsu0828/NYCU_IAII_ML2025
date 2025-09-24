#!/usr/bin/env python3
"""
測試深度學習模型儲存功能
用於快速驗證模型包裝器是否能正確序列化，避免長時間訓練後才發現儲存問題
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 加入 src 目錄到路徑
sys.path.append(str(Path(__file__).parent))

from model import RegressionModels

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

def test_dl_model_save():
    """測試深度學習模型儲存功能"""
    
    print("🧪 開始測試深度學習模型儲存功能...")
    
    try:
        # 1. 檢查 TensorFlow
        import tensorflow as tf
        print(f"✅ TensorFlow 版本: {tf.__version__}")
        
        # 2. 創建假資料進行快速測試
        print("\n📊 創建測試資料...")
        np.random.seed(42)
        n_samples = 100  # 少量樣本，快速訓練
        n_features = 20  # 較少特徵
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randn(n_samples) * 100 + 1000  # 模擬房價
        
        print(f"   訓練資料形狀: {X_train.shape}")
        print(f"   目標值範圍: {y_train.min():.0f} ~ {y_train.max():.0f}")
        
        # 3. 獲取深度學習模型
        print("\n🤖 初始化深度學習模型...")
        models = RegressionModels()
        dl_model_factory = models.get_deep_learning_models()
        
        if not dl_model_factory:
            print("❌ 無法獲取深度學習模型工廠")
            return False
        
        # 4. 創建模型
        input_dim = X_train.shape[1]
        dl_models = dl_model_factory(input_dim)
        
        if not dl_models:
            print("❌ 無法創建深度學習模型")
            return False
        
        model_name = list(dl_models.keys())[0]
        model = dl_models[model_name]
        
        print(f"✅ 模型創建成功: {model_name}")
        print(f"   輸入維度: {input_dim}")
        
        # 5. 快速訓練（僅 5 epochs）
        print(f"\n🔄 開始快速訓練（僅 5 epochs）...")
        
        # 暫時修改訓練參數
        original_epochs = model.epochs
        model.epochs = 5  # 只訓練 5 epochs
        
        start_time = pd.Timestamp.now()
        model.fit(X_train, y_train)
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        print(f"✅ 訓練完成，耗時: {training_time:.1f} 秒")
        
        # 恢復原始設定
        model.epochs = original_epochs
        
        # 6. 測試模型預測
        print(f"\n🔮 測試模型預測...")
        X_test = np.random.randn(10, n_features)  # 10 個測試樣本
        predictions = model.predict(X_test)
        
        print(f"✅ 預測成功，預測值範圍: {predictions.min():.0f} ~ {predictions.max():.0f}")
        
        # 7. 測試模型包裝器創建
        print(f"\n📦 測試模型包裝器創建...")
        model_dir = Path("../models")
        model_dir.mkdir(exist_ok=True)
        
        wrapper = create_dl_model_wrapper(model, model_name, model_dir)
        
        if wrapper is None:
            print("❌ 包裝器創建失敗")
            return False
        
        # 8. 測試包裝器預測
        print(f"\n🧪 測試包裝器預測功能...")
        wrapper_predictions = wrapper.predict(X_test)
        
        # 檢查預測是否一致
        diff = np.abs(predictions - wrapper_predictions).max()
        print(f"   原始模型 vs 包裝器最大差異: {diff:.6f}")
        
        if diff < 1e-5:
            print("✅ 包裝器預測與原始模型一致")
        else:
            print("⚠️ 包裝器預測與原始模型有差異")
        
        # 9. 測試 TensorFlow 原生格式儲存
        print(f"\n💾 測試 TensorFlow 原生格式儲存...")
        
        try:
            # 提取模型組件
            if hasattr(model, 'model') and hasattr(model, 'scaler'):
                keras_model = model.model
                scaler = model.scaler
                
                # 儲存路徑
                keras_path = model_dir / f"{model_name}_keras"
                scaler_path = model_dir / f"{model_name}_scaler.joblib"
                info_path = model_dir / f"{model_name}_info.txt"
                
                print(f"   Keras 模型路徑: {keras_path}")
                print(f"   Scaler 路徑: {scaler_path}")
                
                # 儲存 Keras 模型
                keras_model.save(str(keras_path))
                print("✅ Keras 模型儲存成功")
                
                # 儲存 Scaler
                import joblib
                joblib.dump(scaler, scaler_path)
                print("✅ Scaler 儲存成功")
                
                # 儲存資訊檔
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Type: DeepLearning\n")
                    f.write(f"Input_dim: {model.input_dim}\n")
                print("✅ 模型資訊儲存成功")
                
                # 測試載入
                print(f"\n🔄 測試模型載入...")
                loaded_keras_model = tf.keras.models.load_model(str(keras_path))
                loaded_scaler = joblib.load(scaler_path)
                print("✅ 模型載入成功")
                
                # 測試載入的模型預測
                print(f"\n🧪 測試載入模型預測...")
                X_scaled = loaded_scaler.transform(X_test)
                loaded_predictions = loaded_keras_model.predict(X_scaled, verbose=0).flatten()
                
                # 檢查預測是否一致
                load_diff = np.abs(predictions - loaded_predictions).max()
                print(f"   載入模型 vs 原始模型最大差異: {load_diff:.6f}")
                
                if load_diff < 1e-5:
                    print("✅ 載入模型預測與原始模型一致")
                    
                    # 清理測試檔案
                    import shutil
                    if keras_path.exists():
                        shutil.rmtree(keras_path)
                    if scaler_path.exists():
                        scaler_path.unlink()
                    if info_path.exists():
                        info_path.unlink()
                    print("🗑️ 清理測試檔案")
                    
                    return True
                else:
                    print("❌ 載入模型預測與原始模型不一致")
                    return False
            else:
                print("❌ 模型格式不正確")
                return False
                
        except Exception as save_error:
            print(f"❌ TensorFlow 格式儲存失敗: {save_error}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ TensorFlow 未安裝: {e}")
        return False
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("=" * 60)
    print("🧪 深度學習模型儲存功能測試")
    print("=" * 60)
    
    success = test_dl_model_save()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 測試通過！深度學習模型可以正確儲存和載入")
        print("💡 現在可以安全地進行完整訓練了")
    else:
        print("❌ 測試失敗！需要修復儲存功能")
        print("💡 建議先解決儲存問題再進行完整訓練")
    print("=" * 60)

if __name__ == "__main__":
    main()