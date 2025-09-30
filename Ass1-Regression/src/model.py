import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RegressionModels:
    """回歸模型集合類"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def get_tree_models(self):
        """取得樹模型配置"""
        return {          
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            ),
            
            'CatBoost': cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False,
                thread_count=-1
            ),
        }
    
    def get_linear_models(self):
        """取得線性模型配置"""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=1.0, random_state=self.random_state)
        }
    
    def get_optimized_tree_models(self):
        """取得優化後的樹模型配置（更好的參數）"""
        return {
            'XGBoost_Optimized': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'LightGBM_Optimized': lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            ),
        }
    
    def calculate_metrics(self, y_true, y_pred, model_name=""):
        """計算評估指標"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 計算 MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics
    
    def train_single_model(self, model, model_name, X_train, y_train, X_valid, y_valid):
        """訓練單一模型"""
        import time
        
        print(f"\n🔄 開始訓練 {model_name}...")
        start_time = time.time()
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 預測
        print(f"   ⏱️  訓練時間: {training_time:.2f} 秒")
        print(f"   📊 進行預測...")
        
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)
        
        # 計算指標
        train_metrics = self.calculate_metrics(y_train, train_pred, f"{model_name}_Train")
        valid_metrics = self.calculate_metrics(y_valid, valid_pred, f"{model_name}_Valid")
        
        # 儲存結果
        self.trained_models[model_name] = model
        self.results[model_name] = {
            'train': train_metrics,
            'valid': valid_metrics,
            'train_pred': train_pred,
            'valid_pred': valid_pred,
            'training_time': training_time
        }
        
        # 詳細輸出結果
        print(f"   ✅ {model_name} 訓練完成!")
        print(f"      📈 驗證集 RMSE: {valid_metrics['RMSE']:.2f}")
        print(f"      📈 驗證集 R²: {valid_metrics['R2']:.4f}")
        print(f"      📈 驗證集 MAE: {valid_metrics['MAE']:.2f}")
        if training_time > 60:
            print(f"      ⏱️  耗時: {training_time/60:.1f} 分鐘")
        else:
            print(f"      ⏱️  耗時: {training_time:.1f} 秒")
        
        return model
    
    def get_deep_learning_models(self):
        """取得深度學習模型配置"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.preprocessing import StandardScaler
            
            print(f"🔍 TensorFlow 版本: {tf.__version__}")
            
            # 檢查 GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🚀 檢測到 {len(gpus)} 個 GPU")
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("✅ GPU 記憶體增長已啟用")
                except RuntimeError:
                    pass  # GPU 已經初始化
            else:
                print("使用 CPU 訓練")
            
            # 簡化的神經網絡包裝器
            class KerasRegressorWrapper:
                def __init__(self, input_dim, epochs=50, batch_size=32): #CPU batch_size=16
                    self.input_dim = input_dim
                    self.epochs = epochs
                    self.batch_size = batch_size
                    self.model = None
                    self.scaler = StandardScaler()
                
                def _build_model(self):
                    model = Sequential([
                            Dense(256, activation='relu', input_shape=(self.input_dim,)),
                            Dropout(0.3),
                            Dense(512, activation='relu'),
                            Dropout(0.3),
                            Dense(256, activation='relu'),
                            Dropout(0.3),
                            Dense(128, activation='relu'),
                            Dropout(0.2),
                            Dense(64, activation='relu'),
                            Dense(1, activation='linear')
                        ])

                    # # V3: BatchNorm 
                    # model = Sequential([
                    #     Dense(512, input_shape=(self.input_dim,)),
                    #     BatchNormalization(),
                    #     Activation('relu'),
                    #     Dropout(0.3),
                        
                    #     Dense(256),
                    #     BatchNormalization(), 
                    #     Activation('relu'),
                    #     Dropout(0.3),
                        
                    #     Dense(128),
                    #     BatchNormalization(),
                    #     Activation('relu'), 
                    #     Dropout(0.2),
                        
                    #     Dense(64),
                    #     BatchNormalization(),
                    #     Activation('relu'),
                        
                    #     Dense(1, activation='linear')
                    # ])
                    
                    
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='mse',
                        metrics=['mae']
                    )
                    return model
                
                def fit(self, X, y):
                    print(f"🧠 開始訓練神經網絡 - 樣本數: {len(X)}, 特徵數: {X.shape[1]}")
                    
                    # 標準化特徵
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # 建立模型
                    self.model = self._build_model()
                    
                    # 設定回調函數
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                    ]
                    
                    # 訓練模型
                    self.model.fit(
                        X_scaled, y,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    return self
                
                def predict(self, X):
                    if self.model is None:
                        raise ValueError("模型尚未訓練")
                    
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict(X_scaled, verbose=0).flatten()
            
            # 創建一個工廠函數，在訓練時動態設定 input_dim
            def create_dl_models(input_dim):
                return {
                    'DeepLearning_NN': KerasRegressorWrapper(
                        input_dim=input_dim,
                        epochs=100,
                        batch_size=64
                    )
                }
            
            return create_dl_models
            
        except ImportError as e:
            print("⚠️ TensorFlow 未安裝，跳過深度學習模型")
            print(f"   錯誤: {e}")
            return {}
        except Exception as e:
            print(f"❌ 深度學習模型初始化失敗: {e}")
            return {}
    
    def train_multiple_models(self, models_dict, X_train, y_train, X_valid, y_valid):
        """訓練多個模型"""
        import time
        
        total_models = len(models_dict)
        print(f"\n{'='*60}")
        print(f"🚀 開始訓練 {total_models} 個模型")
        print(f"{'='*60}")
        
        total_start_time = time.time()
        successful_models = 0
        failed_models = []
        
        for i, (model_name, model) in enumerate(models_dict.items(), 1):
            print(f"\n📊 進度: {i}/{total_models}")
            try:
                # 特殊處理深度學習模型
                if 'DeepLearning' in model_name and hasattr(model, 'input_dim'):
                    if model.input_dim is None:
                        model.input_dim = X_train.shape[1]
                
                self.train_single_model(model, model_name, X_train, y_train, X_valid, y_valid)
                successful_models += 1
                
            except Exception as e:
                print(f"   ❌ {model_name} 訓練失敗: {e}")
                failed_models.append(model_name)
                if 'DeepLearning' in model_name:
                    print("     💡 提示：深度學習模型需要 TensorFlow，可跳過此模型")
                else:
                    import traceback
                    print("     🔍 詳細錯誤:")
                    traceback.print_exc()
        
        # 總結
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"🎉 模型訓練完成!")
        print(f"✅ 成功: {successful_models}/{total_models} 個模型")
        if failed_models:
            print(f"❌ 失敗: {len(failed_models)} 個模型 ({', '.join(failed_models)})")
        
        if total_time > 60:
            print(f"⏱️  總耗時: {total_time/60:.1f} 分鐘")
        else:
            print(f"⏱️  總耗時: {total_time:.1f} 秒")
        
        if successful_models > 0:
            avg_time = total_time / successful_models
            print(f"📊 平均每模型: {avg_time:.1f} 秒")
        print(f"{'='*60}")
    
    def get_results_summary(self):
        """取得結果摘要"""
        summary_data = []
        
        for model_name, result in self.results.items():
            valid_metrics = result['valid']
            training_time = result.get('training_time', 0)
            
            summary_data.append({
                'Model': model_name,
                'Valid_RMSE': valid_metrics['RMSE'],
                'Valid_MAE': valid_metrics['MAE'],
                'Valid_R2': valid_metrics['R2'],
                'Valid_MAPE': valid_metrics['MAPE'],
                'Training_Time(s)': training_time
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Valid_RMSE')
        
        # 美化輸出
        print(f"\n{'='*80}")
        print(f"📊 模型性能排行榜 (依驗證集 RMSE 排序)")
        print(f"{'='*80}")
        
        for i, row in df_summary.iterrows():
            rank = df_summary.index.get_loc(i) + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            
            print(f"{medal} {row['Model']}")
            print(f"    RMSE: {row['Valid_RMSE']:.2f} | R²: {row['Valid_R2']:.4f} | MAE: {row['Valid_MAE']:.2f}")
            print(f"    MAPE: {row['Valid_MAPE']:.2f}% | 訓練時間: {row['Training_Time(s)']:.1f}s")
            print()
        
        return df_summary
    
    def save_best_model(self, save_dir="models"):
        """儲存最佳模型"""
        if not self.results:
            print("沒有訓練過的模型!")
            return None, None
        
        # 找出最佳模型（Valid RMSE 最低）
        best_model_name = min(self.results.keys(), 
                            key=lambda x: self.results[x]['valid']['RMSE'])
        
        best_model = self.trained_models[best_model_name]
        
        # 建立儲存目錄
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 儲存模型
        model_file = save_path / f"best_model_{best_model_name}.joblib"
        joblib.dump(best_model, model_file)
        
        print(f"最佳模型 {best_model_name} 已儲存到 {model_file}")
        
        return best_model_name, best_model
    
    def save_best_models_by_type(self, save_dir="models", timestamp=None):
        """分類別儲存最佳模型"""
        if not self.results:
            print("沒有訓練過的模型!")
            return {}
        
        # 建立儲存目錄
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 如果沒有提供時間戳記，生成一個
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%m%d_%H%M")
        
        # 分類模型
        model_categories = {
            'tree': [],
            'linear': [],
            'deep_learning': []
        }
        
        for model_name in self.results.keys():
            if any(tree_type in model_name.lower() for tree_type in 
                   ['randomforest', 'xgboost', 'lightgbm', 'catboost', 'gradient']):
                model_categories['tree'].append(model_name)
            elif any(linear_type in model_name.lower() for linear_type in 
                     ['linear', 'ridge', 'lasso', 'elastic']):
                model_categories['linear'].append(model_name)
            elif 'deeplearning' in model_name.lower() or 'neural' in model_name.lower():
                model_categories['deep_learning'].append(model_name)
        
        best_models = {}
        
        # 為每個類別找出最佳模型並儲存
        for category, model_names in model_categories.items():
            if not model_names:
                continue
                
            # 找出該類別的最佳模型
            best_in_category = min(model_names, 
                                 key=lambda x: self.results[x]['valid']['RMSE'])
            
            best_model = self.trained_models[best_in_category]
            
            # 🕐 生成帶時間戳記的檔案名稱
            timestamped_filename = f"best_{category}_model_{best_in_category}_{timestamp}.joblib"
            model_file = save_path / timestamped_filename
            joblib.dump(best_model, model_file)
            
            best_models[category] = {
                'name': best_in_category,
                'timestamped_name': f"{best_in_category}_{timestamp}",
                'model': best_model,
                'rmse': self.results[best_in_category]['valid']['RMSE'],
                'r2': self.results[best_in_category]['valid']['R2'],
                'file': model_file,
                'timestamp': timestamp
            }
            
            print(f"最佳{category}模型 {best_in_category} 已儲存到 {model_file}")
            print(f"  RMSE: {self.results[best_in_category]['valid']['RMSE']:.2f}")
            print(f"  R²: {self.results[best_in_category]['valid']['R2']:.4f}")
            print(f"  時間戳記: {timestamp}")
        
        # 找出整體最佳模型
        if best_models:
            overall_best_category = min(best_models.keys(), 
                                      key=lambda x: best_models[x]['rmse'])
            overall_best = best_models[overall_best_category]
            
            print(f"\n🏆 整體最佳模型: {overall_best['name']} ({overall_best_category})")
            print(f"   RMSE: {overall_best['rmse']:.2f}")
            print(f"   R²: {overall_best['r2']:.4f}")
            
            # 儲存整體最佳模型的副本
            overall_best_file = save_path / f"overall_best_model_{overall_best['name']}.joblib"
            joblib.dump(overall_best['model'], overall_best_file)
            print(f"   檔案: {overall_best_file}")
            
            best_models['overall_best'] = overall_best
        
        return best_models
    
    def plot_predictions(self, model_names=None, figsize=(15, 10)):
        """繪製預測結果圖"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        n_models = len(model_names)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, model_name in enumerate(model_names):
            if model_name not in self.results:
                continue
                
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            result = self.results[model_name]
            y_true = result['valid']['y_true'] if 'y_true' in result['valid'] else None
            y_pred = result['valid_pred']
            
            if y_true is not None:
                ax.scatter(y_true, y_pred, alpha=0.5)
                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predictions')
                ax.set_title(f'{model_name}\nR² = {result["valid"]["R2"]:.4f}')
        
        plt.tight_layout()
        plt.show()
    
    def predict_test(self, model_name, X_test):
        """使用指定模型預測測試集"""
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 尚未訓練!")
        
        model = self.trained_models[model_name]
        test_pred = model.predict(X_test)
        
        return test_pred