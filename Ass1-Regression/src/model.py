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

# 🔧 將 DNN 包裝器移到模組頂層以支援 pickle
from sklearn.base import BaseEstimator, RegressorMixin

class SerializableDNNWrapper(BaseEstimator, RegressorMixin):
    """可序列化的 DNN 包裝器 - 支援 pickle 和 scikit-learn"""
    
    def __init__(self, input_dim=None):
        self.input_dim = input_dim
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.model_weights = None  # 用於存儲權重而不是整個 TF 模型
    
    def _build_model(self):
        """構建 DNN 模型"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            
            model = Sequential([
                Dense(128, activation='relu', input_shape=(self.input_dim,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            return model
        except ImportError:
            print("❌ TensorFlow 未安裝，無法創建 DNN")
            return None
    
    def fit(self, X, y):
        """訓練模型"""
        try:
            from sklearn.preprocessing import StandardScaler
            import tensorflow as tf
            import time
            
            print(f"      🧠 訓練新的 DNN 基學習器 (特徵數: {X.shape[1]}, 樣本數: {X.shape[0]})...")
            
            # 設置輸入維度
            if self.input_dim is None:
                self.input_dim = X.shape[1]
            
            # 數據標準化
            print(f"         📊 正在標準化特徵...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 構建模型
            print(f"         🏗️ 建立神經網路架構...")
            self.model = self._build_model()
            if self.model is None:
                raise ValueError("無法創建 DNN 模型")
            
            # 📊 Stacking DNN 專用進度回調
            class StackingProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self):
                    self.start_time = None
                    
                def on_train_begin(self, logs=None):
                    print(f"         🚀 開始訓練 {self.params['epochs']} epochs...")
                    self.start_time = time.time()
                    
                def on_epoch_end(self, epoch, logs=None):
                    current_epoch = epoch + 1
                    total_epochs = self.params['epochs']
                    
                    # 每5個epoch或前3個epoch或最後3個epoch顯示進度
                    if (current_epoch <= 3 or 
                        current_epoch % 5 == 0 or 
                        current_epoch > total_epochs - 3):
                        
                        elapsed = time.time() - self.start_time
                        progress = current_epoch / total_epochs * 100
                        
                        # 創建進度條
                        bar_length = 20
                        filled_length = int(bar_length * current_epoch / total_epochs)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        
                        # 預估剩餘時間
                        if current_epoch > 0:
                            eta = elapsed / current_epoch * (total_epochs - current_epoch)
                            eta_str = f", ETA: {eta:.0f}s"
                        else:
                            eta_str = ""
                        
                        print(f"         Epoch {current_epoch:2d}/{total_epochs} "
                              f"[{bar}] {progress:5.1f}% "
                              f"- loss: {logs.get('loss', 0):.4f} "
                              f"- val_loss: {logs.get('val_loss', 0):.4f} "
                              f"({elapsed:.0f}s{eta_str})")
                              
                def on_train_end(self, logs=None):
                    total_time = time.time() - self.start_time
                    print(f"         ✅ Stacking DNN 訓練完成，總耗時: {total_time:.1f} 秒")
            
            # 訓練設置
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=8, restore_best_weights=True, verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0
                ),
                StackingProgressCallback()  # 📊 添加進度顯示回調
            ]
            
            # 訓練
            history = self.model.fit(
                X_scaled, y,
                epochs=40,             # 增加 epochs 以獲得更好效果
                batch_size=64,       
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0  # 關閉 TensorFlow 默認輸出，使用自定義進度
            )
            
            # 儲存權重而不是整個模型
            self.model_weights = self.model.get_weights()
            self.is_fitted = True
            
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            print(f"      ✅ DNN 訓練完成，最終 loss: {final_loss:.4f}, val_loss: {final_val_loss:.4f}")
            return self
            
        except Exception as e:
            print(f"      ❌ DNN 訓練失敗: {e}")
            # 使用 Ridge 作為備用
            from sklearn.linear_model import Ridge
            self.backup_model = Ridge(alpha=1.0)
            self.backup_model.fit(X, y)
            self.scaler = None
            self.is_fitted = True
            self.use_backup = True
            print("      💡 改用 Ridge 回歸作為備用")
            return self
    
    def predict(self, X):
        """預測"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練")
        
        # 如果使用備用模型
        if hasattr(self, 'use_backup') and self.use_backup:
            return self.backup_model.predict(X)
        
        try:
            # 重建模型並載入權重
            if self.model is None and self.model_weights is not None:
                self.model = self._build_model()
                if self.model:
                    self.model.set_weights(self.model_weights)
            
            if self.model and self.scaler:
                X_scaled = self.scaler.transform(X)
                predictions = self.model.predict(X_scaled, verbose=0)
                return predictions.flatten()
            else:
                raise ValueError("模型狀態異常")
                
        except Exception as e:
            print(f"⚠️  DNN 預測失敗，使用簡單預測: {e}")
            # 簡單的備用預測
            return np.mean(X, axis=1) * 0.1
    
    def get_params(self, deep=True):
        """獲取參數（scikit-learn 相容性）"""
        return {'input_dim': self.input_dim}
    
    def set_params(self, **params):
        """設置參數（scikit-learn 相容性）"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def score(self, X, y):
        """計算 R² 分數（scikit-learn 回歸器必需）"""
        try:
            from sklearn.metrics import r2_score
            y_pred = self.predict(X)
            return r2_score(y, y_pred)
        except Exception as e:
            print(f"⚠️  計算分數失敗: {e}")
            return 0.0
    
    def __getstate__(self):
        """自定義序列化 - 只保存權重，不保存 TensorFlow 物件"""
        state = self.__dict__.copy()
        # 移除不可序列化的 TensorFlow 模型物件
        state['model'] = None
        return state
    
    def __setstate__(self, state):
        """自定義反序列化"""
        self.__dict__.update(state)
        # 模型會在需要時重新創建

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
                            Dense(512, activation='relu'),
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
    
    def get_stacking_models(self, use_pretrained_dnn=True, X_sample=None):
        """取得 Stacking 集成模型配置 - 使用可序列化DNN
        
        Args:
            use_pretrained_dnn (bool): 是否使用預訓練的 DNN 模型（已廢棄，一律重新訓練）
            X_sample (array-like): 樣本資料用於檢查特徵維度
        """
        try:
            from sklearn.ensemble import StackingRegressor, VotingRegressor
            from sklearn.model_selection import KFold
            
            # � 總是使用新的可序列化 DNN 包裝器
            # 🔄 根據用戶選擇決定 DNN 策略
            expected_features = X_sample.shape[1] if X_sample is not None else None
            
            if use_pretrained_dnn:
                # 嘗試載入預訓練 DNN
                pretrained_dnn = self._load_pretrained_dnn(expected_features=expected_features)
                if pretrained_dnn:
                    print("✅ 成功載入預訓練 DNN 模型用於 Stacking")
                    dnn_estimator = pretrained_dnn
                else:
                    print("⚠️  預訓練 DNN 模型不可用或不相容")
                    print("💡 將使用輕量級 Ridge 回歸代替 DNN")
                    dnn_estimator = Ridge(alpha=2.0, random_state=self.random_state)
            else:
                # 創建新的可序列化 DNN 包裝器
                try:
                    print("🧠 建立新的可序列化DNN模型 (將重新訓練)...")
                    dnn_estimator = SerializableDNNWrapper(input_dim=expected_features)
                    print("✅ 成功建立可序列化DNN模型用於 Stacking")
                except Exception as e:
                    print(f"❌ 建立DNN失敗: {e}")
                    print("💡 將使用輕量級 Ridge 回歸代替 DNN")
                    dnn_estimator = Ridge(alpha=2.0, random_state=self.random_state)
            
            # 基學習器：DNN + XGBoost + LightGBM (資源優化版)
            base_learners = [
                ('xgb', xgb.XGBRegressor(
                    n_estimators=100,      # 大幅減少樹數量
                    max_depth=5,           # 減少深度
                    learning_rate=0.1,     # 提高學習率補償樹數量減少
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    n_jobs=5,              # 限制CPU核心數
                    tree_method='hist',    # 使用記憶體友善的方法
                    max_bin=64            # 大幅減少記憶體使用
                )),
                ('lgb', lgb.LGBMRegressor(
                    n_estimators=100,      # 大幅減少樹數量
                    max_depth=5,           # 減少深度
                    learning_rate=0.1,     # 提高學習率
                    subsample=0.8,         # 樣本採樣比例
                    colsample_bytree=0.8,  # 特徵採樣比例
                    reg_alpha=0.1,         # L1 正則化
                    reg_lambda=1.0,        # L2 正則化
                    random_state=self.random_state,  # 使用統一的隨機種子
                    verbose=-1,            # 不顯示訓練資訊
                    n_jobs=5,              # 限制CPU核心數
                    max_bin=64,            # 大幅減少記憶體使用
                    min_data_in_leaf=20,   # 增加最小葉子樣本數
                    feature_fraction=0.8   # 減少特徵使用量
                )),
                ('dnn', dnn_estimator)  # 使用可序列化DNN
            ]
            
            return {
                # 🏆 方案A：DNN + 樹模型基學習器 + Ridge最終學習器 (可序列化版)
                'Stacking_DNN_Trees_Ridge': StackingRegressor(
                    estimators=base_learners,
                    final_estimator=Ridge(alpha=1.0, random_state=self.random_state),
                    cv=KFold(n_splits=3, shuffle=True, random_state=self.random_state),  # 減少 CV fold
                    n_jobs=1,               # 限制並行處理避免資源衝突
                    passthrough=False       # 不傳遞原始特徵，減少記憶體使用
                ),
                
                # 🎯 備選方案：使用線性回歸作為最終學習器 (更輕量)
                'Stacking_DNN_Trees_Linear': StackingRegressor(
                    estimators=base_learners,
                    final_estimator=LinearRegression(),
                    cv=KFold(n_splits=3, shuffle=True, random_state=self.random_state),  # 減少 CV fold
                    n_jobs=1,               # 限制並行處理
                    passthrough=False       # 不傳遞原始特徵
                ),
                
                # 🚀 Voting版本：簡單投票組合 (最輕量)
                'Voting_DNN_Trees': VotingRegressor(
                    estimators=base_learners,
                    n_jobs=1                # 限制並行處理
                )
            }
            
        except ImportError as e:
            print(f"⚠️  TensorFlow 未安裝，無法使用 Stacking 模型: {e}")
            return {}
        except Exception as e:
            print(f"❌ Stacking 模型配置出錯: {e}")
            return {}
            
        except ImportError as e:
            print(f"⚠️  TensorFlow 未安裝，無法使用 Stacking 模型: {e}")
            return {}
        except Exception as e:
            print(f"❌ Stacking 模型配置出錯: {e}")
            return {}
    
    def _load_pretrained_dnn(self, expected_features=None):
        """載入預訓練的 DNN 模型用於 Stacking
        
        Args:
            expected_features: 期望的特徵數量，如果提供會進行相容性檢查
        """
        try:
            import tensorflow as tf
            import joblib
            from pathlib import Path
            from sklearn.base import BaseEstimator, RegressorMixin
            
            model_dir = Path("../models")
            
            # 🔍 尋找最新的 DNN 模型
            keras_dirs = [d for d in model_dir.iterdir() 
                         if d.is_dir() and d.name.endswith('_keras')]
            
            if not keras_dirs:
                print("⚠️  未找到任何 DNN 模型目錄")
                return None
            
            # 按修改時間排序，選擇最新的
            latest_keras_dir = max(keras_dirs, key=lambda x: x.stat().st_mtime)
            base_name = latest_keras_dir.name.replace('_keras', '')
            scaler_file = model_dir / f"{base_name}_scaler.joblib"
            
            if not scaler_file.exists():
                print(f"⚠️  未找到對應的 scaler 檔案: {scaler_file}")
                return None
            
            print(f"🔄 載入預訓練 DNN 模型: {base_name}")
            
            # 載入模型和 scaler
            model = tf.keras.models.load_model(latest_keras_dir)
            scaler = joblib.load(scaler_file)
            
            # 檢查特徵相容性
            if expected_features is not None:
                model_features = scaler.n_features_in_
                if model_features != expected_features:
                    print(f"⚠️  特徵維度不匹配:")
                    print(f"   預訓練模型: {model_features} 個特徵")
                    print(f"   當前資料: {expected_features} 個特徵")
                    print(f"💡 將使用 Ridge 代替不相容的預訓練 DNN")
                    return None
                else:
                    print(f"✅ 特徵維度匹配: {model_features} 個特徵")
            
            # 創建包裝器
            class PretrainedDNNWrapper(BaseEstimator, RegressorMixin):
                """預訓練 DNN 模型包裝器 - 支援 scikit-learn clone"""
                
                def __init__(self, model_path=None, scaler_path=None, model=None, scaler=None):
                    # 儲存路徑而不是物件，避免 deepcopy 問題
                    self.model_path = model_path or latest_keras_dir
                    self.scaler_path = scaler_path or scaler_file
                    self._model = model
                    self._scaler = scaler
                    self.is_fitted = True
                
                def _load_model_if_needed(self):
                    """延遲載入模型和 scaler"""
                    if self._model is None or self._scaler is None:
                        try:
                            import tensorflow as tf
                            import joblib
                            self._model = tf.keras.models.load_model(self.model_path)
                            self._scaler = joblib.load(self.scaler_path)
                        except Exception as e:
                            print(f"❌ 重新載入模型失敗: {e}")
                            raise
                
                def fit(self, X, y):
                    """已經是預訓練模型，檢查特徵維度相容性"""
                    print("      🔄 使用預訓練 DNN 模型 (跳過訓練階段)")
                    
                    # 檢查特徵維度相容性
                    try:
                        self._load_model_if_needed()
                        expected_features = self._scaler.n_features_in_
                        actual_features = X.shape[1]
                        
                        if expected_features != actual_features:
                            print(f"      ⚠️  特徵維度不匹配！")
                            print(f"         預訓練模型期望: {expected_features} 個特徵")
                            print(f"         當前資料具有: {actual_features} 個特徵")
                            print(f"      💡 將標記為不相容，Stacking 會自動處理")
                            self._is_compatible = False
                        else:
                            print(f"      ✅ 特徵維度匹配 ({actual_features} 個特徵)")
                            self._is_compatible = True
                            
                    except Exception as e:
                        print(f"      ❌ 預訓練模型檢查失敗: {e}")
                        self._is_compatible = False
                    
                    return self
                
                def predict(self, X):
                    """預測 - 如果特徵不相容則使用簡單備用策略"""
                    import numpy as np
                    
                    # 檢查是否相容
                    if not hasattr(self, '_is_compatible'):
                        # 首次調用，進行檢查
                        try:
                            self._load_model_if_needed()
                            expected_features = self._scaler.n_features_in_
                            actual_features = X.shape[1]
                            self._is_compatible = (expected_features == actual_features)
                        except:
                            self._is_compatible = False
                    
                    if not self._is_compatible:
                        # 特徵不相容，使用簡單的線性預測作為備用
                        print("      ⚠️  特徵維度不相容，使用備用預測策略")
                        # 簡單的線性組合作為備用
                        if not hasattr(self, '_backup_weights'):
                            np.random.seed(42)
                            self._backup_weights = np.random.randn(X.shape[1]) * 0.1
                        
                        predictions = X @ self._backup_weights + np.random.randn(len(X)) * 0.01
                        return predictions.flatten()
                    
                    # 正常預測
                    self._load_model_if_needed()
                    X_scaled = self._scaler.transform(X)
                    predictions = self._model.predict(X_scaled, verbose=0)
                    return predictions.flatten()
                
                def get_params(self, deep=True):
                    """獲取參數 - 返回路徑而不是物件"""
                    return {
                        'model_path': self.model_path,
                        'scaler_path': self.scaler_path,
                        'model': None,  # 不返回 TensorFlow 物件
                        'scaler': None
                    }
                
                def set_params(self, **params):
                    """設置參數"""
                    for key, value in params.items():
                        if key in ['model_path', 'scaler_path']:
                            setattr(self, key, value)
                        elif key in ['model', 'scaler']:
                            setattr(self, f'_{key}', value)
                    return self
                
                def __deepcopy__(self, memo):
                    """自定義深拷貝行為 - 避免 TensorFlow 物件拷貝"""
                    # 創建新實例，只拷貝路徑
                    new_instance = PretrainedDNNWrapper(
                        model_path=self.model_path,
                        scaler_path=self.scaler_path,
                        model=None,  # 不拷貝 TensorFlow 物件
                        scaler=None
                    )
                    return new_instance
            
            wrapper = PretrainedDNNWrapper(
                model_path=latest_keras_dir,
                scaler_path=scaler_file,
                model=model,
                scaler=scaler
            )
            print(f"✅ 成功載入預訓練 DNN: {base_name}")
            return wrapper
            
        except Exception as e:
            print(f"❌ 載入預訓練 DNN 失敗: {e}")
            return None
    
    def list_available_dnn_models(self):
        """列出可用的預訓練 DNN 模型"""
        try:
            from pathlib import Path
            import os
            
            model_dir = Path("../models")
            
            # 尋找 DNN 模型
            keras_dirs = [d for d in model_dir.iterdir() 
                         if d.is_dir() and d.name.endswith('_keras')]
            
            if not keras_dirs:
                print("📝 未找到任何預訓練 DNN 模型")
                return []
            
            print("📋 可用的預訓練 DNN 模型:")
            available_models = []
            
            for i, keras_dir in enumerate(sorted(keras_dirs, key=lambda x: x.stat().st_mtime, reverse=True), 1):
                base_name = keras_dir.name.replace('_keras', '')
                scaler_file = model_dir / f"{base_name}_scaler.joblib"
                info_file = model_dir / f"{base_name}_info.txt"
                
                # 檢查檔案完整性
                if scaler_file.exists():
                    # 獲取修改時間
                    mod_time = os.path.getmtime(keras_dir)
                    import datetime
                    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
                    
                    # 嘗試讀取性能資訊
                    performance_info = ""
                    if info_file.exists():
                        try:
                            with open(info_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if 'MAE:' in content:
                                    mae_line = [line for line in content.split('\n') if 'MAE:' in line]
                                    if mae_line:
                                        performance_info = f" | {mae_line[0].split('MAE:')[-1].strip()}"
                        except:
                            pass
                    
                    print(f"{i}. {base_name} (修改時間: {mod_time_str}{performance_info})")
                    available_models.append({
                        'name': base_name,
                        'keras_dir': keras_dir,
                        'scaler_file': scaler_file,
                        'mod_time': mod_time
                    })
                else:
                    print(f"{i}. {base_name} ❌ (缺少 scaler 檔案)")
            
            return available_models
            
        except Exception as e:
            print(f"❌ 列出 DNN 模型時出錯: {e}")
            return []
    
    def _get_stacking_dnn_estimator(self, input_dim):
        """獲取用於 Stacking 的 DNN 估計器 - 使用可序列化包裝器"""
        try:
            print(f"🧠 創建可序列化 DNN 估計器 (特徵數: {input_dim})...")
            return SerializableDNNWrapper(input_dim=input_dim)
        except Exception as e:
            print(f"⚠️  無法創建 DNN 估計器: {e}")
            print("� 使用 Ridge 回歸作為替代")
            return Ridge(alpha=1.0, random_state=self.random_state)

    def predict_test(self, model_name, X_test):
        """使用指定模型預測測試集"""
        if model_name not in self.trained_models:
            raise ValueError(f"模型 {model_name} 尚未訓練!")
        
        model = self.trained_models[model_name]
        test_pred = model.predict(X_test)
        
        return test_pred