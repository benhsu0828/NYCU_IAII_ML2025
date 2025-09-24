import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np

def load_data():
    """
    讀取 train, valid, test 三個 Excel 檔案
    返回: train_df, valid_df, test_df
    """
    # 取得上級目錄的 Dataset/raw/ 路徑
    current_dir = Path(os.getcwd())
    data_dir = current_dir.parent / "Dataset" / "raw"
    
    print(f"資料目錄: {data_dir}")
    # 讀取三個檔案
    train_path = data_dir / "train-v2.xlsx"
    valid_path = data_dir / "valid-v2.xlsx" 
    test_path = data_dir / "test-reindex-test-v2.1.xlsx"
    
    print(f"讀取訓練資料: {train_path}")
    train_df = pd.read_excel(train_path)
    
    print(f"讀取驗證資料: {valid_path}")
    valid_df = pd.read_excel(valid_path)
    
    print(f"讀取測試資料: {test_path}")
    test_df = pd.read_excel(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, valid_df, test_df

def drop_columns(df, columns_to_drop):
    """
    刪除指定欄位
    
    Args:
        df: DataFrame
        columns_to_drop: list of column names to drop
    
    Returns:
        DataFrame with specified columns dropped
    """
    if not columns_to_drop:
        return df
        
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"警告: 以下欄位不存在於資料中: {missing_cols}")
    
    if existing_cols:
        print(f"刪除欄位: {existing_cols}")
        df = df.drop(columns=existing_cols)
    
    return df

def save_processed_data(train_df, valid_df, test_df, suffix="processed"):
    """
    儲存前處理後的資料到 Dataset/processed/ 目錄
    
    Args:
        train_df, valid_df, test_df: 前處理後的 DataFrames
        suffix: 檔案名稱後綴
    """
    # 建立 processed 目錄路徑
    current_dir = Path(os.getcwd())
    processed_dir = current_dir.parent / "Dataset" / "processed"
    print(f"儲存前處理後的資料到: {processed_dir}")
    
    # 建立目錄（如果不存在）
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 定義儲存路徑
    train_save_path = processed_dir / f"train_{suffix}.csv"
    valid_save_path = processed_dir / f"valid_{suffix}.csv"
    test_save_path = processed_dir / f"test_{suffix}.csv"
    
    # 儲存為 CSV 檔案
    print(f"儲存訓練資料到: {train_save_path}")
    train_df.to_csv(train_save_path, index=False, encoding='utf-8-sig')
    
    print(f"儲存驗證資料到: {valid_save_path}")
    valid_df.to_csv(valid_save_path, index=False, encoding='utf-8-sig')
    
    print(f"儲存測試資料到: {test_save_path}")
    test_df.to_csv(test_save_path, index=False, encoding='utf-8-sig')
    
    print("所有資料已成功儲存!")

def preprocess_data(columns_to_drop=None, save_data=True):
    """
    完整資料前處理流程
    
    Args:
        columns_to_drop: list of column names to drop
        save_data: bool, 是否儲存前處理後的資料
    
    Returns:
        train_df, valid_df, test_df (preprocessed)
    """
    if columns_to_drop is None:
        # 預設要刪除的欄位（可根據實際需求修改）
        columns_to_drop = []
        # 範例: columns_to_drop = ['id', 'unnecessary_col']
    
    # 載入資料
    train_df, valid_df, test_df = load_data()
    
    # 刪除指定欄位
    train_df = drop_columns(train_df, columns_to_drop)
    valid_df = drop_columns(valid_df, columns_to_drop)
    test_df = drop_columns(test_df, columns_to_drop)
    

    print("\n資料欄位移除完畢:")
    print(f"Train shape after preprocessing: {train_df.shape}")
    print(f"Valid shape after preprocessing: {valid_df.shape}")
    print(f"Test shape after preprocessing: {test_df.shape}")

    print("\n開始資料編碼...")
    
    '''
    自定義規則處理交易筆棠數，透過正則表達式提取數字部分並轉為整數後加總
    例如: '土地2建物1車位1' -> 4，'土地7建物1車位0' -> 8
    '''
    def _sum_numbers(text):
        nums = re.findall(r'\d+', str(text))
        return sum(int(n) for n in nums) if nums else 0
    if '交易筆棟數' in train_df.columns:
        train_df['交易筆棟數'] = train_df['交易筆棟數'].apply(_sum_numbers)
    if '交易筆棟數' in valid_df.columns:
        valid_df['交易筆棟數'] = valid_df['交易筆棟數'].apply(_sum_numbers)
    if '交易筆棟數' in test_df.columns:
        test_df['交易筆棟數'] = test_df['交易筆棟數'].apply(_sum_numbers)

    # 處理日期欄位
    if '交易年月日' in train_df.columns:
        print("處理交易年月日...")
        train_df, valid_df, test_df = encode_date_features(
            train_df, valid_df, test_df, 
            date_columns=['交易年月日'], 
            method='multiple_features'  # 或改為 'days_since', 'cyclical', 'timestamp'
        )

    onehot_cols = ['鄉鎮市區', '建物型態']
    label_cols = ['交易標的', '都市土地使用分區']
    train_df, valid_df, test_df = dataEncode(train_df, valid_df, test_df, onehot_cols=onehot_cols, label_cols=label_cols)

    # 對齊資料集欄位，確保維度一致
    train_df, valid_df, test_df = align_dataframe_columns(train_df, valid_df, test_df)

    train_Val_dropCol = ['編號']
    train_df = drop_columns(train_df, train_Val_dropCol)
    valid_df = drop_columns(valid_df, train_Val_dropCol)

    # 處理缺失值：將 NaN 值填補為 0
    print("\n處理缺失值...")
    train_df, valid_df, test_df = handle_missing_values(
        train_df, valid_df, test_df, 
        strategy='zero',  # 可以改為 'mean', 'median', 'mode', 'drop'
        target_col='總價元'
    )

    # 儲存前處理後的資料
    if save_data:
        save_processed_data(train_df, valid_df, test_df)
    
    return train_df, valid_df, test_df


def dataEncode(train_df, valid_df, test_df, onehot_cols=None, label_cols=None):
    """
    指定欄位 one-hot encoding 與 label encoding
    Args:
        train_df, valid_df, test_df: DataFrame
        onehot_cols: list, 要做 one-hot 的欄位
        label_cols: list, 要做 label encoding 的欄位
    """
    if onehot_cols is None:
        onehot_cols = []
    if label_cols is None:
        label_cols = []

    # One-hot encoding
    if onehot_cols:
        print(f"One-hot encoding 欄位: {onehot_cols}")
        for col in onehot_cols:
            print("處理欄位:", col)
            # 對單一欄位做 one-hot，然後合併回原 DataFrame
            train_dummy = pd.get_dummies(train_df[col], prefix=col, drop_first=False)
            valid_dummy = pd.get_dummies(valid_df[col], prefix=col, drop_first=False)
            test_dummy = pd.get_dummies(test_df[col], prefix=col, drop_first=False)
            
            # 刪除原欄位，加入編碼後的欄位
            train_df = train_df.drop(columns=[col]).join(train_dummy)
            valid_df = valid_df.drop(columns=[col]).join(valid_dummy)
            test_df = test_df.drop(columns=[col]).join(test_dummy)

        print("One-hot encoding 完成")
        # save_processed_data(train_df, valid_df, test_df, suffix="onehot")

    # Label encoding
    if label_cols:
        print(f"Label encoding 欄位: {label_cols}")
        le_dict = {}
        
        for col in label_cols:
            print(f"處理 Label encoding: {col}")
            
            # Fit only on training data
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            le_dict[col] = le
            
            # Transform valid data with unseen label handling  
            def safe_transform_with_new_labels(series, encoder):
                """安全轉換，未見過的標籤給新編號"""
                result = []
                classes_set = set(encoder.classes_)
                next_label = len(encoder.classes_)  # 下一個可用的編號
                new_label_map = {}  # 記錄新標籤的映射
                
                for value in series.astype(str):
                    if value in classes_set:
                        result.append(encoder.transform([value])[0])
                    elif value in new_label_map:
                        result.append(new_label_map[value])
                    else:
                        # 給未見過的標籤新編號
                        new_label_map[value] = next_label
                        result.append(next_label)
                        next_label += 1
                        
                return result
            
            valid_df[col] = safe_transform_with_new_labels(valid_df[col], le)
            test_df[col] = safe_transform_with_new_labels(test_df[col], le)
            
            # 印出編碼資訊
            print(f"  {col}: {len(le.classes_)} 個類別")
            
            # 檢查未見過的標籤
            valid_unseen = sum(1 for x in valid_df[col] if x == -1)
            test_unseen = sum(1 for x in test_df[col] if x == -1)
            if valid_unseen > 0:
                print(f"  Valid 中未見過的標籤: {valid_unseen} 個")
            if test_unseen > 0:
                print(f"  Test 中未見過的標籤: {test_unseen} 個")

    print("\n資料編碼完成:")
    print(f"Train shape after encoding: {train_df.shape}")
    print(f"Valid shape after encoding: {valid_df.shape}")
    print(f"Test shape after encoding: {test_df.shape}")
    
    return train_df, valid_df, test_df

def process_transaction_date(df, date_column='交易年月日', method='multiple_features'):
    """
    處理交易年月日欄位
    
    Parameters:
    method: 編碼方式
        - 'timestamp': 轉換為時間戳記
        - 'days_since': 計算距離某個基準日的天數
        - 'multiple_features': 拆分為年、月、日、星期等多個特徵
        - 'cyclical': 循環編碼（適合月份、星期）
    """
    if date_column not in df.columns:
        print(f"欄位 {date_column} 不存在")
        return df
    
    df_result = df.copy()
    
    def parse_taiwan_date(date_str):
        """將民國年月日轉換為西元年月日"""
        try:
            date_str = str(int(date_str))  # 確保是整數字串
            if len(date_str) == 7:  # 1080721 格式
                year = int(date_str[:3]) + 1911  # 民國轉西元
                month = int(date_str[3:5])
                day = int(date_str[5:7])
                return pd.Timestamp(year, month, day)
            else:
                return pd.NaT
        except:
            return pd.NaT
    
    # 先轉換為 pandas Timestamp
    df_result['parsed_date'] = df_result[date_column].apply(parse_taiwan_date)
    
    if method == 'multiple_features':
        # 方法3: 拆分為多個特徵（推薦）
        df_result[f'{date_column}_年'] = df_result['parsed_date'].dt.year
        df_result[f'{date_column}_月'] = df_result['parsed_date'].dt.month
        # df_result[f'{date_column}_日'] = df_result['parsed_date'].dt.day
        # df_result[f'{date_column}_星期'] = df_result['parsed_date'].dt.dayofweek  # 0=週一
        df_result[f'{date_column}_季度'] = df_result['parsed_date'].dt.quarter
        # df_result[f'{date_column}_是否週末'] = (df_result['parsed_date'].dt.dayofweek >= 5).astype(int)
        
        # 移除原始欄位
        df_result = df_result.drop(columns=[date_column])
        print("拆分為多個時間特徵")
        
    elif method == 'cyclical':
        # 方法4: 循環編碼（保持週期性）
        import math
        
        # 年份（線性）
        df_result[f'{date_column}_年'] = df_result['parsed_date'].dt.year
        
        # 月份（循環編碼）
        month = df_result['parsed_date'].dt.month
        df_result[f'{date_column}_月_sin'] = month.apply(lambda x: math.sin(2 * math.pi * x / 12))
        df_result[f'{date_column}_月_cos'] = month.apply(lambda x: math.cos(2 * math.pi * x / 12))
        
        # 日期（循環編碼，假設30天一循環）
        day = df_result['parsed_date'].dt.day
        df_result[f'{date_column}_日_sin'] = day.apply(lambda x: math.sin(2 * math.pi * x / 30))
        df_result[f'{date_column}_日_cos'] = day.apply(lambda x: math.cos(2 * math.pi * x / 30))
        
        # 星期（循環編碼）
        weekday = df_result['parsed_date'].dt.dayofweek
        df_result[f'{date_column}_星期_sin'] = weekday.apply(lambda x: math.sin(2 * math.pi * x / 7))
        df_result[f'{date_column}_星期_cos'] = weekday.apply(lambda x: math.cos(2 * math.pi * x / 7))
        
        # 移除原始欄位
        df_result = df_result.drop(columns=[date_column])
        print("使用循環編碼")
    
    # 清理臨時欄位
    if 'parsed_date' in df_result.columns:
        df_result = df_result.drop(columns=['parsed_date'])
    
    return df_result

def encode_date_features(train_df, valid_df, test_df, date_columns=None, method='multiple_features'):
    """
    對多個資料集統一處理日期欄位
    """
    if date_columns is None:
        date_columns = ['交易年月日']
    
    for col in date_columns:
        if col in train_df.columns:
            print(f"處理日期欄位: {col}")
            train_df = process_transaction_date(train_df, col, method)
            valid_df = process_transaction_date(valid_df, col, method)
            test_df = process_transaction_date(test_df, col, method)
    
    return train_df, valid_df, test_df

def align_dataframe_columns(train_df, valid_df, test_df, target_columns=None):
    """
    對齊三個資料集的欄位，確保維度一致
    
    Parameters:
    target_columns: 目標變數欄位名稱列表，這些欄位只在 train/valid 保留
    """
    if target_columns is None:
        target_columns = ['總價元']
    
    print("對齊資料集欄位...")
    print(f"對齊前 - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")
    
    # 取得所有特徵欄位（排除目標變數）
    all_train_cols = set(train_df.columns)
    all_valid_cols = set(valid_df.columns)
    all_test_cols = set(test_df.columns)
    
    # 找出所有特徵欄位（非目標變數）
    feature_cols_train = all_train_cols - set(target_columns)
    feature_cols_valid = all_valid_cols - set(target_columns)
    feature_cols_test = all_test_cols - set(target_columns)
    
    # 取聯集作為統一的特徵欄位
    unified_feature_cols = list(feature_cols_train | feature_cols_valid | feature_cols_test)
    unified_feature_cols.sort()  # 確保順序一致
    
    print(f"統一特徵欄位數量: {len(unified_feature_cols)}")
    
    # 對齊特徵欄位
    train_df_aligned = train_df.reindex(columns=unified_feature_cols, fill_value=0)
    valid_df_aligned = valid_df.reindex(columns=unified_feature_cols, fill_value=0)
    test_df_aligned = test_df.reindex(columns=unified_feature_cols, fill_value=0)
    
    # 加回目標變數（如果存在）
    for target_col in target_columns:
        if target_col in train_df.columns:
            train_df_aligned[target_col] = train_df[target_col]
        if target_col in valid_df.columns:
            valid_df_aligned[target_col] = valid_df[target_col]
        # test 通常沒有目標變數，不加回
    
    print(f"對齊後 - Train: {train_df_aligned.shape}, Valid: {valid_df_aligned.shape}, Test: {test_df_aligned.shape}")
    
    # 檢查是否還有差異
    train_features = [col for col in train_df_aligned.columns if col not in target_columns]
    valid_features = [col for col in valid_df_aligned.columns if col not in target_columns]
    test_features = list(test_df_aligned.columns)
    
    if len(train_features) != len(valid_features) or len(train_features) != len(test_features):
        print("警告：特徵欄位數量仍不一致！")
        print(f"Train 特徵數: {len(train_features)}")
        print(f"Valid 特徵數: {len(valid_features)}")
        print(f"Test 特徵數: {len(test_features)}")
    else:
        print("✓ 特徵欄位已成功對齊")
    
    return train_df_aligned, valid_df_aligned, test_df_aligned

def validate_model_ready_data(train_df, valid_df, test_df, target_col):
    """
    驗證資料是否準備好訓練模型
    """
    # 分離特徵
    X_train = train_df.drop([target_col], axis=1, errors='ignore')
    X_valid = valid_df.drop([target_col], axis=1, errors='ignore')
    X_test = test_df.drop(["編號"], axis=1, errors='ignore')

    print("=== 模型訓練資料驗證 ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # 檢查特徵維度
    if X_train.shape[1] != X_valid.shape[1]:
        print("❌ Train 和 Valid 特徵數量不一致！")
        return False
    
    if X_train.shape[1] != X_test.shape[1]:
        print("❌ Train 和 Test 特徵數量不一致！")
        return False
    
    # 檢查欄位名稱
    if list(X_train.columns) != list(X_valid.columns):
        print("❌ Train 和 Valid 欄位名稱不一致！")
        return False
        
    if list(X_train.columns) != list(X_test.columns):
        print("❌ Train 和 Test 欄位名稱不一致！")
        return False
    
    print("✅ 資料已準備好訓練模型！")
    return True

def handle_missing_values(train_df, valid_df, test_df, strategy='zero', target_col='總價元'):
    """
    處理缺失值的函數
    
    Args:
        train_df, valid_df, test_df: 資料框
        strategy: 處理策略
            - 'zero': 填補為 0 (預設)
            - 'mean': 用訓練集的平均值填補
            - 'median': 用訓練集的中位數填補
            - 'mode': 用訓練集的眾數填補
            - 'drop': 刪除有缺失值的列
        target_col: 目標變數欄位名稱
    
    Returns:
        處理後的 train_df, valid_df, test_df
    """
    print(f"\n=== 處理缺失值 (策略: {strategy}) ===")
    
    # 檢查缺失值情況
    train_nan_count = train_df.isnull().sum().sum()
    valid_nan_count = valid_df.isnull().sum().sum()
    test_nan_count = test_df.isnull().sum().sum()
    
    print(f"處理前缺失值數量:")
    print(f"  Train: {train_nan_count}")
    print(f"  Valid: {valid_nan_count}")
    print(f"  Test: {test_nan_count}")
    
    if train_nan_count == 0 and valid_nan_count == 0 and test_nan_count == 0:
        print("✅ 沒有發現缺失值，無需處理")
        return train_df, valid_df, test_df
    
    # 顯示有缺失值的欄位
    all_dfs = {'Train': train_df, 'Valid': valid_df, 'Test': test_df}
    for name, df in all_dfs.items():
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isnull().sum()
            print(f"  {name} 有缺失值的欄位:")
            for col in nan_cols:
                print(f"    {col}: {nan_counts[col]} 個缺失值")
    
    if strategy == 'zero':
        # 策略 1: 填補為 0
        train_df_clean = train_df.fillna(0)
        valid_df_clean = valid_df.fillna(0)
        test_df_clean = test_df.fillna(0)
        print("✅ 所有缺失值已填補為 0")
        
    elif strategy == 'mean':
        # 策略 2: 用訓練集的平均值填補
        # 分離數值欄位和非數值欄位
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)  # 不對目標變數做填補
        
        # 計算訓練集數值欄位的平均值
        fill_values = train_df[numeric_cols].mean()
        
        # 填補數值欄位
        train_df_clean = train_df.copy()
        valid_df_clean = valid_df.copy()
        test_df_clean = test_df.copy()
        
        for col in numeric_cols:
            if col in train_df_clean.columns:
                train_df_clean[col].fillna(fill_values[col], inplace=True)
            if col in valid_df_clean.columns:
                valid_df_clean[col].fillna(fill_values[col], inplace=True)
            if col in test_df_clean.columns:
                test_df_clean[col].fillna(fill_values[col], inplace=True)
        
        # 非數值欄位用 0 填補
        train_df_clean = train_df_clean.fillna(0)
        valid_df_clean = valid_df_clean.fillna(0)
        test_df_clean = test_df_clean.fillna(0)
        
        print("✅ 數值欄位用平均值填補，非數值欄位用 0 填補")
        
    elif strategy == 'median':
        # 策略 3: 用訓練集的中位數填補
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        fill_values = train_df[numeric_cols].median()
        
        train_df_clean = train_df.copy()
        valid_df_clean = valid_df.copy()
        test_df_clean = test_df.copy()
        
        for col in numeric_cols:
            if col in train_df_clean.columns:
                train_df_clean[col].fillna(fill_values[col], inplace=True)
            if col in valid_df_clean.columns:
                valid_df_clean[col].fillna(fill_values[col], inplace=True)
            if col in test_df_clean.columns:
                test_df_clean[col].fillna(fill_values[col], inplace=True)
        
        train_df_clean = train_df_clean.fillna(0)
        valid_df_clean = valid_df_clean.fillna(0)
        test_df_clean = test_df_clean.fillna(0)
        
        print("✅ 數值欄位用中位數填補，非數值欄位用 0 填補")
        
    elif strategy == 'drop':
        # 策略 4: 刪除有缺失值的列
        print("⚠️ 注意：刪除有缺失值的列可能會影響模型性能")
        original_cols = len(train_df.columns)
        
        # 找出沒有缺失值的欄位
        train_clean_cols = train_df.columns[~train_df.isnull().any()].tolist()
        valid_clean_cols = valid_df.columns[~valid_df.isnull().any()].tolist()
        test_clean_cols = test_df.columns[~test_df.isnull().any()].tolist()
        
        # 取交集，確保所有資料集都有這些欄位
        common_clean_cols = list(set(train_clean_cols) & set(valid_clean_cols) & set(test_clean_cols))
        
        # 確保目標變數在訓練和驗證集中
        if target_col in train_df.columns and target_col not in common_clean_cols:
            common_clean_cols.append(target_col)
        
        train_df_clean = train_df[common_clean_cols]
        valid_df_clean = valid_df[common_clean_cols if target_col in valid_df.columns else [col for col in common_clean_cols if col != target_col]]
        test_df_clean = test_df[[col for col in common_clean_cols if col != target_col]]
        
        removed_cols = original_cols - len(common_clean_cols)
        print(f"✅ 刪除了 {removed_cols} 個有缺失值的欄位")
        
    else:
        raise ValueError(f"不支援的策略: {strategy}")
    
    # 最終驗證
    final_train_nan = train_df_clean.isnull().sum().sum()
    final_valid_nan = valid_df_clean.isnull().sum().sum()
    final_test_nan = test_df_clean.isnull().sum().sum()
    
    print(f"處理後缺失值數量:")
    print(f"  Train: {final_train_nan}")
    print(f"  Valid: {final_valid_nan}")
    print(f"  Test: {final_test_nan}")
    
    if final_train_nan + final_valid_nan + final_test_nan == 0:
        print("🎉 所有缺失值已成功處理!")
    else:
        print("⚠️ 仍有缺失值存在，請檢查處理邏輯")
    
    return train_df_clean, valid_df_clean, test_df_clean

if __name__ == "__main__":
    # 測試載入資料
    try:
        train_df, valid_df, test_df = load_data()
        print("\n欄位資訊:")
        print("Train columns:", list(train_df.columns))
        print("Valid columns:", list(valid_df.columns))
        print("Test columns:", list(test_df.columns))
        
        # 範例: 刪除特定欄位並儲存
        # columns_to_drop = ['column1', 'column2']  # 請根據實際需求修改
        # train_df, valid_df, test_df = preprocess_data(columns_to_drop, save_data=True)
        columns_to_drop = ['土地位置建物門牌',
                           '非都市土地使用分區',
                           '非都市土地使用編定',
                           '移轉層次',
                           '主要用途',
                           '主要建材', 
                           '建築完成年月', 
                           '建物現況格局-隔間',
                           '有無管理組織',
                           '車位類別', 
                           '車位移轉總面積平方公尺',
                           '備註', 
                           '棟及號', 
                           '建案名稱',
                           '解約情形']  
        train_df, valid_df, test_df = preprocess_data(columns_to_drop, save_data=True)
        
    except Exception as e:
        print(f"讀取資料時發生錯誤: {e}")

def load_processed_data(suffix="processed"):
    """
    載入前處理後的資料
    
    Args:
        suffix: 檔案名稱後綴
    
    Returns:
        train_df, valid_df, test_df
    """
    current_dir = Path(__file__).parent
    processed_dir = current_dir.parent / "Dataset" / "processed"
    
    train_path = processed_dir / f"train_{suffix}.csv"
    valid_path = processed_dir / f"valid_{suffix}.csv"
    test_path = processed_dir / f"test_{suffix}.csv"
    
    print(f"讀取前處理後的訓練資料: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"讀取前處理後的驗證資料: {valid_path}")
    valid_df = pd.read_csv(valid_path)
    
    print(f"讀取前處理後的測試資料: {test_path}")
    test_df = pd.read_csv(test_path)
    
    return train_df, valid_df, test_df
