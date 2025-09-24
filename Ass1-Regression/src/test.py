import pandas as pd
from data_preprocess import load_data, save_processed_data

def print_train_column_categories(train_df, columns):
    """
    輸出指定欄位在 train_df 中的類別數量和所有類別名稱
    Args:
        train_df: DataFrame
        columns: list, 欲檢查的欄位
    """
    for col in columns:
        if col in train_df.columns:
            unique_vals = train_df[col].dropna().unique()
            print(f"欄位: {col}")
            print(f"  類別數量: {len(unique_vals)}")
            print(f"  類別名稱: {list(unique_vals)}\n")
        else:
            print(f"欄位 {col} 不存在於 train_df\n")

if __name__ == "__main__":
    # 只讀入原始訓練資料
    train_df, valid_df, test_df = load_data()
    # 取得所有欄位名稱
    # all_columns = list(train_df.columns)
    col = ['鄉鎮市區', '交易標的', '都市土地使用分區', '交易筆棟數', '建物型態']
    # 輸出每個欄位的 unique 類別數量和名稱
    # print_train_column_categories(train_df, col)
    save_processed_data(valid_df, valid_df, valid_df)