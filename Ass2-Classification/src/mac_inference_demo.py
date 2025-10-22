#!/usr/bin/env python3
"""
🔮 Mac EfficientNet 推理工具 - 測試版本

這是一個測試版本，會創建模擬的推理結果。
在有實際模型檔案時，請使用完整版本。
"""

import os
import glob
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
import json

# 模擬的辛普森家庭角色類別（從 character_class_mapping.json 獲取）
SIMPSON_CHARACTERS = [
    "abraham_grampa_simpson",
    "agnes_skinner", 
    "apu_nahasapeemapetilon",
    "barney_gumble",
    "bart_simpson",
    "brandine_spuckler",
    "carl_carlson",
    "charles_montgomery_burns",
    "chief_wiggum",
    "cletus_spuckler",
    "comic_book_guy",
    "disco_stu",
    "dolph_starbeam",
    "duff_man",
    "edna_krabappel",
    "fat_tony",
    "gary_chalmers",
    "gil",
    "groundskeeper_willie",
    "homer_simpson",
    "jimbo_jones",
    "kearney_zzyzwicz",
    "kent_brockman",
    "krusty_the_clown",
    "lenny_leonard",
    "lionel_hutz",
    "lisa_simpson",
    "lunchlady_doris",
    "maggie_simpson",
    "marge_simpson",
    "martin_prince",
    "mayor_quimby",
    "milhouse_van_houten",
    "miss_hoover",
    "moe_szyslak",
    "ned_flanders",
    "nelson_muntz",
    "otto_mann",
    "patty_bouvier",
    "principal_skinner",
    "professor_john_frink",
    "rainier_wolfcastle",
    "ralph_wiggum",
    "selma_bouvier",
    "sideshow_bob",
    "sideshow_mel",
    "snake_jailbird",
    "timothy_lovejoy",
    "troy_mclure",
    "waylon_smithers"
]

def predict_test_dataset_demo(test_dir="Dataset/test", output_file="predictions.csv"):
    """
    演示版本的測試資料集預測（使用隨機預測）
    
    Args:
        test_dir: 測試圖片目錄
        output_file: 輸出 CSV 檔案名稱
        
    Returns:
        pd.DataFrame: 預測結果
    """
    print(f"🎭 Mac EfficientNet 推理工具 - 演示模式")
    print(f"📁 處理測試資料集: {test_dir}")
    
    # 支援的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
    
    # 收集所有圖片路徑
    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(test_dir, ext)
        image_paths.extend(glob.glob(pattern))
    
    # 按檔名數字排序
    def sort_key(path):
        filename = os.path.basename(path)
        try:
            number = int(filename.split('.')[0])
            return number
        except:
            return 0
    
    image_paths.sort(key=sort_key)
    
    if not image_paths:
        print("❌ 找不到任何圖片！")
        return None
    
    print(f"🔍 找到 {len(image_paths)} 張圖片")
    
    # 設定隨機種子以便重現結果
    random.seed(42)
    
    # 準備結果列表
    results = []
    
    # 模擬預測（使用進度條）
    print("🎲 開始模擬預測...")
    for image_path in tqdm(image_paths, desc="預測進度"):
        filename = os.path.basename(image_path)
        
        # 模擬預測（隨機選擇角色）
        predicted_class = random.choice(SIMPSON_CHARACTERS)
        
        results.append({
            'filename': filename,
            'prediction': predicted_class
        })
    
    # 創建 DataFrame
    df = pd.DataFrame(results)
    
    # 保存 CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 預測結果已保存至: {output_file}")
    
    # 顯示統計資訊
    print(f"\n📊 預測統計:")
    print(f"   總圖片數: {len(df)}")
    print(f"   預測類別分布 (前10名):")
    for class_name, count in df['prediction'].value_counts().head(10).items():
        print(f"     {class_name}: {count}")
    
    # 顯示前幾個結果
    print(f"\n📋 前 10 個預測結果:")
    print(df.head(10).to_string(index=False))
    
    return df

def main():
    """主函數"""
    print("🍎 Mac EfficientNet 推理工具 - 演示模式")
    print("=" * 50)
    print("⚠️  注意: 這是演示版本，使用隨機預測結果")
    print("📝 如需使用實際模型，請確保有 .pth 模型檔案")
    print("")
    
    # 設定測試目錄
    test_dir = input("測試圖片目錄 (預設: Dataset/test): ").strip()
    if not test_dir:
        test_dir = "Dataset/test"
    
    # 檢查測試目錄是否存在
    if not os.path.exists(test_dir):
        print(f"❌ 測試目錄不存在: {test_dir}")
        return 1
    
    # 設定輸出檔案
    output_file = input("輸出檔案名稱 (預設: predictions_demo.csv): ").strip()
    if not output_file:
        output_file = "predictions_demo.csv"
    
    # 執行演示推理
    try:
        print(f"\n🚀 開始演示推理...")
        df = predict_test_dataset_demo(test_dir, output_file)
        
        if df is not None:
            print(f"\n🎉 演示推理完成！")
            print(f"📄 結果已保存至: {output_file}")
            
            # 提示如何使用實際模型
            print(f"\n💡 使用實際模型的步驟:")
            print(f"   1. 確保有訓練好的 .pth 模型檔案")
            print(f"   2. 安裝依賴: pip install -r requirements.txt")
            print(f"   3. 使用完整版: python src/mac_inference.py --model your_model.pth")
            
        else:
            print("❌ 演示推理失敗！")
            return 1
            
    except Exception as e:
        print(f"❌ 演示過程發生錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
