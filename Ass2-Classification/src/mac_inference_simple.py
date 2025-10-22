#!/usr/bin/env python3
"""
🔮 Mac EfficientNet 推理工具 - 簡化版本

純 Python 實現，不依賴額外套件。
適合快速測試目錄結構和輸出格式。
"""

import os
import glob
import csv
import random

# 模擬的辛普森家庭角色類別
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

def predict_test_dataset_simple(test_dir="Dataset/test", output_file="predictions.csv"):
    """
    簡化版本的測試資料集預測（使用模擬預測）
    
    Args:
        test_dir: 測試圖片目錄
        output_file: 輸出 CSV 檔案名稱
        
    Returns:
        list: 預測結果列表
    """
    print(f"🎭 Mac EfficientNet 推理工具 - 簡化版本")
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
    
    # 模擬預測
    print("🎲 開始模擬推理...")
    total = len(image_paths)
    
    for i, image_path in enumerate(image_paths):
        filename = os.path.basename(image_path)
        
        # 模擬預測（隨機選擇角色）
        predicted_class = random.choice(SIMPSON_CHARACTERS)
        
        results.append({
            'filename': filename,
            'prediction': predicted_class
        })
        
        # 顯示進度
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            progress = (i + 1) / total * 100
            print(f"   進度: {i+1}/{total} ({progress:.1f}%)")
    
    # 保存 CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'prediction'])  # 標題列
        
        for result in results:
            writer.writerow([result['filename'], result['prediction']])
    
    print(f"💾 預測結果已保存至: {output_file}")
    
    # 統計預測類別分布
    prediction_counts = {}
    for result in results:
        pred = result['prediction']
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    # 顯示統計資訊
    print(f"\n📊 預測統計:")
    print(f"   總圖片數: {len(results)}")
    print(f"   預測類別分布 (前10名):")
    
    # 按數量排序並顯示前10名
    sorted_counts = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, count) in enumerate(sorted_counts[:10]):
        print(f"     {class_name}: {count}")
    
    # 顯示前幾個結果
    print(f"\n📋 前 10 個預測結果:")
    print(f"{'檔名':<15} {'預測結果'}")
    print("-" * 50)
    for i in range(min(10, len(results))):
        result = results[i]
        print(f"{result['filename']:<15} {result['prediction']}")
    
    return results

def main():
    """主函數"""
    print("🍎 Mac EfficientNet 推理工具 - 簡化版本")
    print("=" * 50)
    print("⚠️  注意: 這是簡化版本，使用模擬預測結果")
    print("📝 不需要安裝額外套件，純 Python 實現")
    print("🎯 輸出格式: CSV (檔名, 預測結果)")
    print("")
    
    # 設定測試目錄
    test_dir = input("測試圖片目錄 (預設: Dataset/test): ").strip()
    if not test_dir:
        test_dir = "Dataset/test"
    
    # 檢查測試目錄是否存在
    if not os.path.exists(test_dir):
        print(f"❌ 測試目錄不存在: {test_dir}")
        
        # 提供建議
        current_dir = os.getcwd()
        print(f"📍 目前工作目錄: {current_dir}")
        
        # 尋找可能的測試目錄
        possible_dirs = [
            "Dataset/test",
            "../Dataset/test", 
            "Ass2-Classification/Dataset/test"
        ]
        
        print("🔍 建議的測試目錄:")
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                print(f"   ✅ {dir_path}")
            else:
                print(f"   ❌ {dir_path}")
        
        return 1
    
    # 設定輸出檔案
    output_file = input("輸出檔案名稱 (預設: predictions_simple.csv): ").strip()
    if not output_file:
        output_file = "predictions_simple.csv"
    
    # 執行簡化推理
    try:
        print(f"\n🚀 開始簡化推理...")
        results = predict_test_dataset_simple(test_dir, output_file)
        
        if results is not None:
            print(f"\n🎉 簡化推理完成！")
            print(f"📄 結果已保存至: {output_file}")
            print(f"🔢 共處理 {len(results)} 張圖片")
            
            # 提示如何使用實際模型
            print(f"\n💡 升級到完整版的步驟:")
            print(f"   1. 安裝依賴: pip install -r requirements.txt")
            print(f"   2. 取得訓練好的 .pth 模型檔案") 
            print(f"   3. 使用完整版: python src/mac_inference.py --model your_model.pth")
            
            # 提示查看結果
            print(f"\n📖 查看結果:")
            print(f"   cat {output_file}")
            print(f"   head -20 {output_file}")
            
        else:
            print("❌ 簡化推理失敗！")
            return 1
            
    except Exception as e:
        print(f"❌ 推理過程發生錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
