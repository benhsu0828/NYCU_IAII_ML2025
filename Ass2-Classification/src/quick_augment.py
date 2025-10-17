#!/usr/bin/env python
"""
快速資料增強腳本 - 使用預設參數
"""

import os
import sys
from pathlib import Path
import platform

def get_correct_paths():
    """根據運行環境自動選擇正確的路徑格式"""
    
    # 檢測是否在 WSL 環境中
    is_wsl = "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")
    
    if is_wsl:
        # WSL 路徑格式
        base_path = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification"
        input_dir = f"{base_path}/Dataset/preprocessed/train"
        output_dir = f"{base_path}/Dataset/augmented/train"
        backgrounds_dir = f"{base_path}/backgrounds"
        print("🐧 檢測到 WSL 環境，使用 Linux 路徑格式")
    else:
        # Windows 路徑格式
        input_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\Dataset\preprocessed\train"
        output_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\Dataset\augmented\train"
        backgrounds_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\backgrounds"
        print("🪟 檢測到 Windows 環境，使用 Windows 路徑格式")
    
    return input_dir, output_dir, backgrounds_dir

def run_data_augmentation():
    """執行資料增強"""
    
    # 自動獲取正確路徑
    input_dir, output_dir, backgrounds_dir = get_correct_paths()
    
    print("🎨 快速資料增強")
    print("=" * 40)
    print(f"📂 輸入: {input_dir}")
    print(f"📂 輸出: {output_dir}")
    print(f"🌅 背景: {backgrounds_dir}")
    
    # 檢查輸入資料夾
    if not os.path.exists(input_dir):
        print(f"❌ 找不到輸入資料夾: {input_dir}")
        print("請確認你的預處理資料已準備好")
        return False
    
    # 檢查類別數量
    class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    print(f"📊 找到 {len(class_dirs)} 個類別")
    
    # 統計原始圖片數量
    total_images = 0
    for class_dir in class_dirs:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(class_dir.glob(ext)))
        total_images += len(image_files)
    
    print(f"📷 原始圖片總數: {total_images}")
    
    # 詢問用戶
    print("\n增強設定 (使用你的 data_aggV1.py 和 data_aggV2.py 的確切方法):")
    print("  - 每張圖片增強 3 次")
    print("  - data_aggV1.py: 翻轉、旋轉、顏色調整、4種噪聲、透視變換、模糊等")
    print("  - data_aggV2.py: 移除黑白背景 + 替換為自定義背景 + 隨機邊距")
    print("  - 所有參數與你的原始腳本完全一致")
    
    choice = input("\n是否開始增強? (y/n): ")
    if choice.lower() not in ['y', 'yes', '是']:
        print("已取消")
        return False
    
    # 導入增強模組
    try:
        from data_augmentation import augment_dataset
    except ImportError as e:
        print(f"❌ 導入增強模組失敗: {e}")
        print("請確保在正確的 Python 環境中運行")
        return False
    
    # 執行增強
    try:
        augment_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            backgrounds_dir=backgrounds_dir,
            augment_per_image=3,
            use_background_aug=True,
            use_transform_aug=True
        )
        
        print(f"\n✅ 增強完成！")
        print(f"🎯 預期總圖片數: ~{total_images * 4}")  # 原始 + 3倍增強
        return True
        
    except Exception as e:
        print(f"❌ 增強過程出錯: {e}")
        return False

def create_background_folder():
    """創建背景圖片資料夾"""
    _, _, backgrounds_dir = get_correct_paths()
    
    if not os.path.exists(backgrounds_dir):
        os.makedirs(backgrounds_dir)
        print(f"📁 已創建背景圖片資料夾: {backgrounds_dir}")
        print("💡 你可以在這個資料夾放入背景圖片 (.jpg, .png)")
        print("   這些背景會用於背景合成增強")
    else:
        bg_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            bg_files.extend(list(Path(backgrounds_dir).glob(ext)))
        print(f"📁 背景圖片資料夾已存在，包含 {len(bg_files)} 張背景圖")

def main():
    """主函數"""
    print("🚀 辛普森角色資料增強 - 快速啟動")
    print("=" * 50)
    
    # 1. 檢查/創建背景資料夾
    create_background_folder()
    
    # 2. 執行資料增強
    success = run_data_augmentation()
    
    if success:
        print("\n下一步:")
        print("1. 檢查增強後的資料:")
        print("   E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/augmented/train")
        print("2. 更新訓練腳本中的資料路徑")
        print("3. 開始訓練!")
    else:
        print("\n請檢查:")
        print("1. Python 環境是否正確 (conda activate vit_env)")
        print("2. 預處理資料是否存在")
        print("3. 必要套件是否已安裝")

if __name__ == "__main__":
    main()