#!/usr/bin/env python
"""
路徑檢測腳本 - 檢查 WSL/Windows 環境和資料路徑
"""

import os
import platform
from pathlib import Path

def detect_environment():
    """檢測運行環境"""
    print("🔍 環境檢測")
    print("=" * 40)
    
    # 基本系統資訊
    print(f"🖥️  作業系統: {platform.system()}")
    print(f"📋 平台: {platform.platform()}")
    print(f"🔧 架構: {platform.machine()}")
    
    # 檢測 WSL
    is_wsl = False
    wsl_indicators = []
    
    # 方法 1: 檢查 kernel release
    if "microsoft" in platform.uname().release.lower():
        is_wsl = True
        wsl_indicators.append("kernel release 包含 'microsoft'")
    
    # 方法 2: 檢查環境變數
    if "WSL_DISTRO_NAME" in os.environ:
        is_wsl = True
        wsl_indicators.append(f"WSL_DISTRO_NAME = {os.environ['WSL_DISTRO_NAME']}")
    
    # 方法 3: 檢查 /proc/version (Linux only)
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower():
                is_wsl = True
                wsl_indicators.append("/proc/version 包含 'microsoft'")
    except:
        pass
    
    if is_wsl:
        print("🐧 檢測結果: WSL (Windows Subsystem for Linux)")
        for indicator in wsl_indicators:
            print(f"   ✅ {indicator}")
    else:
        print("🪟 檢測結果: Windows 原生環境")
    
    return is_wsl

def get_paths(is_wsl):
    """根據環境獲取路徑"""
    print(f"\n📁 路徑配置")
    print("=" * 40)
    
    if is_wsl:
        base_path = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification"
        input_dir = f"{base_path}/Dataset/preprocessed/train"
        output_dir = f"{base_path}/Dataset/augmented/train"
        backgrounds_dir = f"{base_path}/backgrounds"
        print("🔗 使用 WSL 路徑格式:")
    else:
        input_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\Dataset\preprocessed\train"
        output_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\Dataset\augmented\train"
        backgrounds_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\backgrounds"
        print("🔗 使用 Windows 路徑格式:")
    
    print(f"   📂 輸入路徑: {input_dir}")
    print(f"   📂 輸出路徑: {output_dir}")
    print(f"   🌅 背景路徑: {backgrounds_dir}")
    
    return input_dir, output_dir, backgrounds_dir

def test_paths(input_dir, output_dir, backgrounds_dir):
    """測試路徑是否可訪問"""
    print(f"\n🧪 路徑測試")
    print("=" * 40)
    
    # 測試輸入路徑
    if os.path.exists(input_dir):
        class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
        print(f"✅ 輸入路徑存在，找到 {len(class_dirs)} 個類別")
        
        # 統計圖片數量
        total_images = 0
        for class_dir in class_dirs[:5]:  # 只檢查前5個類別
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(class_dir.glob(ext)))
            total_images += len(image_files)
            print(f"   📷 {class_dir.name}: {len(image_files)} 張圖片")
        
        if len(class_dirs) > 5:
            print(f"   ... (還有 {len(class_dirs) - 5} 個類別)")
        
    else:
        print(f"❌ 輸入路徑不存在: {input_dir}")
        print("   請確認:")
        print("   1. 資料是否在正確位置")
        print("   2. 路徑格式是否正確")
        print("   3. 如果在 WSL，Windows 的 E: 槽是否可訪問")
    
    # 測試輸出路徑的父目錄
    output_parent = Path(output_dir).parent
    if os.path.exists(output_parent):
        print(f"✅ 輸出父目錄存在: {output_parent}")
    else:
        print(f"⚠️  輸出父目錄不存在: {output_parent}")
        print("   將在運行時自動創建")
    
    # 測試背景路徑
    if os.path.exists(backgrounds_dir):
        bg_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            bg_files.extend(list(Path(backgrounds_dir).glob(ext)))
        print(f"✅ 背景路徑存在，找到 {len(bg_files)} 張背景圖")
    else:
        print(f"⚠️  背景路徑不存在: {backgrounds_dir}")
        print("   將在運行時自動創建")

def test_wsl_mount():
    """測試 WSL 掛載點"""
    if os.path.exists("/mnt"):
        print(f"\n🔗 WSL 掛載點檢查")
        print("=" * 40)
        
        mounts = [d for d in Path("/mnt").iterdir() if d.is_dir()]
        print(f"📍 可用掛載點: {[m.name for m in mounts]}")
        
        # 檢查 E: 槽
        if Path("/mnt/e").exists():
            print("✅ E: 槽已掛載到 /mnt/e")
            
            # 檢查 NYCU 資料夾
            nycu_path = Path("/mnt/e/NYCU")
            if nycu_path.exists():
                print("✅ 找到 /mnt/e/NYCU 資料夾")
            else:
                print("❌ 找不到 /mnt/e/NYCU 資料夾")
                
                # 列出 E: 槽的內容
                e_contents = list(Path("/mnt/e").iterdir())[:10]
                print(f"📋 E: 槽內容 (前10項): {[p.name for p in e_contents]}")
        else:
            print("❌ E: 槽未掛載到 /mnt/e")

def main():
    """主函數"""
    print("🔬 WSL/Windows 路徑檢測工具")
    print("=" * 50)
    
    # 1. 檢測環境
    is_wsl = detect_environment()
    
    # 2. 如果是 WSL，檢查掛載點
    if is_wsl:
        test_wsl_mount()
    
    # 3. 獲取路徑
    input_dir, output_dir, backgrounds_dir = get_paths(is_wsl)
    
    # 4. 測試路徑
    test_paths(input_dir, output_dir, backgrounds_dir)
    
    print(f"\n🎯 總結")
    print("=" * 40)
    if is_wsl:
        print("🐧 你在 WSL 環境中")
        print("✅ 腳本已自動配置 WSL 路徑格式")
        print("💡 如果路徑測試失敗，請檢查 Windows 檔案是否在正確位置")
    else:
        print("🪟 你在 Windows 環境中")
        print("✅ 腳本已自動配置 Windows 路徑格式")
    
    print(f"\n🚀 下一步:")
    print("   如果路徑測試通過，可以運行:")
    print("   python quick_augment.py")

if __name__ == "__main__":
    main()