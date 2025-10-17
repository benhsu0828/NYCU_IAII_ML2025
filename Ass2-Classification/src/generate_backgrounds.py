#!/usr/bin/env python
"""
背景圖片下載器 - 為資料增強準備背景
"""

import os
import requests
from PIL import Image
from io import BytesIO
import random

def create_simple_backgrounds():
    """創建一些簡單的背景圖片"""
    print("🎨 創建簡單背景圖片...")
    
    backgrounds_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\backgrounds"
    os.makedirs(backgrounds_dir, exist_ok=True)
    
    # 創建純色背景
    colors = [
        (240, 240, 240),  # 淺灰
        (220, 220, 220),  # 中灰
        (200, 200, 200),  # 深灰
        (255, 248, 220),  # 米色
        (245, 245, 220),  # 象牙色
        (230, 230, 250),  # 淺紫
        (240, 248, 255),  # 愛麗絲藍
        (248, 248, 255),  # 幽靈白
        (255, 250, 240),  # 花白
        (253, 245, 230),  # 舊蕾絲
    ]
    
    sizes = [(256, 256), (300, 300), (400, 400)]
    
    for i, color in enumerate(colors):
        for j, size in enumerate(sizes):
            # 創建純色背景
            img = Image.new('RGB', size, color)
            filename = f"simple_bg_{i:02d}_{j}.png"
            filepath = os.path.join(backgrounds_dir, filename)
            img.save(filepath)
    
    print(f"✅ 創建了 {len(colors) * len(sizes)} 張簡單背景")

def create_gradient_backgrounds():
    """創建漸層背景"""
    print("🌈 創建漸層背景...")
    
    backgrounds_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\backgrounds"
    
    import numpy as np
    
    # 漸層顏色組合
    gradients = [
        ((255, 255, 255), (200, 200, 200)),  # 白到灰
        ((240, 240, 240), (180, 180, 180)),  # 淺灰到中灰
        ((255, 248, 220), (240, 230, 200)),  # 米色漸層
        ((245, 245, 220), (225, 225, 200)),  # 象牙色漸層
        ((248, 248, 255), (228, 228, 235)),  # 淺藍漸層
    ]
    
    size = (300, 300)
    
    for i, (start_color, end_color) in enumerate(gradients):
        # 垂直漸層
        img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        for y in range(size[1]):
            ratio = y / size[1]
            color = [
                int(start_color[c] * (1 - ratio) + end_color[c] * ratio)
                for c in range(3)
            ]
            img_array[y, :] = color
        
        img = Image.fromarray(img_array)
        filename = f"gradient_v_{i:02d}.png"
        filepath = os.path.join(backgrounds_dir, filename)
        img.save(filepath)
        
        # 水平漸層
        img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        for x in range(size[0]):
            ratio = x / size[0]
            color = [
                int(start_color[c] * (1 - ratio) + end_color[c] * ratio)
                for c in range(3)
            ]
            img_array[:, x] = color
        
        img = Image.fromarray(img_array)
        filename = f"gradient_h_{i:02d}.png"
        filepath = os.path.join(backgrounds_dir, filename)
        img.save(filepath)
    
    print(f"✅ 創建了 {len(gradients) * 2} 張漸層背景")

def create_texture_backgrounds():
    """創建紋理背景"""
    print("🔲 創建紋理背景...")
    
    backgrounds_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\backgrounds"
    
    import numpy as np
    
    size = (300, 300)
    
    # 噪聲紋理
    for i in range(5):
        # 生成隨機噪聲
        np.random.seed(i)
        noise = np.random.randint(0, 50, size=(size[1], size[0], 3))
        base_color = np.array([220, 220, 220])  # 淺灰基底
        
        img_array = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        filename = f"texture_noise_{i:02d}.png"
        filepath = os.path.join(backgrounds_dir, filename)
        img.save(filepath)
    
    print("✅ 創建了 5 張噪聲紋理背景")

def main():
    """主函數"""
    print("🖼️  背景圖片生成器")
    print("=" * 40)
    
    backgrounds_dir = r"E:\NYCU\NYCU_IAII_ML2025\Ass2-Classification\backgrounds"
    
    # 檢查資料夾
    if not os.path.exists(backgrounds_dir):
        os.makedirs(backgrounds_dir)
        print(f"📁 創建背景資料夾: {backgrounds_dir}")
    
    # 檢查現有背景數量
    existing_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        existing_files.extend(list(Path(backgrounds_dir).glob(ext)))
    
    print(f"📊 現有背景圖片: {len(existing_files)} 張")
    
    if len(existing_files) > 0:
        choice = input("已有背景圖片，是否要生成更多? (y/n): ")
        if choice.lower() not in ['y', 'yes', '是']:
            print("跳過背景生成")
            return
    
    # 生成背景
    try:
        import numpy as np
        from pathlib import Path
        
        create_simple_backgrounds()
        create_gradient_backgrounds()
        create_texture_backgrounds()
        
        # 統計總數
        all_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            all_files.extend(list(Path(backgrounds_dir).glob(ext)))
        
        print(f"\n✅ 背景生成完成！")
        print(f"📊 總背景圖片: {len(all_files)} 張")
        print(f"📁 保存位置: {backgrounds_dir}")
        
    except ImportError:
        print("❌ 缺少 numpy，只能創建簡單背景")
        create_simple_backgrounds()
        
    except Exception as e:
        print(f"❌ 生成背景時出錯: {e}")

if __name__ == "__main__":
    main()