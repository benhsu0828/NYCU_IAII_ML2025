#!/usr/bin/env python
"""
資料增強腳本 - 針對預處理的辛普森角色資料
結合 data_aggV1.py 和 data_aggV2.py 的增強方法
"""

import os
import random
import shutil
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms.v2 as T
from pathlib import Path
import argparse
from tqdm import tqdm
import platform

def get_default_paths():
    """根據運行環境自動選擇預設路徑"""
    
    # 檢測是否在 WSL 環境中
    is_wsl = "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")
    
    if is_wsl:
        # WSL 路徑格式
        base_path = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification"
        input_dir = f"{base_path}/Dataset/preprocessed/train"
        output_dir = f"{base_path}/Dataset/augmented/train"
        backgrounds_dir = f"{base_path}/backgrounds"  # DTD 資料集根目錄
    else:
        # Windows 路徑格式
        input_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/train"
        output_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/augmented/train"
        backgrounds_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/backgrounds"  # DTD 資料集根目錄
    
    return input_dir, output_dir, backgrounds_dir

# ===== 自定義噪聲增強類別 (來自 data_aggV1.py) =====

class AddGaussianNoise(object):
    """添加高斯噪聲"""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddSpeckleNoise(object):
    """添加散斑噪聲"""
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        noisy_tensor = tensor * (1 + noise)
        return torch.clamp(noisy_tensor, 0, 1)

class AddPoissonNoise(object):
    """添加泊松噪聲"""
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, tensor):
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        noisy_tensor = tensor + noise / 255.0
        return torch.clamp(noisy_tensor, 0, 1)

class AddSaltPepperNoise(object):
    """添加椒鹽噪聲 (與你的 data_aggV1.py 完全一致)"""
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor = tensor.clone()  # 防止修改原始 tensor
        tensor[(noise < self.salt_prob)] = 1  # Salt noise: setting some pixels to 1
        tensor[(noise > 1 - self.pepper_prob)] = 0  # Pepper noise: setting some pixels to 0
        return tensor

# ===== 資料增強策略定義 =====

def get_augmentation_transforms(device='cpu'):
    """
    定義資料增強變換 (支援 GPU 加速)
    
    Args:
        device: 'cuda' 或 'cpu'
    """
    transform = T.Compose([
        T.ToTensor(),  # Convert PIL image to tensor

        T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
        T.RandomApply([T.RandomVerticalFlip()], p=0.1),
        T.RandomApply([T.RandomRotation(10)], p=0.1),

        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
        T.RandomGrayscale(p=0.1),
        T.RandomInvert(p=0.1),
        T.RandomPosterize(bits=2, p=0.1),
        T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

        T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
        T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
        T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
        T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

        T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
        T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
        T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

        T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
        T.ToPILImage()  # Convert tensor back to PIL image for saving
    ])
    
    return transform

def create_background_composite(foreground_img, background_img):
    """
    創建背景合成圖片 (完全使用你的 data_aggV2.py 的邏輯)
    
    Args:
        foreground_img: PIL Image，前景圖片
        background_img: PIL Image，背景圖片
    
    Returns:
        PIL Image: 合成後的圖片
    """
    # 轉換為 RGBA (與你的 data_aggV2.py 一致)
    img = foreground_img.convert("RGBA")

    # 創建遮罩 (完全複製你的邏輯)
    mask = Image.new("L", img.size, 0)
    for x in range(img.width):
        for y in range(img.height):
            r, g, b, a = img.getpixel((x, y))
            if (r < 20 and g < 20 and b < 20) or (r > 235 and g > 235 and b > 235):
                mask.putpixel((x, y), 0)
            else:
                mask.putpixel((x, y), 255)

    # 背景處理 (與你的 data_aggV2.py 一致)
    background_image = background_img.convert("RGBA")

    # 添加隨機邊距 (與你的 data_aggV2.py 一致)
    padding_x = random.randint(10, 30)
    padding_y = random.randint(10, 30)
    new_size = (img.width + 2 * padding_x, img.height + 2 * padding_y)

    # 調整背景尺寸 (與你的 data_aggV2.py 一致)
    background_image = background_image.resize(new_size)

    # 合成圖片 (與你的 data_aggV2.py 一致)
    paste_position = (padding_x, padding_y)
    background_image.paste(img, paste_position, mask)

    # 返回 RGB 格式
    return background_image.convert("RGB")

def load_background_images(backgrounds_dir, max_per_category=10, total_limit=200):
    """
    載入 DTD (Describable Textures Dataset) 背景圖片
    
    Args:
        backgrounds_dir: 背景圖片根目錄 (應該包含 images/ 子目錄)
        max_per_category: 每個紋理類別最多載入多少張圖片
        total_limit: 總共最多載入多少張圖片
    
    Returns:
        list: 背景圖片列表 (PIL Image 物件)
    """
    if not os.path.exists(backgrounds_dir):
        print(f"⚠️  背景資料夾不存在: {backgrounds_dir}")
        return []
    
    # DTD 資料集的圖片通常在 backgrounds/images/ 底下
    images_dir = Path(backgrounds_dir) / "images"
    if not images_dir.exists():
        # 如果沒有 images 子目錄，直接搜尋當前目錄
        images_dir = Path(backgrounds_dir)
        print(f"📁 在根目錄搜尋背景圖片: {images_dir}")
    else:
        print(f"📁 在 DTD images 目錄搜尋背景圖片: {images_dir}")
    
    # 收集所有圖片檔案路徑
    all_background_files = []
    category_counts = {}
    
    # 遞迴搜尋所有子目錄中的圖片
    for category_dir in images_dir.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            category_files = []
            
            # 搜尋該類別目錄下的所有圖片
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG']:
                category_files.extend(list(category_dir.glob(ext)))
            
            # 隨機選擇該類別的部分圖片
            if category_files:
                selected_count = min(len(category_files), max_per_category)
                selected_files = random.sample(category_files, selected_count)
                all_background_files.extend(selected_files)
                category_counts[category_name] = selected_count
                
                print(f"  📂 {category_name}: {selected_count}/{len(category_files)} 張圖片")
    
    # 如果總數超過限制，進行二次隨機選擇
    if len(all_background_files) > total_limit:
        print(f"🎲 圖片總數 {len(all_background_files)} 超過限制 {total_limit}，進行隨機選擇...")
        all_background_files = random.sample(all_background_files, total_limit)
    
    # 載入背景圖片
    backgrounds = []
    successful_loads = 0
    failed_loads = 0
    
    print(f"🔄 開始載入 {len(all_background_files)} 張背景圖片...")
    
    for bg_file in tqdm(all_background_files, desc="載入背景圖片"):
        try:
            bg_img = Image.open(bg_file).convert("RGB")
            
            # 調整圖片尺寸以節省記憶體 (如果圖片太大)
            if max(bg_img.size) > 512:
                ratio = 512 / max(bg_img.size)
                new_size = (int(bg_img.size[0] * ratio), int(bg_img.size[1] * ratio))
                bg_img = bg_img.resize(new_size, Image.Resampling.LANCZOS)
            
            backgrounds.append(bg_img)
            successful_loads += 1
            
        except Exception as e:
            failed_loads += 1
            if failed_loads <= 5:  # 只顯示前5個錯誤，避免刷屏
                print(f"    ❌ 無法載入背景圖片 {bg_file.name}: {e}")
    
    print(f"\n✅ 背景圖片載入完成:")
    print(f"   📊 成功載入: {successful_loads} 張")
    if failed_loads > 0:
        print(f"   ⚠️  載入失敗: {failed_loads} 張")
    
    print(f"   🎯 各類別分布:")
    for category, count in sorted(category_counts.items()):
        print(f"      {category}: {count} 張")
    
    return backgrounds

def process_images_gpu_batch(image_files, output_class_dir, transform, backgrounds,
                           augment_per_image, use_background_aug, background_prob,
                           device, batch_size, class_name):
    """
    GPU 批量處理圖片增強
    
    Args:
        image_files: 圖片檔案列表
        output_class_dir: 輸出類別目錄
        transform: 增強變換
        backgrounds: 背景圖片列表
        augment_per_image: 每張圖片增強數量
        use_background_aug: 是否使用背景增強
        background_prob: 背景增強機率
        device: GPU 設備
        batch_size: 批量大小
        class_name: 類別名稱
    
    Returns:
        int: 增強圖片數量
    """
    augmented_count = 0
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    print(f"  🚀 使用 GPU 批量處理 (批量大小: {batch_size})")
    
    for batch_idx in tqdm(range(total_batches), desc=f"  GPU增強 {class_name}"):
        try:
            # 準備當前批量的圖片
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]
            
            # 載入批量圖片
            batch_images = []
            batch_info = []  # 儲存檔案資訊
            
            for img_file in batch_files:
                try:
                    img = Image.open(img_file).convert("RGB")
                    batch_images.append(img)
                    batch_info.append({
                        'file': img_file,
                        'name': img_file.stem,
                        'ext': img_file.suffix
                    })
                except Exception as e:
                    print(f"    ⚠️ 載入失敗 {img_file.name}: {e}")
            
            if not batch_images:
                continue
            
            # 轉換為 tensor 批量
            tensors = []
            for img in batch_images:
                tensor = T.ToTensor()(img)
                tensors.append(tensor)
            
            # 批量移到 GPU
            batch_tensor = torch.stack(tensors).to(device)
            
            # 對每張圖片生成多個增強版本
            for aug_idx in range(augment_per_image):
                # 批量應用變換 (在 GPU 上)
                if transform:
                    # 對每個 tensor 單獨應用變換 (因為隨機性)
                    augmented_batch = []
                    for i in range(batch_tensor.size(0)):
                        single_tensor = batch_tensor[i].unsqueeze(0)
                        # 移回 CPU 暫時處理 (因為某些變換可能不支援 GPU)
                        single_tensor_cpu = single_tensor.cpu()
                        single_pil = T.ToPILImage()(single_tensor_cpu[0])
                        
                        # 應用變換
                        augmented_pil = transform(single_pil)
                        augmented_batch.append(augmented_pil)
                    
                    # 處理背景合成和保存
                    for i, (augmented_img, info) in enumerate(zip(augmented_batch, batch_info)):
                        aug_methods = ["trans"]
                        current_img = augmented_img
                        
                        # 背景增強
                        if use_background_aug and backgrounds and random.random() < background_prob:
                            bg_img = random.choice(backgrounds)
                            current_img = create_background_composite(current_img, bg_img)
                            aug_methods.append("bg")
                        
                        # 保存
                        methods_str = "_".join(aug_methods)
                        aug_filename = f"{info['name']}_aug_{methods_str}_{aug_idx:02d}{info['ext']}"
                        aug_path = output_class_dir / aug_filename
                        current_img.save(aug_path, quality=95)
                        augmented_count += 1
            
            # 清理 GPU 記憶體
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"    ❌ 批量處理錯誤 (batch {batch_idx}): {e}")
            # 發生錯誤時清理 GPU 記憶體
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return augmented_count

def augment_dataset(input_dir, output_dir, backgrounds_dir=None, 
                   augment_per_image=2, use_background_aug=True, use_transform_aug=True,
                   max_bg_per_category=10, max_total_backgrounds=200, background_prob=0.3,
                   use_gpu=True, batch_size=32):
    """
    對整個資料集進行增強 (支援 GPU 加速)
    
    Args:
        input_dir: 輸入資料夾 (包含類別子資料夾)
        output_dir: 輸出資料夾
        backgrounds_dir: 背景圖片資料夾 (DTD 資料集根目錄)
        augment_per_image: 每張圖片生成的增強版本數量
        use_background_aug: 是否使用背景合成增強
        use_transform_aug: 是否使用變換增強
        max_bg_per_category: 每個紋理類別最多載入的背景圖片數
        max_total_backgrounds: 總共最多載入的背景圖片數
        background_prob: 使用背景增強的機率
        use_gpu: 是否使用 GPU 加速
        batch_size: GPU 批量處理大小
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # GPU 設定
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用設備: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"📦 批量大小: {batch_size}")
    
    # 創建輸出資料夾
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 載入背景圖片 (DTD 資料集)
    backgrounds = []
    if use_background_aug and backgrounds_dir:
        backgrounds = load_background_images(
            backgrounds_dir, 
            max_per_category=max_bg_per_category,
            total_limit=max_total_backgrounds
        )
    
    # 準備增強變換
    transform = get_augmentation_transforms(device=device.type) if use_transform_aug else None
    
    # 處理每個類別
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f"🎯 開始增強 {len(class_dirs)} 個類別的資料...")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n📁 處理類別: {class_name}")
        
        # 創建輸出類別資料夾
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        # 獲取該類別的所有圖片
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(class_dir.glob(ext)))
        
        print(f"  📷 找到 {len(image_files)} 張原始圖片")
        
        # 先複製原始圖片
        for img_file in image_files:
            shutil.copy2(img_file, output_class_dir / img_file.name)
        
        # 生成增強版本
        augmented_count = 0
        
        if device.type == 'cuda' and use_transform_aug and len(image_files) > batch_size:
            # GPU 批量處理模式
            augmented_count = process_images_gpu_batch(
                image_files, output_class_dir, transform, backgrounds,
                augment_per_image, use_background_aug, background_prob,
                device, batch_size, class_name
            )
        else:
            # CPU 單張處理模式 (原始邏輯)
            for img_file in tqdm(image_files, desc=f"  增強 {class_name}"):
                try:
                    # 載入原始圖片
                    original_img = Image.open(img_file).convert("RGB")
                    base_name = img_file.stem
                    ext = img_file.suffix
                    
                    # 生成多個增強版本
                    for aug_idx in range(augment_per_image):
                        current_img = original_img.copy()
                        aug_methods = []
                        
                        # 方法 1: 變換增強 (使用你的 data_aggV1.py 變換)
                        if use_transform_aug and transform:
                            current_img = transform(current_img)
                            aug_methods.append("trans")
                        
                        # 方法 2: 背景合成增強 (使用你的 data_aggV2.py 邏輯)
                        if use_background_aug and backgrounds and random.random() < background_prob:
                            # 隨機選擇一張背景圖片
                            bg_img = random.choice(backgrounds)
                            current_img = create_background_composite(current_img, bg_img)
                            aug_methods.append("bg")
                        
                        # 如果有應用增強，則保存
                        if aug_methods:
                            methods_str = "_".join(aug_methods)
                            aug_filename = f"{base_name}_aug_{methods_str}_{aug_idx:02d}{ext}"
                            aug_path = output_class_dir / aug_filename
                            current_img.save(aug_path, quality=95)
                            augmented_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 處理 {img_file} 時出錯: {e}")
        
        print(f"  ✅ 完成，生成了 {augmented_count} 張增強圖片")
        
        # 統計該類別的總圖片數
        total_images = len(list(output_class_dir.glob("*")))
        print(f"  📊 該類別總圖片數: {total_images}")

def main():
    """主函數"""
    # 獲取環境適配的預設路徑
    default_input, default_output, default_backgrounds = get_default_paths()
    
    parser = argparse.ArgumentParser(description="辛普森角色資料增強")
    parser.add_argument("--input_dir", type=str, 
                       default=default_input,
                       help="輸入資料夾路徑")
    parser.add_argument("--output_dir", type=str,
                       default=default_output,
                       help="輸出資料夾路徑")
    parser.add_argument("--backgrounds_dir", type=str,
                       default=default_backgrounds,
                       help="背景圖片資料夾路徑")
    parser.add_argument("--augment_per_image", type=int, default=3,
                       help="每張圖片生成的增強版本數量")
    parser.add_argument("--no_background", action="store_true",
                       help="不使用背景合成增強")
    parser.add_argument("--no_transform", action="store_true", 
                       help="不使用變換增強")
    parser.add_argument("--max_bg_per_category", type=int, default=10,
                       help="每個紋理類別最多載入多少張背景圖片")
    parser.add_argument("--max_total_backgrounds", type=int, default=200,
                       help="總共最多載入多少張背景圖片")
    parser.add_argument("--background_prob", type=float, default=0.4,
                       help="使用背景增強的機率 (0.0-1.0)")
    parser.add_argument("--no_gpu", action="store_true",
                       help="不使用 GPU 加速，強制使用 CPU")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="GPU 批量處理大小 (僅在使用 GPU 時有效)")
    
    args = parser.parse_args()
    
    print("🎨 辛普森角色資料增強腳本")
    print("=" * 50)
    print(f"📂 輸入資料夾: {args.input_dir}")
    print(f"📂 輸出資料夾: {args.output_dir}")
    print(f"🖼️  背景資料夾: {args.backgrounds_dir}")
    print(f"🔢 每張圖片增強數量: {args.augment_per_image}")
    print(f"🌅 背景合成增強: {'關閉' if args.no_background else '開啟'}")
    print(f"🔄 變換增強: {'關閉' if args.no_transform else '開啟'}")
    if not args.no_background:
        print(f"📊 每類別最多背景數: {args.max_bg_per_category}")
        print(f"📊 背景總數限制: {args.max_total_backgrounds}")
        print(f"🎲 背景使用機率: {args.background_prob:.1%}")
    
    # GPU 設定資訊
    use_gpu = not args.no_gpu and torch.cuda.is_available()
    print(f"🚀 GPU 加速: {'開啟' if use_gpu else '關閉'}")
    if use_gpu:
        print(f"📦 GPU 批量大小: {args.batch_size}")
    
    # 檢查輸入資料夾
    if not os.path.exists(args.input_dir):
        print(f"❌ 輸入資料夾不存在: {args.input_dir}")
        return
    
    # 開始增強
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        backgrounds_dir=args.backgrounds_dir if not args.no_background else None,
        augment_per_image=args.augment_per_image,
        use_background_aug=not args.no_background,
        use_transform_aug=not args.no_transform,
        max_bg_per_category=args.max_bg_per_category,
        max_total_backgrounds=args.max_total_backgrounds,
        background_prob=args.background_prob,
        use_gpu=use_gpu,
        batch_size=args.batch_size
    )
    
    print(f"\n✅ 資料增強完成！")
    print(f"📊 增強後的資料保存在: {args.output_dir}")

if __name__ == "__main__":
    main()