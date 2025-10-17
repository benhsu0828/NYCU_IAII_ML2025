#!/usr/bin/env python
"""
GPU 加速資料增強腳本 - 針對預處理的辛普森角色資料
使用 GPU 批次處理提升速度
"""

import os
import random
import shutil
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision.transforms import functional as TF
from pathlib import Path
import argparse
from tqdm import tqdm
import platform
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def get_default_paths():
    """根據運行環境自動選擇預設路徑"""
    is_wsl = "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")
    
    if is_wsl:
        base_path = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification"
        input_dir = f"{base_path}/Dataset/preprocessed/train"
        output_dir = f"{base_path}/Dataset/augmented/train"
        backgrounds_dir = f"{base_path}/backgrounds"
    else:
        input_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/train"
        output_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/augmented/train"
        backgrounds_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/backgrounds"
    
    return input_dir, output_dir, backgrounds_dir

# ===== GPU 加速的噪聲類別 =====

class GPUAddGaussianNoise(torch.nn.Module):
    """GPU 加速的高斯噪聲"""
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        if tensor.is_cuda:
            noise = torch.randn_like(tensor, device=tensor.device) * self.std + self.mean
        else:
            noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

class GPUAddSpeckleNoise(torch.nn.Module):
    """GPU 加速的散斑噪聲"""
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, tensor):
        if tensor.is_cuda:
            noise = torch.randn_like(tensor, device=tensor.device) * self.noise_level
        else:
            noise = torch.randn_like(tensor) * self.noise_level
        noisy_tensor = tensor * (1 + noise)
        return torch.clamp(noisy_tensor, 0, 1)

class GPUAddPoissonNoise(torch.nn.Module):
    """GPU 加速的泊松噪聲"""
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def forward(self, tensor):
        if tensor.is_cuda:
            noise = torch.poisson(self.lam * torch.ones_like(tensor, device=tensor.device))
        else:
            noise = torch.poisson(self.lam * torch.ones_like(tensor))
        noisy_tensor = tensor + noise / 255.0
        return torch.clamp(noisy_tensor, 0, 1)

class GPUAddSaltPepperNoise(torch.nn.Module):
    """GPU 加速的椒鹽噪聲"""
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        super().__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def forward(self, tensor):
        if tensor.is_cuda:
            noise = torch.rand_like(tensor, device=tensor.device)
        else:
            noise = torch.rand_like(tensor)
        
        tensor = tensor.clone()
        tensor[noise < self.salt_prob] = 1
        tensor[noise > 1 - self.pepper_prob] = 0
        return tensor

class GPUAugmentationPipeline(torch.nn.Module):
    """GPU 加速的資料增強管道"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # 與你的 data_aggV1.py 完全一致的變換
        self.transforms = torch.nn.Sequential(
            T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
            T.RandomApply([T.RandomVerticalFlip()], p=0.1),
            T.RandomApply([T.RandomRotation(10)], p=0.1),
            
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
            T.RandomGrayscale(p=0.1),
            T.RandomInvert(p=0.1),
            T.RandomPosterize(bits=2, p=0.1),
            T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
            T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),
            
            T.RandomApply([GPUAddGaussianNoise(0., 0.05)], p=0.1),
            T.RandomApply([GPUAddPoissonNoise(lam=0.1)], p=0.1),
            T.RandomApply([GPUAddSpeckleNoise(noise_level=0.1)], p=0.1),
            T.RandomApply([GPUAddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),
            
            T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
            T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
            T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),
            
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),
            T.RandomApply([GPUAddGaussianNoise(0., 0.001)], p=1.0)
        )
        
        self.to(device)
    
    def forward(self, batch_tensor):
        """
        處理一個批次的圖片
        Args:
            batch_tensor: (B, C, H, W) 的 tensor
        Returns:
            增強後的 tensor
        """
        return self.transforms(batch_tensor)

def load_images_as_batch(image_paths, batch_size=8, target_size=(256, 256)):
    """
    批次載入圖片為 tensor
    
    Args:
        image_paths: 圖片路徑列表
        batch_size: 批次大小
        target_size: 目標尺寸
    
    Returns:
        batches: [(tensor, paths), ...] 列表
    """
    batches = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        valid_paths = []
        
        for img_path in batch_paths:
            try:
                # 載入並轉換為 tensor
                img = Image.open(img_path).convert("RGB")
                img = img.resize(target_size)
                tensor = TF.to_tensor(img)  # (C, H, W)
                batch_tensors.append(tensor)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️  跳過損壞的圖片: {img_path} - {e}")
        
        if batch_tensors:
            # 組合成批次 (B, C, H, W)
            batch_tensor = torch.stack(batch_tensors)
            batches.append((batch_tensor, valid_paths))
    
    return batches

def save_batch_images(batch_tensor, output_paths, quality=95):
    """
    批次儲存 tensor 為圖片
    
    Args:
        batch_tensor: (B, C, H, W) tensor
        output_paths: 輸出路徑列表
        quality: JPEG 品質
    """
    # 確保 tensor 在 CPU 上
    if batch_tensor.is_cuda:
        batch_tensor = batch_tensor.cpu()
    
    for i, (tensor, output_path) in enumerate(zip(batch_tensor, output_paths)):
        # tensor 轉換為 PIL Image
        img = TF.to_pil_image(tensor)
        
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 儲存圖片
        img.save(output_path, "JPEG", quality=quality)

def gpu_augment_class(class_dir, output_class_dir, augmentation_pipeline, 
                     augment_per_image=3, batch_size=8, device='cuda'):
    """
    GPU 加速單一類別的增強
    
    Args:
        class_dir: 輸入類別資料夾
        output_class_dir: 輸出類別資料夾
        augmentation_pipeline: GPU 增強管道
        augment_per_image: 每張圖片的增強數量
        batch_size: 批次大小
        device: 設備
    """
    # 獲取所有圖片路徑
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(class_dir.glob(ext)))
    
    if not image_files:
        return 0
    
    # 創建輸出資料夾
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    # 先複製原始圖片
    for img_file in image_files:
        shutil.copy2(img_file, output_class_dir / img_file.name)
    
    total_augmented = 0
    
    # 對每張圖片生成多個增強版本
    for aug_idx in range(augment_per_image):
        # 批次載入圖片
        batches = load_images_as_batch(image_files, batch_size)
        
        for batch_tensor, batch_paths in batches:
            # 移動到 GPU
            batch_tensor = batch_tensor.to(device)
            
            # 執行增強
            with torch.no_grad():
                augmented_batch = augmentation_pipeline(batch_tensor)
            
            # 生成輸出路徑
            output_paths = []
            for img_path in batch_paths:
                base_name = Path(img_path).stem
                ext = Path(img_path).suffix
                aug_filename = f"{base_name}_aug_gpu_{aug_idx:02d}{ext}"
                output_paths.append(output_class_dir / aug_filename)
            
            # 批次儲存
            save_batch_images(augmented_batch, output_paths)
            total_augmented += len(batch_paths)
    
    return total_augmented

def gpu_augment_dataset(input_dir, output_dir, augment_per_image=3, 
                       batch_size=8, device=None):
    """
    GPU 加速資料集增強
    
    Args:
        input_dir: 輸入資料夾
        output_dir: 輸出資料夾
        augment_per_image: 每張圖片增強數量
        batch_size: GPU 批次大小
        device: GPU 設備
    """
    # 自動選擇設備
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 GPU 加速資料增強")
    print(f"⚡ 使用設備: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"📦 批次大小: {batch_size}")
    
    # 檢查 GPU 記憶體並調整批次大小
    if device == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory < 4 * 1024**3:  # < 4GB
            batch_size = min(batch_size, 4)
            print(f"⚠️  GPU 記憶體較小，調整批次大小為: {batch_size}")
    
    # 創建增強管道
    augmentation_pipeline = GPUAugmentationPipeline(device)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 處理每個類別
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f"\n📁 開始處理 {len(class_dirs)} 個類別...")
    
    total_augmented = 0
    
    for class_dir in tqdm(class_dirs, desc="處理類別"):
        class_name = class_dir.name
        output_class_dir = output_path / class_name
        
        try:
            augmented_count = gpu_augment_class(
                class_dir, output_class_dir, augmentation_pipeline,
                augment_per_image, batch_size, device
            )
            total_augmented += augmented_count
            
            # 清理 GPU 快取
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU 記憶體不足處理 {class_name}，降低批次大小重試...")
                torch.cuda.empty_cache()
                
                # 降低批次大小重試
                smaller_batch = max(1, batch_size // 2)
                augmented_count = gpu_augment_class(
                    class_dir, output_class_dir, augmentation_pipeline,
                    augment_per_image, smaller_batch, device
                )
                total_augmented += augmented_count
            else:
                raise e
    
    print(f"\n✅ GPU 加速增強完成!")
    print(f"📊 總共生成 {total_augmented} 張增強圖片")

def main():
    """主函數"""
    default_input, default_output, _ = get_default_paths()
    
    parser = argparse.ArgumentParser(description="GPU 加速辛普森角色資料增強")
    parser.add_argument("--input_dir", type=str, default=default_input, help="輸入資料夾路徑")
    parser.add_argument("--output_dir", type=str, default=default_output, help="輸出資料夾路徑")
    parser.add_argument("--augment_per_image", type=int, default=3, help="每張圖片增強數量")
    parser.add_argument("--batch_size", type=int, default=8, help="GPU 批次大小")
    parser.add_argument("--device", type=str, default=None, help="指定設備 (cuda/cpu)")
    parser.add_argument("--cpu_only", action="store_true", help="強制使用 CPU")
    
    args = parser.parse_args()
    
    # 設備選擇
    if args.cpu_only:
        device = 'cpu'
    elif args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("⚡ GPU 加速辛普森角色資料增強")
    print("=" * 50)
    print(f"📂 輸入: {args.input_dir}")
    print(f"📂 輸出: {args.output_dir}")
    print(f"🔢 每張圖片增強: {args.augment_per_image} 次")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"⚡ 設備: {device}")
    
    # 檢查輸入
    if not os.path.exists(args.input_dir):
        print(f"❌ 輸入資料夾不存在: {args.input_dir}")
        return
    
    # 執行增強
    gpu_augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augment_per_image=args.augment_per_image,
        batch_size=args.batch_size,
        device=device
    )

if __name__ == "__main__":
    main()