#!/usr/bin/env python3
"""
🎯 模型第一層權重與特徵圖可視化工具

功能：
- 可視化模型第一層的權重 (filters/kernels)
- 分析單張圖片經過第一層後的特徵圖
- 計算並顯示注意力熱力圖
- 比較不同通道的響應強度
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import glob
import timm
from pathlib import Path
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class FirstLayerVisualizer:
    """
    第一層權重與特徵圖可視化器
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化可視化器
        
        Args:
            model_path: 訓練好的模型檔案路徑 (.pth)
            device: 計算設備
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.first_layer = None
        self.model_name = ""
        self.num_classes = 0
        
        print(f"🔍 第一層可視化器初始化")
        print(f"📁 模型檔案: {model_path}")
        print(f"🖥️ 使用設備: {self.device}")
        
        # 載入模型
        self._load_model()
        
    def _load_model(self):
        """載入訓練好的模型並提取第一層"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型檔案: {self.model_path}")
        
        print(f"📂 載入模型...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 提取模型資訊
        self.model_name = checkpoint.get('model_name', 'unknown')
        self.num_classes = checkpoint.get('num_classes', 50)
        
        # 創建模型架構
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=self.num_classes
            )
        except Exception as e:
            print(f"⚠️ 使用 timm 創建模型失敗: {e}")
            raise e
        
        # 載入訓練好的權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 提取第一層
        self._extract_first_layer()
        
        print(f"✅ 模型載入成功！")
        print(f"   模型: {self.model_name}")
        print(f"   第一層: {type(self.first_layer).__name__}")
        if hasattr(self.first_layer, 'weight'):
            weight_shape = self.first_layer.weight.shape
            print(f"   權重形狀: {weight_shape}")
            print(f"   filters數量: {weight_shape[0]}")
            print(f"   kernel大小: {weight_shape[2]}x{weight_shape[3]}")
        
    def _extract_first_layer(self):
        """提取模型的第一層卷積層"""
        # 尋找第一個卷積層
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.first_layer = module
                self.first_layer_name = name
                print(f"🎯 找到第一層卷積: {name}")
                break
        
        if self.first_layer is None:
            raise ValueError("找不到卷積層！")
    
    def get_transforms(self):
        """獲取圖片預處理變換"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path):
        """
        載入並預處理圖片
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            original_image: 原始圖片 (PIL)
            processed_image: 預處理後的圖片 (tensor)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到圖片: {image_path}")
        
        # 載入原始圖片
        original_image = Image.open(image_path).convert('RGB')
        
        # 預處理
        transform = self.get_transforms()
        processed_image = transform(original_image).unsqueeze(0)  # 添加 batch 維度
        
        print(f"📷 圖片載入成功: {image_path}")
        print(f"   原始尺寸: {original_image.size}")
        print(f"   處理後尺寸: {processed_image.shape}")
        
        return original_image, processed_image
    
    def visualize_first_layer_weights(self, save_path=None, figsize=(20, 15)):
        """
        可視化第一層的權重 (filters/kernels)
        
        Args:
            save_path: 保存路徑
            figsize: 圖片尺寸
        """
        print(f"\n🎨 可視化第一層權重...")
        
        if self.first_layer is None or not hasattr(self.first_layer, 'weight'):
            raise ValueError("第一層沒有權重可以可視化")
        
        # 取得權重
        weights = self.first_layer.weight.data.cpu().numpy()  # shape: (out_channels, in_channels, H, W)
        out_channels, in_channels, kernel_h, kernel_w = weights.shape
        
        print(f"   權重形狀: {weights.shape}")
        print(f"   將顯示前64個filter...")
        
        # 限制顯示的filter數量
        max_filters = min(64, out_channels)
        cols = 8
        rows = (max_filters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
        
        for i in range(max_filters):
            ax = axes[i]
            
            # 對於RGB輸入，取第一個輸入通道或合併所有通道
            if in_channels == 3:  # RGB圖片
                # 將RGB三個通道的權重合併為灰度圖
                filter_weight = np.mean(weights[i], axis=0)
            else:
                filter_weight = weights[i, 0]  # 取第一個通道
            
            # 標準化到 [0, 1]
            filter_weight = (filter_weight - filter_weight.min()) / (filter_weight.max() - filter_weight.min() + 1e-8)
            
            # 顯示權重
            im = ax.imshow(filter_weight, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Filter {i+1}', fontsize=8)
            ax.axis('off')
            
            # 添加顏色條
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隱藏多餘的子圖
        for i in range(max_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'First Layer Weights ({self.model_name})\nShowing {max_filters}/{out_channels} filters', 
                     fontsize=16, y=0.98)
        plt.tight_layout()
        
        # 保存圖片
        if save_path is None:
            save_path = f"{self.model_name}_first_layer_weights.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 第一層權重已保存: {save_path}")
        
        plt.show()
        
        return weights
    
    def get_feature_maps(self, image_tensor):
        """
        獲取圖片經過第一層後的特徵圖
        
        Args:
            image_tensor: 預處理後的圖片tensor
            
        Returns:
            feature_maps: 特徵圖 (numpy array)
        """
        print(f"\n🔍 計算特徵圖...")
        
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            
            # 前向傳播到第一層
            feature_maps = self.first_layer(image_tensor)
            
        feature_maps = feature_maps.cpu().numpy()[0]  # 移除batch維度
        
        print(f"   特徵圖形狀: {feature_maps.shape}")
        print(f"   通道數: {feature_maps.shape[0]}")
        
        return feature_maps
    
    def visualize_feature_maps(self, image_tensor, save_path=None, figsize=(20, 15), max_channels=64):
        """
        可視化特徵圖
        
        Args:
            image_tensor: 預處理後的圖片tensor
            save_path: 保存路徑
            figsize: 圖片尺寸
            max_channels: 最大顯示通道數
        """
        print(f"\n🎨 可視化特徵圖...")
        
        # 獲取特徵圖
        feature_maps = self.get_feature_maps(image_tensor)
        num_channels = feature_maps.shape[0]
        
        # 限制顯示的通道數
        max_channels = min(max_channels, num_channels)
        cols = 8
        rows = (max_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
        
        for i in range(max_channels):
            ax = axes[i]
            
            # 取得特徵圖
            feature_map = feature_maps[i]
            
            # 顯示特徵圖
            im = ax.imshow(feature_map, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Channel {i+1}', fontsize=8)
            ax.axis('off')
            
            # 添加顏色條
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 隱藏多餘的子圖
        for i in range(max_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps After First Layer ({self.model_name})\nShowing {max_channels}/{num_channels} channels', 
                     fontsize=16, y=0.98)
        plt.tight_layout()
        
        # 保存圖片
        if save_path is None:
            save_path = f"{self.model_name}_feature_maps.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 特徵圖已保存: {save_path}")
        
        plt.show()
        
        return feature_maps
    
    def create_attention_heatmap(self, image_tensor, original_image, save_path=None, figsize=(15, 5)):
        """
        創建注意力熱力圖 - 顯示模型對圖片不同區域的注意程度
        
        Args:
            image_tensor: 預處理後的圖片tensor
            original_image: 原始圖片 (PIL)
            save_path: 保存路徑
            figsize: 圖片尺寸
        """
        print(f"\n🔥 創建注意力熱力圖...")
        
        # 獲取特徵圖
        feature_maps = self.get_feature_maps(image_tensor)
        
        # 計算所有通道的平均響應強度
        avg_feature_map = np.mean(feature_maps, axis=0)
        
        # 計算最大響應通道
        max_response_channel = np.argmax(np.sum(feature_maps.reshape(feature_maps.shape[0], -1), axis=1))
        max_feature_map = feature_maps[max_response_channel]
        
        # 將特徵圖縮放到原始圖片尺寸
        original_size = original_image.size  # (width, height)
        avg_heatmap = cv2.resize(avg_feature_map, original_size, interpolation=cv2.INTER_LINEAR)
        max_heatmap = cv2.resize(max_feature_map, original_size, interpolation=cv2.INTER_LINEAR)
        
        # 標準化熱力圖
        avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min() + 1e-8)
        max_heatmap = (max_heatmap - max_heatmap.min()) / (max_heatmap.max() - max_heatmap.min() + 1e-8)
        
        # 創建圖片
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 原始圖片
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # 平均注意力熱力圖
        axes[1].imshow(original_image, alpha=0.6)
        im1 = axes[1].imshow(avg_heatmap, cmap='jet', alpha=0.4, interpolation='bilinear')
        axes[1].set_title('Average Attention Heatmap\n(All Channels)', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 最大響應通道熱力圖
        axes[2].imshow(original_image, alpha=0.6)
        im2 = axes[2].imshow(max_heatmap, cmap='jet', alpha=0.4, interpolation='bilinear')
        axes[2].set_title(f'Max Response Channel\n(Channel {max_response_channel+1})', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Attention Heatmaps ({self.model_name})', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存圖片
        if save_path is None:
            save_path = f"{self.model_name}_attention_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 注意力熱力圖已保存: {save_path}")
        print(f"   最大響應通道: {max_response_channel+1}")
        print(f"   最大響應值: {feature_maps[max_response_channel].max():.4f}")
        
        plt.show()
        
        return avg_heatmap, max_heatmap, max_response_channel
    
    def analyze_channel_responses(self, image_tensor, save_path=None, figsize=(15, 10)):
        """
        分析各通道的響應強度
        
        Args:
            image_tensor: 預處理後的圖片tensor
            save_path: 保存路徑
            figsize: 圖片尺寸
        """
        print(f"\n📊 分析通道響應強度...")
        
        # 獲取特徵圖
        feature_maps = self.get_feature_maps(image_tensor)
        num_channels = feature_maps.shape[0]
        
        # 計算每個通道的統計資訊
        channel_stats = []
        for i in range(num_channels):
            fm = feature_maps[i]
            stats = {
                'channel': i + 1,
                'mean': np.mean(fm),
                'max': np.max(fm),
                'std': np.std(fm),
                'sum': np.sum(fm),
                'positive_ratio': np.sum(fm > 0) / fm.size
            }
            channel_stats.append(stats)
        
        df = pd.DataFrame(channel_stats)
        
        # 創建圖表
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 最大響應值排序
        df_sorted_max = df.sort_values('max', ascending=False)
        axes[0, 0].bar(range(len(df_sorted_max)), df_sorted_max['max'])
        axes[0, 0].set_title('Maximum Response by Channel')
        axes[0, 0].set_xlabel('Channel (sorted by max response)')
        axes[0, 0].set_ylabel('Max Response')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. 平均響應值排序
        df_sorted_mean = df.sort_values('mean', ascending=False)
        axes[0, 1].bar(range(len(df_sorted_mean)), df_sorted_mean['mean'])
        axes[0, 1].set_title('Mean Response by Channel')
        axes[0, 1].set_xlabel('Channel (sorted by mean response)')
        axes[0, 1].set_ylabel('Mean Response')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. 響應總和排序
        df_sorted_sum = df.sort_values('sum', ascending=False)
        axes[1, 0].bar(range(len(df_sorted_sum)), df_sorted_sum['sum'])
        axes[1, 0].set_title('Total Response by Channel')
        axes[1, 0].set_xlabel('Channel (sorted by total response)')
        axes[1, 0].set_ylabel('Total Response')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. 正值比例
        df_sorted_pos = df.sort_values('positive_ratio', ascending=False)
        axes[1, 1].bar(range(len(df_sorted_pos)), df_sorted_pos['positive_ratio'])
        axes[1, 1].set_title('Positive Response Ratio by Channel')
        axes[1, 1].set_xlabel('Channel (sorted by positive ratio)')
        axes[1, 1].set_ylabel('Positive Ratio')
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(f'Channel Response Analysis ({self.model_name})', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存圖片和數據
        if save_path is None:
            save_path = f"{self.model_name}_channel_analysis"
        
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        df.to_csv(f"{save_path}.csv", index=False)
        
        print(f"💾 通道分析已保存: {save_path}.png, {save_path}.csv")
        
        # 顯示統計摘要
        print(f"\n📊 通道響應統計摘要:")
        print(f"   總通道數: {num_channels}")
        print(f"   最高響應: {df['max'].max():.4f} (通道 {df.loc[df['max'].idxmax(), 'channel']})")
        print(f"   最低響應: {df['max'].min():.4f} (通道 {df.loc[df['max'].idxmin(), 'channel']})")
        print(f"   平均響應: {df['mean'].mean():.4f}")
        print(f"   響應標準差: {df['mean'].std():.4f}")
        
        plt.show()
        
        return df

def get_available_models():
    """尋找可用的模型檔案"""
    print("🔍 尋找可用的模型檔案...")
    
    patterns = ["*.pth", "efficientnet*.pth", "convnext*.pth"]
    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(pattern))
    
    model_files = list(set(model_files))
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return model_files

def get_sample_images():
    """獲取範例圖片"""
    print("🖼️ 尋找範例圖片...")
    
    # 可能的圖片路徑
    possible_paths = [
        "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/val/**/*.jpg",
        "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/val/**/*.png",
        "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/val/**/*.jpg",
        "Dataset/preprocessed/val/**/*.jpg",
        "../Dataset/preprocessed/val/**/*.jpg",
        "*.jpg", "*.png", "*.jpeg"
    ]
    
    image_files = []
    for pattern in possible_paths:
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # 去除重複並限制數量
    image_files = list(set(image_files))
    image_files = image_files[:20]  # 最多顯示20張
    
    return image_files

def main():
    """主函數"""
    print("🎨 第一層權重與特徵圖可視化工具")
    print("=" * 60)
    print("📊 功能: 權重可視化, 特徵圖分析, 注意力熱力圖")
    print("=" * 60)
    
    # 1. 選擇模型檔案
    model_files = get_available_models()
    
    if not model_files:
        print("❌ 找不到任何模型檔案！")
        return
    
    print(f"\n📂 找到 {len(model_files)} 個模型檔案:")
    for i, file in enumerate(model_files, 1):
        file_size = os.path.getsize(file) / (1024 * 1024)
        time_str = pd.Timestamp.fromtimestamp(os.path.getmtime(file)).strftime('%m/%d %H:%M')
        print(f"  {i}. {file} ({file_size:.1f}MB, {time_str})")
    
    try:
        choice = int(input(f"\n請選擇模型檔案 (1-{len(model_files)}): ")) - 1
        model_path = model_files[choice]
    except (ValueError, IndexError):
        print("❌ 選擇無效，使用最新的模型檔案")
        model_path = model_files[0]
    
    # 2. 初始化可視化器
    try:
        visualizer = FirstLayerVisualizer(model_path)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return
    
    # 3. 可視化第一層權重
    print(f"\n🎨 步驟1: 可視化第一層權重")
    try:
        weights = visualizer.visualize_first_layer_weights()
    except Exception as e:
        print(f"❌ 權重可視化失敗: {e}")
        return
    
    # 4. 選擇測試圖片
    image_files = get_sample_images()
    
    if not image_files:
        print("❌ 找不到範例圖片！")
        return
    
    print(f"\n🖼️ 找到 {len(image_files)} 張圖片:")
    for i, file in enumerate(image_files[:10], 1):  # 只顯示前10張
        print(f"  {i}. {os.path.basename(file)}")
    
    try:
        choice = int(input(f"\n請選擇圖片 (1-{min(10, len(image_files))}): ")) - 1
        image_path = image_files[choice]
    except (ValueError, IndexError):
        print("❌ 選擇無效，使用第一張圖片")
        image_path = image_files[0]
    
    # 5. 載入圖片
    try:
        original_image, processed_image = visualizer.load_image(image_path)
    except Exception as e:
        print(f"❌ 圖片載入失敗: {e}")
        return
    
    # 6. 可視化特徵圖
    print(f"\n🎨 步驟2: 可視化特徵圖")
    try:
        feature_maps = visualizer.visualize_feature_maps(processed_image)
    except Exception as e:
        print(f"❌ 特徵圖可視化失敗: {e}")
        return
    
    # 7. 創建注意力熱力圖
    print(f"\n🎨 步驟3: 創建注意力熱力圖")
    try:
        avg_heatmap, max_heatmap, max_channel = visualizer.create_attention_heatmap(
            processed_image, original_image
        )
    except Exception as e:
        print(f"❌ 注意力熱力圖創建失敗: {e}")
        return
    
    # 8. 分析通道響應
    print(f"\n🎨 步驟4: 分析通道響應")
    try:
        channel_df = visualizer.analyze_channel_responses(processed_image)
    except Exception as e:
        print(f"❌ 通道分析失敗: {e}")
        return
    
    print(f"\n🎉 所有分析完成！")
    print(f"📁 結果已保存到當前目錄")
    print(f"🖼️ 分析圖片: {os.path.basename(image_path)}")

if __name__ == "__main__":
    main()