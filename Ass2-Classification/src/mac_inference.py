#!/usr/bin/env python3
"""
🔮 EfficientNet 模型推理工具 - Mac 版本

功能：
- 從 Dataset/test 目錄讀取所有測試圖片
- 使用訓練好的 EfficientNet 模型進行預測
- 輸出 CSV 格式結果：檔名, 預測結果
- 針對 Mac 系統優化，支援 MPS（Metal Performance Shaders）加速
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MacEfficientNetInference:
    """
    Mac 優化版 EfficientNet 模型推理器
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化推理器
        
        Args:
            model_path: 模型檔案路徑 (.pth)
            device: 計算設備
        """
        # Mac 設備優先級：MPS > CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.model_name = None
        
        print(f"🔮 Mac EfficientNet 推理器")
        print(f"🖥️ 使用設備: {self.device}")
        
        # 載入模型
        self.load_model(model_path)
        
        # 準備變換
        self.transform = self._get_inference_transform()
        
    def load_model(self, model_path):
        """載入訓練好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        
        print(f"📂 載入模型: {model_path}")
        
        # 載入 checkpoint，強制使用 CPU 以避免設備不匹配
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 獲取模型資訊
        self.model_name = checkpoint['model_name']
        num_classes = checkpoint['num_classes']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        print(f"🎯 模型: {self.model_name}")
        print(f"📝 類別數: {num_classes}")
        print(f"🏷️ 類別: {list(self.class_to_idx.keys())}")
        
        # 重建模型架構
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,  # 不需要預訓練權重
            num_classes=num_classes
        )
        
        # 載入權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # 設為推理模式
        
        print("✅ 模型載入完成！")
    
    def _get_inference_transform(self):
        """獲取推理用的圖片變換"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path):
        """
        預測單張圖片
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            str: 預測的類別名稱
        """
        try:
            # 載入並預處理圖片
            image = Image.open(image_path).convert('RGB')
            
            # 變換圖片
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_idx = predicted.item()
                predicted_class = self.idx_to_class[predicted_idx]
            
            return predicted_class
        
        except Exception as e:
            print(f"❌ 預測失敗 {image_path}: {e}")
            return "unknown"
    
    def predict_test_dataset(self, test_dir="Dataset/test", output_file="predictions.csv"):
        """
        對測試資料集進行批量預測並輸出 CSV
        
        Args:
            test_dir: 測試圖片目錄
            output_file: 輸出 CSV 檔案名稱
            
        Returns:
            pd.DataFrame: 預測結果
        """
        print(f"\n📁 開始處理測試資料集: {test_dir}")
        
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
            # 提取數字部分進行排序
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
        
        # 準備結果列表
        results = []
        
        # 批量預測（使用進度條）
        print("🚀 開始批量預測...")
        for image_path in tqdm(image_paths, desc="預測進度"):
            filename = os.path.basename(image_path)
            # 移除副檔名 (.jpg, .png 等)
            id_name = os.path.splitext(filename)[0]
            predicted_class = self.predict_single(image_path)
            
            results.append({
                'id': id_name,
                'character': predicted_class
            })
        
        # 創建 DataFrame
        df = pd.DataFrame(results)
        
        # 保存 CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 預測結果已保存至: {output_file}")
        
        # 顯示統計資訊
        print(f"\n📊 預測統計:")
        print(f"   總圖片數: {len(df)}")
        print(f"   預測類別分布:")
        for class_name, count in df['character'].value_counts().items():
            print(f"     {class_name}: {count}")
        
        return df
    
    def get_model_info(self):
        """獲取模型資訊"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        info = {
            'model_name': self.model_name,
            'num_classes': len(self.idx_to_class),
            'total_parameters': f"{total_params/1e6:.1f}M",
            'device': str(self.device),
            'class_names': list(self.class_to_idx.keys())
        }
        
        return info

def main():
    """主函數 - 命令列介面"""
    parser = argparse.ArgumentParser(description="Mac EfficientNet 模型推理工具")
    parser.add_argument('--model', '-m', required=True, help='模型檔案路徑 (.pth)')
    parser.add_argument('--test-dir', '-t', default='Dataset/test', help='測試圖片目錄')
    parser.add_argument('--output', '-o', default='predictions.csv', help='輸出 CSV 檔案名稱')
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto', help='計算設備')
    
    args = parser.parse_args()
    
    # 設定設備
    if args.device == 'auto':
        device = None
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 初始化推理器
    try:
        inferencer = MacEfficientNetInference(args.model, device=device)
        
        # 顯示模型資訊
        info = inferencer.get_model_info()
        print(f"\n📊 模型資訊:")
        print(f"   模型: {info['model_name']}")
        print(f"   類別數: {info['num_classes']}")
        print(f"   參數量: {info['total_parameters']}")
        print(f"   設備: {info['device']}")
        
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return 1
    
    # 執行批量推理
    try:
        # 確保測試目錄存在
        if not os.path.exists(args.test_dir):
            print(f"❌ 測試目錄不存在: {args.test_dir}")
            return 1
        
        # 開始預測
        df = inferencer.predict_test_dataset(args.test_dir, args.output)
        
        if df is not None:
            print(f"\n✅ 推理完成！")
            print(f"📄 結果檔案: {args.output}")
            print(f"🔢 總預測數: {len(df)}")
        else:
            print("❌ 推理失敗！")
            return 1
            
    except Exception as e:
        print(f"❌ 推理過程發生錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 如果沒有命令列參數，使用互動模式
    import sys
    
    if len(sys.argv) == 1:
        print("🔮 Mac EfficientNet 推理工具 - 互動模式")
        print("=" * 50)
        
        # 尋找可用的模型
        possible_paths = [
            "*.pth",
            "../*.pth", 
            "models/*.pth",
            "../models/*.pth"
        ]
        
        model_files = []
        for pattern in possible_paths:
            model_files.extend(glob.glob(pattern))
        
        if not model_files:
            print("❌ 找不到模型檔案 (.pth)")
            model_path = input("請輸入模型檔案路徑: ").strip()
        else:
            print("🔍 找到以下模型檔案:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")
            
            choice = input(f"請選擇模型 (1-{len(model_files)}): ").strip()
            try:
                model_path = model_files[int(choice)-1]
            except (ValueError, IndexError):
                model_path = model_files[0]
                print(f"使用預設模型: {model_path}")
        
        # 設定測試目錄
        test_dir = input("測試圖片目錄 (預設: Dataset/test): ").strip()
        if not test_dir:
            test_dir = "Dataset/test"
        
        # 設定輸出檔案
        output_file = input("輸出檔案名稱 (預設: predictions.csv): ").strip()
        if not output_file:
            output_file = "predictions.csv"
        
        # 初始化推理器
        try:
            print("\n🚀 初始化推理器...")
            inferencer = MacEfficientNetInference(model_path)
            
            # 顯示模型資訊
            info = inferencer.get_model_info()
            print(f"\n📊 模型資訊:")
            print(f"   模型: {info['model_name']}")
            print(f"   類別數: {info['num_classes']}")
            print(f"   參數量: {info['total_parameters']}")
            print(f"   設備: {info['device']}")
            print(f"   類別: {', '.join(info['class_names'])}")
            
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            exit(1)
        
        # 執行推理
        try:
            print(f"\n🎯 開始預測...")
            df = inferencer.predict_test_dataset(test_dir, output_file)
            
            if df is not None:
                print(f"\n🎉 推理完成！")
                print(f"📄 結果已保存至: {output_file}")
                
                # 顯示前幾個結果
                print(f"\n📋 前 10 個預測結果:")
                print(df.head(10).to_string(index=False))
                
            else:
                print("❌ 推理失敗！")
                
        except Exception as e:
            print(f"❌ 推理過程發生錯誤: {e}")
    else:
        # 命令列模式
        exit(main())
