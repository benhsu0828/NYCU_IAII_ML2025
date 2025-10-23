#!/usr/bin/env python3
"""
🔮 EfficientNet 模型推理工具 - Simpson 角色預測

功能：
- 載入訓練好的 EfficientNet 模型
- 對單張圖片或批量圖片進行預測
- 支援信心分數和前 N 預測
- 高效推理，適合部署使用
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from tqdm import tqdm

class EfficientNetInference:
    """
    EfficientNet 模型推理器
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化推理器
        
        Args:
            model_path: 模型檔案路徑 (.pth)
            device: 計算設備
        """
        # Windows 設備優先級：CUDA > CPU  
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # 顯示 GPU 資訊
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🎮 偵測到 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.device = torch.device("cpu")
                print("⚠️ 未偵測到 GPU，使用 CPU")
        else:
            self.device = device
            
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.model_name = None
        
        print(f"🔮 EfficientNet 推理器")
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
        
        # 載入 checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 獲取模型資訊
        self.model_name = checkpoint['model_name']
        num_classes = checkpoint['num_classes']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        print(f"🎯 模型: {self.model_name}")
        print(f"📝 類別數: {num_classes}")
        
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
    
    def predict_single(self, image_path, top_k=1):
        """
        預測單張圖片
        
        Args:
            image_path: 圖片路徑
            top_k: 返回前 k 個預測結果
            
        Returns:
            str 或 list: 預測的類別名稱
        """
        try:
            # 載入並預處理圖片
            image = Image.open(image_path).convert('RGB')
            
            # 變換圖片
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # GPU 推理
            with torch.no_grad():
                if self.device.type == 'cuda':
                    # GPU 記憶體優化
                    torch.cuda.empty_cache()
                
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # 獲取 top-k 結果
                top_prob, top_indices = torch.topk(probabilities, top_k)
                
                if top_k == 1:
                    predicted_idx = top_indices[0].item()
                    predicted_class = self.idx_to_class[predicted_idx]
                    return predicted_class
                else:
                    results = []
                    for i in range(top_k):
                        idx = top_indices[i].item()
                        prob = top_prob[i].item()
                        class_name = self.idx_to_class[idx]
                        results.append({
                            'class': class_name,
                            'confidence': prob
                        })
                    return results
            
        except Exception as e:
            print(f"❌ 預測失敗 {image_path}: {e}")
            return "unknown" if top_k == 1 else [{'class': 'unknown', 'confidence': 0.0}]
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        批次預測 - GPU 加速版本
        
        Args:
            image_paths: 圖片路徑列表
            batch_size: 批次大小
            
        Returns:
            list: 預測結果列表
        """
        predictions = []
        
        # 計算總批次數
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        # 添加進度條
        with tqdm(total=len(image_paths), desc="GPU 批次推理進度", unit="張") as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                valid_indices = []
                
                # 載入批次圖片
                for idx, path in enumerate(batch_paths):
                    try:
                        image = Image.open(path).convert('RGB')
                        tensor = self.transform(image)
                        batch_images.append(tensor)
                        valid_indices.append(idx)
                    except Exception as e:
                        print(f"❌ 載入失敗 {path}: {e}")
                        predictions.append("unknown")
                
                if batch_images:
                    # 批次推理
                    try:
                        batch_tensor = torch.stack(batch_images).to(self.device)
                        
                        with torch.no_grad():
                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                            
                            outputs = self.model(batch_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            
                            # 轉換為類別名稱
                            for j, pred_idx in enumerate(predicted.cpu().numpy()):
                                if j < len(valid_indices):
                                    predicted_class = self.idx_to_class[pred_idx]
                                    # 插入正確位置
                                    while len(predictions) <= i + valid_indices[j]:
                                        predictions.append("unknown")
                                    predictions[i + valid_indices[j]] = predicted_class
                    
                    except Exception as e:
                        print(f"❌ 批次推理失敗: {e}")
                        # 回退到單張推理
                        for path in batch_paths:
                            predictions.append(self.predict_single(path))
                
                # 更新進度條
                pbar.update(len(batch_paths))
        
        return predictions
    
    def predict_test_dataset(self, test_dir="Dataset/test", output_file="predictions.csv", batch_size=32, use_gpu_batch=True):
        """
        對測試資料集進行批量預測並輸出 CSV - GPU 優化版本
        
        Args:
            test_dir: 測試圖片目錄
            output_file: 輸出 CSV 檔案名稱
            batch_size: 批次大小 (GPU 時建議 32-64)
            use_gpu_batch: 是否使用 GPU 批次推理
            
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
        
        # 根據設備選擇推理方式
        if self.device.type == 'cuda' and use_gpu_batch:
            print(f"🎮 使用 GPU 批次推理 (批次大小: {batch_size})")
            predicted_classes = self.predict_batch(image_paths, batch_size)
        else:
            print(f"💻 使用逐張推理")
            predicted_classes = []
            for image_path in tqdm(image_paths, desc="逐張推理進度", unit="張"):
                predicted_class = self.predict_single(image_path)
                predicted_classes.append(predicted_class)
        
        # 準備結果
        results = []
        for image_path, predicted_class in zip(image_paths, predicted_classes):
            filename = os.path.basename(image_path)
            # 移除副檔名 (.jpg, .png 等)
            id_name = os.path.splitext(filename)[0]
            
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
        print(f"   使用設備: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU 記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"   預測類別分布:")
        for class_name, count in df['character'].value_counts().head(10).items():
            print(f"     {class_name}: {count}")
        
        return df
    
    def predict_and_show(self, image_path, save_plot=True):
        """
        預測圖片並視覺化結果
        
        Args:
            image_path: 圖片路徑
            save_plot: 是否保存結果圖片
        """
        # 預測
        result = self.predict_single(image_path, top_k=5)
        
        # 載入原圖
        image = Image.open(image_path).convert('RGB')
        
        # 繪製結果
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 顯示原圖
        ax1.imshow(image)
        ax1.set_title(f'原圖: {os.path.basename(image_path)}')
        ax1.axis('off')
        
        # 顯示預測結果
        predictions = result['predictions']
        class_names = [pred['class_name'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]
        
        bars = ax2.barh(range(len(class_names)), confidences)
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels(class_names)
        ax2.set_xlabel('信心分數')
        ax2.set_title('預測結果 (前5名)')
        ax2.set_xlim(0, 1)
        
        # 添加數值標籤
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(conf + 0.01, i, f'{conf:.3f}', 
                    va='center', fontsize=10)
        
        # 標記最佳預測
        if confidences:
            bars[0].set_color('gold')
            ax2.text(0.5, len(class_names), 
                    f'最佳預測: {class_names[0]} ({confidences[0]:.3f})',
                    ha='center', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_plot:
            output_name = f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"📊 結果圖已保存: {output_name}")
        
        plt.show()
        
        # 打印結果
        print(f"\n🎯 預測結果:")
        for i, pred in enumerate(predictions):
            icon = "🏆" if i == 0 else f"{i+1}."
            print(f"{icon} {pred['class_name']}: {pred['confidence']:.3f}")
        
        return result
    
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
    
    def print_model_info(self):
        """打印模型資訊"""
        info = self.get_model_info()
        
        print(f"\n📊 模型資訊:")
        print(f"   模型: {info['model_name']}")
        print(f"   類別數: {info['num_classes']}")
        print(f"   參數量: {info['total_parameters']}")
        print(f"   設備: {info['device']}")
        print(f"   類別列表: {info['class_names'][:10]}..." if len(info['class_names']) > 10 else f"   類別列表: {info['class_names']}")

def main():
    """主函數 - 命令列介面"""
    parser = argparse.ArgumentParser(description="EfficientNet 模型推理工具")
    parser.add_argument('--model', '-m', required=True, help='模型檔案路徑 (.pth)')
    parser.add_argument('--image', '-i', help='單張圖片路徑')
    parser.add_argument('--folder', '-f',default='/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/raw/test',help='圖片資料夾路徑(Default:/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/raw/test)')
    parser.add_argument('--output', '-o', help='輸出結果檔案 (JSON)')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='前 k 個預測結果')
    parser.add_argument('--show', action='store_true', help='顯示預測結果圖')
    
    args = parser.parse_args()
    
    # 初始化推理器
    try:
        inferencer = EfficientNetInference(args.model)
        inferencer.print_model_info()
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return
    
    # 執行推理
    if args.image:
        # 單張圖片推理
        try:
            if args.show:
                result = inferencer.predict_and_show(args.image)
            else:
                result = inferencer.predict_single(args.image, top_k=args.top_k)
                print(f"\n🎯 預測結果:")
                for i, pred in enumerate(result['predictions']):
                    print(f"{i+1}. {pred['class_name']}: {pred['confidence']:.3f}")
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
    
    elif args.folder:
        # 批量推理
        try:
            results = inferencer.predict_batch(
                args.folder, 
                output_file=args.output, 
                top_k=args.top_k
            )
        except Exception as e:
            print(f"❌ 批量推理失敗: {e}")
    
    else:
        print("❌ 請指定 --image 或 --folder 參數")
        print("💡 使用範例:")
        print("   python EfficientNet_inference.py -m best_model.pth -i test_image.jpg --show")
        print("   python EfficientNet_inference.py -m best_model.pth -f test_folder/ -o results.json")

if __name__ == "__main__":
    # 如果沒有命令列參數，使用互動模式
    import sys
    
    if len(sys.argv) == 1:
        print("🔮 EfficientNet 推理工具 - 互動模式")
        print("=" * 50)
        
        # 尋找可用的模型
        model_files = glob.glob("*.pth") + glob.glob("best_*.pth")
        
        if not model_files:
            print("❌ 當前目錄找不到模型檔案 (.pth)")
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
        
        # 初始化推理器
        try:
            inferencer = EfficientNetInference(model_path)
            inferencer.print_model_info()
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            exit(1)
        
        import platform
        is_wsl = "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")

        if is_wsl:
            test_dir = input("測試圖片目錄 (預設: /mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/raw/test): ").strip()
            if not test_dir:
                test_dir = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/raw/test"
        else:  
            # 設定測試目錄 - Windows 路徑
            test_dir = input("測試圖片目錄 (預設: E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/raw/test): ").strip()
            if not test_dir:
                test_dir = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/raw/test"

        # 設定批次大小 (GPU 優化)
        batch_size_input = input("批次大小 (預設: 32, GPU 建議 32-64): ").strip()
        try:
            batch_size = int(batch_size_input) if batch_size_input else 32
        except ValueError:
            batch_size = 32
        
        # 設定輸出檔案
        model_name = model_files[int(choice)-1].split('_')[0]
        output_file = input(f"輸出檔案名稱 (預設: {model_name}_predictions.csv): ").strip()
        if not output_file:
            output_file = f"{model_name}_predictions.csv"

        # 執行推理
        try:
            print(f"\n🎯 開始預測...")
            print(f"📊 批次大小: {batch_size}")
            print(f"🎮 設備: {inferencer.device}")
            
            df = inferencer.predict_test_dataset(
                test_dir, 
                output_file, 
                batch_size=batch_size,
                use_gpu_batch=inferencer.device.type == 'cuda'
            )
            
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