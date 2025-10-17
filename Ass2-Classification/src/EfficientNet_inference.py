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
from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse

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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    def predict_single(self, image_path, top_k=5):
        """
        預測單張圖片
        
        Args:
            image_path: 圖片路徑
            top_k: 返回前 k 個預測結果
            
        Returns:
            dict: 預測結果
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"找不到圖片: {image_path}")
        
        # 載入並預處理圖片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"無法載入圖片 {image_path}: {e}")
        
        # 變換圖片
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 獲取前 k 個結果
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.idx_to_class)))
        
        # 整理結果
        results = {
            'image_path': image_path,
            'predictions': []
        }
        
        for i in range(len(top_indices[0])):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            class_name = self.idx_to_class[class_idx]
            
            results['predictions'].append({
                'class_name': class_name,
                'confidence': prob,
                'class_idx': class_idx
            })
        
        return results
    
    def predict_batch(self, image_folder, output_file=None, top_k=3):
        """
        批量預測資料夾中的圖片
        
        Args:
            image_folder: 圖片資料夾路徑
            output_file: 輸出結果檔案 (JSON)
            top_k: 每張圖片返回前 k 個結果
            
        Returns:
            list: 所有預測結果
        """
        print(f"\n📁 批量推理: {image_folder}")
        
        # 支援的圖片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.webp']
        
        # 收集所有圖片
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))
        
        if not image_paths:
            print("❌ 找不到任何圖片！")
            return []
        
        print(f"🔍 找到 {len(image_paths)} 張圖片")
        
        # 批量預測
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path, top_k=top_k)
                all_results.append(result)
                
                # 顯示進度
                if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                    print(f"⚡ 進度: {i+1}/{len(image_paths)} ({(i+1)/len(image_paths)*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ 預測失敗 {image_path}: {e}")
                continue
        
        # 保存結果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"💾 結果已保存: {output_file}")
        
        print(f"✅ 批量推理完成！成功預測 {len(all_results)} 張圖片")
        return all_results
    
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
    parser.add_argument('--folder', '-f', help='圖片資料夾路徑')
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
        
        # 選擇推理模式
        print(f"\n🎯 選擇推理模式:")
        print("1. 單張圖片預測")
        print("2. 批量圖片預測")
        
        mode = input("請選擇 (1/2): ").strip()
        
        if mode == "1":
            image_path = input("請輸入圖片路徑: ").strip()
            show = input("顯示結果圖？(y/n): ").strip().lower() == 'y'
            
            try:
                if show:
                    inferencer.predict_and_show(image_path)
                else:
                    result = inferencer.predict_single(image_path)
                    print(f"\n🎯 預測結果:")
                    for i, pred in enumerate(result['predictions']):
                        print(f"{i+1}. {pred['class_name']}: {pred['confidence']:.3f}")
            except Exception as e:
                print(f"❌ 預測失敗: {e}")
        
        elif mode == "2":
            folder_path = input("請輸入圖片資料夾路徑: ").strip()
            output_file = input("輸出檔案名稱 (預設 results.json): ").strip() or "results.json"
            
            try:
                inferencer.predict_batch(folder_path, output_file=output_file)
            except Exception as e:
                print(f"❌ 批量推理失敗: {e}")
    else:
        main()