#!/usr/bin/env python3
"""
MemoryViT 模型載入和推論工具
載入已訓練的模型進行單張圖片預測或批量推論
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
from vit_pytorch.learnable_memory_vit import ViT, Adapter

class MemoryViTPredictor:
    def __init__(self, model_path, device='cuda'):
        """
        初始化預測器
        
        Args:
            model_path: 模型檔案路徑
            device: 運算裝置
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # 載入模型
        self.load_model()
        
    def load_model(self):
        """載入預訓練模型"""
        print(f"🔄 載入模型: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 獲取模型配置
            if 'training_config' in checkpoint:
                config = checkpoint['training_config']
                self.image_size = config.get('image_size', 224)
                self.num_classes = config.get('num_classes', 50)
            else:
                self.image_size = 224
                self.num_classes = 50
                
            # 獲取類別映射
            if 'class_mapping' in checkpoint:
                self.class_to_idx = checkpoint['class_mapping']['class_to_idx']
                self.idx_to_class = checkpoint['class_mapping']['idx_to_class']
                # 將字符串鍵轉換為整數鍵
                self.idx_to_class = {int(k): v for k, v in self.idx_to_class.items()}
            else:
                raise ValueError("模型檔案中沒有類別映射信息")
            
            # 設定圖像變換
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 重建模型架構
            self.base_vit = ViT(
                image_size=self.image_size,
                patch_size=16,
                num_classes=1000,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1,
                emb_dropout=0.1
            ).to(self.device)
            
            self.character_adapter = Adapter(
                vit=self.base_vit,
                num_classes=self.num_classes,
                num_memories_per_layer=20
            ).to(self.device)
            
            # 載入權重
            self.character_adapter.load_state_dict(checkpoint['model_state_dict'])
            self.character_adapter.eval()
            
            # 顯示模型資訊
            if 'val_acc' in checkpoint:
                print(f"✅ 模型載入成功！驗證準確率: {checkpoint['val_acc']:.2f}%")
            
            if 'training_config' in checkpoint:
                config = checkpoint['training_config']
                print(f"📊 模型配置:")
                print(f"   圖像尺寸: {config.get('image_size', 224)}x{config.get('image_size', 224)}")
                print(f"   類別數量: {self.num_classes}")
                print(f"   批次大小: {config.get('batch_size', 'N/A')}")
                print(f"   混合精度: {'是' if config.get('use_mixed_precision', False) else '否'}")
            
            print(f"🎭 可識別的角色: {len(self.class_to_idx)} 個")
            
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            raise
    
    def predict_single_image(self, image_path, top_k=5):
        """
        預測單張圖片
        
        Args:
            image_path: 圖片路徑
            top_k: 返回前 k 個預測結果
            
        Returns:
            list: [(class_name, probability), ...]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"圖片不存在: {image_path}")
        
        # 載入並預處理圖片
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"無法載入圖片 {image_path}: {e}")
        
        # 推論
        with torch.no_grad():
            outputs = self.character_adapter(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        # 整理結果
        results = []
        for i in range(top_probs.size(1)):
            class_idx = top_indices[0][i].item()
            class_name = self.idx_to_class[class_idx]
            prob = top_probs[0][i].item()
            results.append((class_name, prob))
        
        return results
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        批量預測多張圖片
        
        Args:
            image_paths: 圖片路徑列表
            batch_size: 批次大小
            
        Returns:
            list: [predictions, ...] 每個預測是 (class_name, probability) 的列表
        """
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # 載入並預處理批次圖片
            for j, path in enumerate(batch_paths):
                try:
                    if os.path.exists(path):
                        image = Image.open(path).convert('RGB')
                        image_tensor = self.transform(image)
                        batch_images.append(image_tensor)
                        valid_indices.append(i + j)
                    else:
                        print(f"⚠️ 跳過不存在的圖片: {path}")
                except Exception as e:
                    print(f"⚠️ 跳過無法載入的圖片 {path}: {e}")
            
            if batch_images:
                # 堆疊成批次
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # 推論
                with torch.no_grad():
                    outputs = self.character_adapter(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    top_probs, top_indices = torch.topk(probabilities, 5)
                
                # 整理批次結果
                for k in range(len(batch_images)):
                    batch_results = []
                    for m in range(5):
                        class_idx = top_indices[k][m].item()
                        class_name = self.idx_to_class[class_idx]
                        prob = top_probs[k][m].item()
                        batch_results.append((class_name, prob))
                    all_results.append(batch_results)
            
        return all_results
    
    def get_class_list(self):
        """獲取所有類別名稱"""
        return list(self.class_to_idx.keys())
    
    def get_model_info(self):
        """獲取模型詳細資訊"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            info = {
                'model_path': self.model_path,
                'num_classes': self.num_classes,
                'image_size': self.image_size,
                'validation_accuracy': checkpoint.get('val_acc', 'N/A'),
                'training_accuracy': checkpoint.get('train_acc', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'classes': list(self.class_to_idx.keys())
            }
            
            if 'training_config' in checkpoint:
                info.update(checkpoint['training_config'])
            
            return info
        except Exception as e:
            return {'error': str(e)}

def demo_prediction():
    """演示預測功能"""
    print("🎭 MemoryViT 預測演示")
    print("=" * 50)
    
    # 尋找模型檔案
    model_files = [
        'best_memory_vit_character_classifier.pth',
        'memoryvit_model_acc*.pth'
    ]
    
    model_path = None
    for pattern in model_files:
        if '*' in pattern:
            import glob
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                break
        elif os.path.exists(pattern):
            model_path = pattern
            break
    
    if not model_path:
        print("❌ 找不到模型檔案")
        print("請確認以下檔案之一存在:")
        for f in model_files:
            print(f"  - {f}")
        return
    
    try:
        # 載入預測器
        predictor = MemoryViTPredictor(model_path)
        
        # 顯示模型資訊
        info = predictor.get_model_info()
        print(f"\n📊 模型資訊:")
        for key, value in info.items():
            if key != 'classes':
                print(f"   {key}: {value}")
        
        # 獲取測試圖片
        test_image = input("\n請輸入測試圖片路徑 (或按 Enter 跳過): ").strip()
        
        if test_image and os.path.exists(test_image):
            print(f"\n🔍 預測圖片: {test_image}")
            results = predictor.predict_single_image(test_image, top_k=5)
            
            print("\n🎯 預測結果:")
            for i, (class_name, prob) in enumerate(results, 1):
                print(f"   {i}. {class_name}: {prob*100:.2f}%")
        else:
            print("⚠️ 跳過圖片預測")
        
        # 顯示所有可識別的角色
        classes = predictor.get_class_list()
        print(f"\n🎭 可識別的角色 ({len(classes)} 個):")
        for i, class_name in enumerate(sorted(classes), 1):
            print(f"   {i:2d}. {class_name}")
            
    except Exception as e:
        print(f"❌ 演示失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_prediction()