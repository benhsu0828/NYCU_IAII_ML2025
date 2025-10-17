#!/usr/bin/env python3
"""
🏃 ResNet Simpson 角色分類器 - 穩定高效版本

ResNet 優勢：
- 訓練穩定，容易調參
- 速度比 ViT 快 3-5 倍
- 在各種資料集上表現穩定
- 成熟的架構，bug 少
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class ResNetCharacterClassifier:
    """
    使用 ResNet 的快速角色分類器
    """
    
    def __init__(self, num_classes=50, model_type='resnet50', device=None):
        """
        初始化分類器
        
        Args:
            num_classes: 類別數量
            model_type: ResNet 類型
                - resnet18: 最快 (11M 參數)
                - resnet34: 快速 (21M 參數)  
                - resnet50: 平衡 (25M 參數) [推薦]
                - resnet101: 準確 (44M 參數)
            device: 計算設備
        """
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🎯 初始化 {model_type} 分類器")
        print(f"🖥️ 使用設備: {self.device}")
        
        # 初始化模型
        self.model = self._create_model()
        self.class_to_idx = {}
        
    def _create_model(self):
        """創建 ResNet 模型"""
        # 載入預訓練模型
        if self.model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif self.model_type == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif self.model_type == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif self.model_type == 'resnet101':
            model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")
        
        # 修改最後一層
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        # 計算參數
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型統計:")
        print(f"   總參數: {total_params/1e6:.1f}M")
        print(f"   可訓練參數: {trainable_params/1e6:.1f}M")
        
        return model.to(self.device)
    
    def get_transforms(self, is_training=True):
        """獲取資料變換"""
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_data(self, data_paths):
        """準備資料集"""
        print("\n📊 準備資料...")
        
        # 訓練資料
        train_transform = self.get_transforms(is_training=True)
        train_dataset = datasets.ImageFolder(
            root=data_paths['train'],
            transform=train_transform
        )
        
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"✅ 訓練集: {len(train_dataset)} 張圖片")
        print(f"📝 類別數: {len(self.class_to_idx)}")
        
        # 驗證資料
        val_dataset = None
        if data_paths.get('val') and os.path.exists(data_paths['val']):
            val_transform = self.get_transforms(is_training=False)
            val_dataset = datasets.ImageFolder(
                root=data_paths['val'],
                transform=val_transform
            )
            print(f"✅ 驗證集: {len(val_dataset)} 張圖片")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset=None, 
              batch_size=64, epochs=25, lr=1e-3):
        """
        快速訓練模型
        
        Args:
            train_dataset: 訓練資料集
            val_dataset: 驗證資料集
            batch_size: batch size
            epochs: 訓練輪數
            lr: 學習率
        """
        print(f"\n🚀 開始訓練 {self.model_type}...")
        
        # 準備資料載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        # 設定優化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # 訓練歷史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        print(f"📊 訓練設定:")
        print(f"   Batch size: {batch_size}")
        print(f"   學習率: {lr}")
        print(f"   訓練輪數: {epochs}")
        
        # 開始訓練
        for epoch in range(epochs):
            start_time = time.time()
            
            # 訓練階段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # 更新進度條
                acc = 100.0 * train_correct / train_total
                pbar.set_postfix({
                    'Loss': f'{train_loss/len(pbar):.4f}',
                    'Acc': f'{acc:.2f}%'
                })
            
            train_acc = 100.0 * train_correct / train_total
            
            # 驗證階段
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                val_loss_sum = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss_sum += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_loss = val_loss_sum / len(val_loader)
                val_acc = 100.0 * val_correct / val_total
            
            # 更新學習率
            scheduler.step()
            
            # 記錄歷史
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # 計算時間
            epoch_time = time.time() - start_time
            
            # 顯示進度
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
            if val_loader:
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_{self.model_type}_acc{val_acc:.1f}.pth")
            
            print("-" * 50)
        
        print(f"\n✅ 訓練完成！")
        if val_loader:
            print(f"🏆 最佳驗證準確率: {best_val_acc:.2f}%")
        
        return history
    
    def save_model(self, filename):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
        
        torch.save(checkpoint, filename)
        print(f"💾 模型已保存: {filename}")

def main():
    """主函數"""
    print("🏃 ResNet Simpson 角色分類器")
    print("=" * 50)
    
    # 檢查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用設備: {device}")
    
    # 自動檢測資料路徑
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    possible_paths = [
        {
            'train': os.path.join(base_dir, 'Dataset', 'processed', 'train'),
            'val': os.path.join(base_dir, 'Dataset', 'processed', 'val')
        },
        {
            'train': os.path.join(base_dir, 'Dataset', 'train'),
            'val': os.path.join(base_dir, 'Dataset', 'val')
        }
    ]
    
    data_paths = None
    for paths in possible_paths:
        if os.path.exists(paths['train']):
            data_paths = paths
            print(f"✅ 找到資料路徑: {paths['train']}")
            break
    
    if data_paths is None:
        print("❌ 找不到訓練資料！")
        return
    
    # 選擇模型
    print("\n🎯 選擇 ResNet 模型:")
    print("1. resnet18 - 最快 (11M 參數)")
    print("2. resnet50 - 平衡 (25M 參數) [推薦]")
    print("3. resnet101 - 準確 (44M 參數)")
    
    choice = input("請選擇 (1/2/3，預設2): ").strip()
    model_mapping = {
        '1': 'resnet18',
        '2': 'resnet50', 
        '3': 'resnet101'
    }
    model_type = model_mapping.get(choice, 'resnet50')
    
    # 初始化分類器
    classifier = ResNetCharacterClassifier(
        num_classes=50,
        model_type=model_type,
        device=device
    )
    
    # 準備資料
    train_dataset, val_dataset = classifier.prepare_data(data_paths)
    
    # 訓練參數
    batch_size = int(input("Batch size (預設 64): ") or "64")
    epochs = int(input("訓練輪數 (預設 25): ") or "25")
    lr = float(input("學習率 (預設 1e-3): ") or "1e-3")
    
    # 開始訓練
    history = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr
    )
    
    print("\n🎉 訓練完成！")

if __name__ == "__main__":
    main()