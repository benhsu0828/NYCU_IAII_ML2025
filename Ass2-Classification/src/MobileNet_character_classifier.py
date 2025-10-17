#!/usr/bin/env python3
"""
📱 MobileNet Simpson 角色分類器 - 超輕量版本

MobileNet 優勢：
- 最輕量，適合快速實驗
- 速度極快，比 ViT 快 10+ 倍
- 記憶體需求最低
- 適合資源受限環境
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

class MobileNetCharacterClassifier:
    """
    使用 MobileNet 的超輕量角色分類器
    """
    
    def __init__(self, num_classes=50, model_type='mobilenet_v2', device=None):
        """
        初始化分類器
        
        Args:
            num_classes: 類別數量
            model_type: MobileNet 類型
                - mobilenet_v2: 經典版本 (3.5M 參數)
                - mobilenet_v3_small: 超輕量 (2.5M 參數)
                - mobilenet_v3_large: 平衡版 (5.5M 參數)
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
        """創建 MobileNet 模型"""
        # 載入預訓練模型
        if self.model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif self.model_type == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        elif self.model_type == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=True)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        else:
            raise ValueError(f"不支援的模型類型: {self.model_type}")
        
        # 計算參數
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型統計:")
        print(f"   總參數: {total_params/1e6:.1f}M")
        print(f"   可訓練參數: {trainable_params/1e6:.1f}M")
        print(f"🚀 預期速度: ViT 的 10+ 倍")
        
        return model.to(self.device)
    
    def get_transforms(self, is_training=True):
        """獲取資料變換"""
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
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
    
    def train_fast(self, train_dataset, val_dataset=None, 
                   batch_size=128, epochs=20, lr=2e-3):
        """
        超快速訓練模型
        
        Args:
            train_dataset: 訓練資料集
            val_dataset: 驗證資料集
            batch_size: batch size (較大，因為模型輕量)
            epochs: 訓練輪數
            lr: 學習率 (較高，因為模型收斂快)
        """
        print(f"\n🚀 開始超快速訓練 {self.model_type}...")
        
        # 準備資料載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,  # 更多 worker，因為模型計算快
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,  # 驗證時可以用更大 batch
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        # 設定優化器 (適合輕量模型的設定)
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=4e-5
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        print(f"📊 超快速訓練設定:")
        print(f"   Batch size: {batch_size}")
        print(f"   學習率: {lr}")
        print(f"   訓練輪數: {epochs}")
        print(f"   優化器: SGD + Cosine Annealing")
        
        # 開始訓練
        total_start_time = time.time()
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 訓練階段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
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
            val_acc = 0.0
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        
                        outputs = self.model(images)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_acc = 100.0 * val_correct / val_total
            
            # 更新學習率
            scheduler.step()
            
            # 計算時間
            epoch_time = time.time() - start_time
            
            # 顯示進度
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"  Train Acc: {train_acc:.2f}%")
            if val_loader:
                print(f"  Val Acc:   {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_{self.model_type}_acc{val_acc:.1f}.pth")
            
            print("-" * 40)
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 超快速訓練完成！")
        print(f"⏱️ 總訓練時間: {total_time/60:.1f} 分鐘")
        if val_loader:
            print(f"🏆 最佳驗證準確率: {best_val_acc:.2f}%")
        
        return best_val_acc
    
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
    print("📱 MobileNet Simpson 角色分類器 - 超快速版")
    print("=" * 60)
    
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
    print("\n🎯 選擇 MobileNet 模型:")
    print("1. mobilenet_v3_small - 超快 (2.5M 參數)")
    print("2. mobilenet_v2 - 平衡 (3.5M 參數) [推薦]")
    print("3. mobilenet_v3_large - 較準確 (5.5M 參數)")
    
    choice = input("請選擇 (1/2/3，預設2): ").strip()
    model_mapping = {
        '1': 'mobilenet_v3_small',
        '2': 'mobilenet_v2',
        '3': 'mobilenet_v3_large'
    }
    model_type = model_mapping.get(choice, 'mobilenet_v2')
    
    # 初始化分類器
    classifier = MobileNetCharacterClassifier(
        num_classes=50,
        model_type=model_type,
        device=device
    )
    
    # 準備資料
    train_dataset, val_dataset = classifier.prepare_data(data_paths)
    
    # 快速訓練設定
    print("\n⚡ 快速訓練設定 (針對 MobileNet 優化):")
    batch_size = 128  # 大 batch size
    epochs = 20       # 較少輪數，因為收斂快
    lr = 2e-3        # 較高學習率
    
    print(f"   Batch size: {batch_size}")
    print(f"   訓練輪數: {epochs}")
    print(f"   學習率: {lr}")
    
    # 開始訓練
    best_acc = classifier.train_fast(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr
    )
    
    print(f"\n🎊 快速訓練完成！最佳準確率: {best_acc:.2f}%")
    print("💡 如果準確率滿意，可以用此模型快速部署")
    print("💡 如果需要更高準確率，考慮使用 EfficientNet 或 ResNet")

if __name__ == "__main__":
    main()