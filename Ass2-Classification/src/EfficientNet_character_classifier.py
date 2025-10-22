#!/usr/bin/env python3
"""
🚀 EfficientNet Simpson 角色分類器 - 高速版本

EfficientNet 優勢：
- 速度比 ViT 快 5-10 倍
- 記憶體需求低
- 在中小型資料集上表現優秀
- 適合快速實驗和部署
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
import timm
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import platform

class EfficientNetCharacterClassifier:
    """
    使用 EfficientNet 的高速角色分類器
    """
    
    def __init__(self, num_classes=50, model_name='efficientnet_b3', device=None):
        """
        初始化分類器
        
        Args:
            num_classes: 類別數量
            model_name: EfficientNet 模型名稱
                🚀 速度優先:
                - efficientnet_b0: 最快 (5.3M參數)
                - efficientnet_b1: 很快 (7.8M參數) 
                - efficientnet_b2: 快 (9.2M參數)
                
                ⚖️ 平衡選擇:
                - efficientnet_b3: 平衡速度與準確率 (12M參數) [當前]
                - efficientnet_b4: 更高準確率 (19M參數)
                
                🎯 抗過擬合 (推薦解決Kaggle問題):
                - efficientnetv2_s: V2小版 (21M參數)
                - efficientnetv2_m: V2中版 (54M參數)
                
                🏆 極致性能:
                - efficientnet_b5: 高準確率 (30M參數)
                - convnext_tiny: ConvNeXt Tiny (28M參數)
                - convnext_base: ConvNeXt Base (89M參數)
            device: 計算設備
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🎯 初始化 {model_name} 分類器")
        print(f"🖥️ 使用設備: {self.device}")
        
        # 初始化模型 (默認標準模式，可在訓練時啟用抗過擬合)
        self.model = self._create_model()
        self.class_to_idx = {}
        self.anti_overfitting_mode = False
        
    def _create_model(self, anti_overfitting=False):
        """
        創建 EfficientNet 模型
        
        Args:
            anti_overfitting: 是否使用抗過擬合設置
        """
        try:
            # 根據是否抗過擬合調整參數
            if anti_overfitting:
                drop_rate = 0.4      # 增加dropout: 0.2 → 0.4
                drop_path_rate = 0.4 # 增加drop_path: 0.2 → 0.4
                print(f"🛡️ 使用抗過擬合模式 (dropout: {drop_rate}, drop_path: {drop_path_rate})")
            else:
                drop_rate = 0.2
                drop_path_rate = 0.2
                print(f"⚡ 使用標準模式 (dropout: {drop_rate}, drop_path: {drop_path_rate})")
            
            # 使用 timm 載入預訓練模型
            try:
                model = timm.create_model(
                    self.model_name,
                    pretrained=True,
                    num_classes=self.num_classes,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate
                )
            except RuntimeError as e:
                if "No pretrained weights exist" in str(e):
                    print(f"⚠️ {self.model_name} 沒有預訓練權重，嘗試替代方案...")
                    
                    # 提供替代方案
                    alternative_models = {
                        'efficientnetv2_l': 'efficientnetv2_m.in21k_ft_in1k',
                        'efficientnetv2_xl': 'efficientnetv2_m.in21k_ft_in1k',
                        'convnext_base.fb_in22k_ft_in1k': 'convnext_base',
                        'convnext_large': 'convnext_base',
                    }
                    
                    if self.model_name in alternative_models:
                        alt_model = alternative_models[self.model_name]
                        print(f"🔄 使用替代模型: {alt_model}")
                        model = timm.create_model(
                            alt_model,
                            pretrained=True,
                            num_classes=self.num_classes,
                            drop_rate=drop_rate,
                            drop_path_rate=drop_path_rate
                        )
                        self.model_name = alt_model  # 更新模型名稱
                    else:
                        raise e
                else:
                    raise e
            
            # 計算模型參數
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"📊 模型統計:")
            print(f"   總參數: {total_params/1e6:.1f}M")
            print(f"   可訓練參數: {trainable_params/1e6:.1f}M")
            
            return model.to(self.device)
            
        except Exception as e:
            print(f"❌ 模型創建失敗: {e}")
            raise
    
    def get_transforms(self, is_training=True):
        """
        獲取資料變換
        
        Args:
            is_training: 是否為訓練模式
        """
        if is_training:
            # 訓練時的資料增強
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 驗證/測試時的變換
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_data(self, data_paths):
        """
        準備資料集 - 使用和 MemoryViT 相同的配置
        
        Args:
            data_paths: 包含 train, val 路徑的字典
        """
        print("\n📊 準備資料...")
        
        # 確認資料路徑
        train_path = data_paths['train']
        val_path = data_paths['val']
        
        print(f"📁 載入訓練資料: {train_path}")
        print(f"📁 載入驗證資料: {val_path}")
        
        # 訓練資料
        train_transform = self.get_transforms(is_training=True)
        train_dataset = datasets.ImageFolder(
            root=train_path,
            transform=train_transform
        )
        
        # 驗證資料
        val_transform = self.get_transforms(is_training=False)
        val_dataset = datasets.ImageFolder(
            root=val_path,
            transform=val_transform
        )
        
        # 建立統一的類別映射（使用訓練集的映射）
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 檢查類別一致性
        if val_dataset.class_to_idx != self.class_to_idx:
            print("⚠️ 警告：訓練集和驗證集的類別映射不完全一致")
            print(f"   訓練集類別數: {len(self.class_to_idx)}")
            print(f"   驗證集類別數: {len(val_dataset.class_to_idx)}")
            
            # 使用訓練集的類別映射重新整理驗證集
            val_dataset.class_to_idx = self.class_to_idx
        
        print(f"✅ 訓練集: {len(train_dataset)} 張圖片")
        print(f"✅ 驗證集: {len(val_dataset)} 張圖片")
        print(f"📝 類別數: {len(self.class_to_idx)}")
        
        # 更新模型的類別數（如果需要）
        if len(self.class_to_idx) != self.num_classes:
            print(f"⚠️ 更新類別數：{self.num_classes} → {len(self.class_to_idx)}")
            self.num_classes = len(self.class_to_idx)
            # 重新創建模型以匹配新的類別數
            self.model = self._create_model()
        
        # 顯示類別資訊
        print(f"📋 前10個類別: {list(self.class_to_idx.keys())[:10]}")
        
        # 測試資料集（從驗證集分出一部分，或者沒有測試集）
        test_dataset = None
        
        return train_dataset, val_dataset, test_dataset
    
    def find_optimal_batch_size(self, train_dataset, start_size=16, max_size=128):
        """
        快速找到最佳 batch size
        
        Args:
            train_dataset: 訓練資料集
            start_size: 起始 batch size
            max_size: 最大 batch size
        """
        print(f"\n⚙️ 尋找最佳 batch size (範圍: {start_size}-{max_size})")
        
        # 測試不同 batch size
        batch_sizes = [32, 64, 128]
        if torch.cuda.is_available():
            batch_sizes = [16, 32, 64, 128]
        
        best_batch_size = 16
        best_speed = 0
        
        for batch_size in batch_sizes:
            try:
                # 測試這個 batch size
                test_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=2,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                
                # 測試速度
                self.model.train()
                start_time = time.time()
                
                for i, (images, labels) in enumerate(test_loader):
                    if i >= 5:  # 只測試 5 個 batch
                        break
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(images)
                
                elapsed = time.time() - start_time
                speed = 5 * batch_size / elapsed  # 每秒處理圖片數
                
                print(f"   Batch {batch_size}: {speed:.1f} 圖片/秒")
                
                if speed > best_speed:
                    best_speed = speed
                    best_batch_size = batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   Batch {batch_size}: ❌ GPU 記憶體不足")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        print(f"✅ 選擇 batch size: {best_batch_size} (速度: {best_speed:.1f} 圖片/秒)")
        return best_batch_size
    
    def train(self, train_dataset, val_dataset=None, 
              batch_size=None, epochs=30, lr=3e-5, 
              auto_batch_size=True, patience=10, resume_from=None):
        """
        訓練模型 (支援斷點續訓)
        
        Args:
            train_dataset: 訓練資料集
            val_dataset: 驗證資料集
            batch_size: batch size (None 表示自動檢測)
            epochs: 訓練輪數
            lr: 學習率
            auto_batch_size: 是否自動檢測 batch size
            patience: 早停耐心值
            resume_from: 斷點續訓檔案路徑
        """
        # 檢查是否從檢查點恢復
        start_epoch = 0
        resume_info = None
        
        if resume_from and os.path.exists(resume_from):
            print(f"🔄 從檢查點恢復訓練: {resume_from}")
            resume_info = self.load_model(resume_from, load_for_training=True)
            start_epoch = resume_info.get('epoch', 0)
            
            print(f"✅ 將從第 {start_epoch + 1} 輪繼續訓練")
        
        print(f"\n🚀 {'繼續' if resume_from else '開始'}訓練 {self.model_name}...")
        
        # 自動檢測 batch size
        if auto_batch_size or batch_size is None:
            batch_size = self.find_optimal_batch_size(train_dataset)
        
        # 準備資料載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
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
        
        # 根據抗過擬合模式調整權重衰減
        weight_decay = 0.02 if self.anti_overfitting_mode else 0.01
        
        # 設定優化器和排程器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        print(f"⚙️ 權重衰減: {weight_decay}")
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr/100
        )
        
        # 恢復優化器和調度器狀態
        if resume_info:
            if resume_info.get('optimizer_state'):
                optimizer.load_state_dict(resume_info['optimizer_state'])
                print("✅ 恢復優化器狀態")
            
            if resume_info.get('scheduler_state'):
                scheduler.load_state_dict(resume_info['scheduler_state'])
                print("✅ 恢復學習率調度器狀態")
        
        # 根據抗過擬合模式調整Label Smoothing
        label_smoothing = 0.15 if self.anti_overfitting_mode else 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"🎯 使用Label Smoothing: {label_smoothing}")
        
        # 訓練歷史 (支援斷點續訓)
        if resume_info and 'history' in resume_info:
            history = resume_info['history']
            best_val_acc = resume_info.get('best_val_acc', 0.0)
            print(f"📊 恢復訓練歷史，已有 {len(history['train_loss'])} 輪記錄")
        else:
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'lr': []
            }
            best_val_acc = 0.0
        
        patience_counter = 0
        
        print(f"📊 訓練設定:")
        print(f"   Batch size: {batch_size}")
        print(f"   學習率: {lr}")
        print(f"   訓練輪數: {epochs}")
        print(f"   早停耐心: {patience}")
        
        # 開始訓練 (支援斷點續訓)
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            
            # 訓練階段
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            
            # 驗證階段
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # 更新學習率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 記錄歷史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # 計算時間
            epoch_time = time.time() - start_time
            
            # 顯示進度
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            if val_loader:
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}")
            
            # 定期保存檢查點 (每5個epoch保存一次)
            if (epoch + 1) % 5 == 0:
                checkpoint_filename = f"{self.model_name}_checkpoint_epoch_{epoch+1:03d}_acc_{val_acc:.2f}.pth"
                self.save_model(
                    checkpoint_filename,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_acc=val_acc,
                    train_acc=train_acc,
                    best_val_acc=best_val_acc,
                    history=history
                )
                print(f"📦 檢查點已保存: {checkpoint_filename}")
            
            # 早停檢查
            if val_loader:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # 保存最佳模型 (新的命名格式: epoch_XX_acc_XX.X)
                    filename = f"{self.model_name}_epoch_{epoch+1:03d}_acc_{val_acc:.2f}.pth"
                    self.save_model(
                        filename,
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        val_acc=val_acc,
                        train_acc=train_acc,
                        best_val_acc=best_val_acc,
                        history=history
                    )
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"\n⏹️ 早停觸發！最佳驗證準確率: {best_val_acc:.2f}%")
                    break
            
            print("-" * 60)
        
        print(f"\n✅ 訓練完成！")
        if val_loader:
            print(f"🏆 最佳驗證準確率: {best_val_acc:.2f}%")
            
            # 顯示最佳模型檔案名
            best_epoch = max(range(len(history['val_acc'])), key=lambda i: history['val_acc'][i]) + start_epoch + 1
            best_filename = f"{self.model_name}_epoch_{best_epoch:03d}_acc_{best_val_acc:.2f}.pth"
            print(f"📁 最佳模型已保存為: {best_filename}")
        
        return history
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新進度條
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{total_loss/len(pbar):.4f}',
                'Acc': f'{acc:.2f}%'
            })
        
        return total_loss / len(train_loader), 100.0 * correct / total
    
    def _validate_epoch(self, val_loader, criterion):
        """驗證一個 epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(val_loader), 100.0 * correct / total
    
    def save_model(self, filename, epoch=None, optimizer=None, scheduler=None, 
                   val_acc=None, train_acc=None, best_val_acc=None, history=None):
        """
        保存模型 (增強版，支援訓練狀態)
        
        Args:
            filename: 保存檔案名
            epoch: 當前訓練輪數
            optimizer: 優化器 (用於斷點續訓)
            scheduler: 學習率調度器
            val_acc: 驗證準確率
            train_acc: 訓練準確率
            best_val_acc: 最佳驗證準確率
            history: 訓練歷史
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
        
        # 添加訓練狀態 (如果提供)
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if val_acc is not None:
            checkpoint['val_acc'] = val_acc
        if train_acc is not None:
            checkpoint['train_acc'] = train_acc
        if best_val_acc is not None:
            checkpoint['best_val_acc'] = best_val_acc
        if history is not None:
            checkpoint['history'] = history
        
        torch.save(checkpoint, filename)
        print(f"💾 模型已保存: {filename}")
    
    def load_model(self, filename, load_for_training=False):
        """
        載入模型
        
        Args:
            filename: 模型檔案路徑
            load_for_training: 是否為訓練載入 (會載入優化器狀態等)
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到模型檔案: {filename}")
        
        print(f"📂 載入模型: {filename}")
        checkpoint = torch.load(filename, map_location=self.device)
        
        # 載入模型權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = checkpoint['idx_to_class']
        
        # 更新類別數
        if len(self.class_to_idx) != self.num_classes:
            self.num_classes = len(self.class_to_idx)
        
        loaded_info = {
            'model_name': checkpoint.get('model_name', 'unknown'),
            'num_classes': len(self.class_to_idx),
            'accuracy': checkpoint.get('val_acc', 'unknown')
        }
        
        if load_for_training:
            # 載入訓練相關資訊
            loaded_info.update({
                'epoch': checkpoint.get('epoch', 0),
                'best_val_acc': checkpoint.get('best_val_acc', 0),
                'train_acc': checkpoint.get('train_acc', 0),
                'optimizer_state': checkpoint.get('optimizer_state_dict'),
                'scheduler_state': checkpoint.get('scheduler_state_dict'),
                'history': checkpoint.get('history', {})
            })
            
            print(f"� 訓練資訊:")
            print(f"   上次訓練到第 {loaded_info['epoch']} 輪")
            print(f"   最佳驗證準確率: {loaded_info['best_val_acc']:.2f}%")
            print(f"   上次訓練準確率: {loaded_info['train_acc']:.2f}%")
        
        print(f"✅ 模型載入成功 (準確率: {loaded_info['accuracy']:.1f}%)")
        return loaded_info
    
    def plot_training_history(self, history):
        """繪製訓練歷史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0,0].plot(history['train_loss'], label='Train Loss')
        if history['val_loss']:
            axes[0,0].plot(history['val_loss'], label='Val Loss')
        axes[0,0].set_title('Loss')
        axes[0,0].legend()
        
        # Accuracy
        axes[0,1].plot(history['train_acc'], label='Train Acc')
        if history['val_acc']:
            axes[0,1].plot(history['val_acc'], label='Val Acc')
        axes[0,1].set_title('Accuracy (%)')
        axes[0,1].legend()
        
        # Learning Rate
        axes[1,0].plot(history['lr'])
        axes[1,0].set_title('Learning Rate')
        
        # Speed comparison
        axes[1,1].text(0.1, 0.8, f'Model: {self.model_name}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.6, f'Classes: {self.num_classes}', transform=axes[1,1].transAxes)
        axes[1,1].text(0.1, 0.4, f'Device: {self.device}', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Model Info')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def get_best_data_path():
    """
    使用和 MemoryViT 相同的資料路徑配置：
    - 訓練資料：augmented/train/
    - 驗證資料：preprocessed/val/
    """
    # 檢測環境
    import platform
    is_wsl = "microsoft" in platform.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")
    
    if is_wsl:
        base_path = "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset"
        augmented_train = f"{base_path}/augmented/train"
        preprocessed_val = f"{base_path}/preprocessed/val"
    else:
        base_path = "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset"
        augmented_train = f"{base_path}/augmented/train"
        preprocessed_val = f"{base_path}/preprocessed/val"
    
    # 檢查必要的資料夾是否存在
    if os.path.exists(augmented_train) and os.path.exists(preprocessed_val):
        return {
            'train': augmented_train,
            'val': preprocessed_val,
            'use_existing_split': True
        }, "🎨 使用增強訓練資料 + 預處理驗證資料"
    else:
        missing = []
        if not os.path.exists(augmented_train):
            missing.append(f"augmented/train: {augmented_train}")
        if not os.path.exists(preprocessed_val):
            missing.append(f"preprocessed/val: {preprocessed_val}")
        raise FileNotFoundError(f"找不到必要的資料夾: {', '.join(missing)}")

def main():
    """主函數"""
    print("🚀 EfficientNet Simpson 角色分類器")
    print("=" * 60)
    
    # 檢查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用設備: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 使用和 MemoryViT 相同的資料路徑
    try:
        data_paths, data_type = get_best_data_path()
        print(f"📂 使用資料: {data_type}")
        print(f"📍 訓練路徑: {data_paths['train']}")
        print(f"📍 驗證路徑: {data_paths['val']}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 請確認以下路徑存在:")
        print("  - Dataset/augmented/train (增強訓練資料)")
        print("  - Dataset/preprocessed/val (預處理驗證資料)")
        return
    
    # 統計資料量
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    total_images = 0
    
    # 統計 train 資料
    for ext in image_extensions:
        total_images += len(glob.glob(os.path.join(data_paths['train'], '**', ext), recursive=True))
    
    # 統計 val 資料
    if data_paths['val'] and os.path.exists(data_paths['val']):
        for ext in image_extensions:
            total_images += len(glob.glob(os.path.join(data_paths['val'], '**', ext), recursive=True))
    
    print(f"📊 總圖片數: {total_images} 張")
    
    if total_images == 0:
        print("❌ 找不到任何圖片檔案！")
        return
    
    # 選擇模型
    print("\n🎯 選擇 EfficientNet 模型:")
    print("🚀 速度優先:")
    print("  1. efficientnet_b0 - 最快 (5.3M參數)")
    print("  2. efficientnet_b1 - 很快 (7.8M參數)")
    print("  3. efficientnet_b2 - 快 (9.2M參數)")
    print("\n⚖️ 平衡選擇:")
    print("  4. efficientnet_b3 - 平衡 (12M參數) [當前使用]")
    print("  5. efficientnet_b4 - 更準確 (19M參數)")
    print("\n🛡️ 抗過擬合 (推薦解決Kaggle問題):")
    print("  6. efficientnetv2_s.in1k - V2小版 (21M參數)")
    print("  7. efficientnetv2_m.in21k_ft_in1k - V2中版 (54M參數) ⭐推薦")
    print("\n🏆 極致性能:")
    print("  8. efficientnet_b5 - 高準確 (30M參數)")
    print("  9. convnext_tiny - ConvNeXt Tiny (28M參數) 🔥推薦")
    print("  10. convnext_base - ConvNeXt Base (89M參數) 🔥最強")
    
    choice = input("請選擇 (1-10，預設4): ").strip()
    model_mapping = {
        '1': 'efficientnet_b0',
        '2': 'efficientnet_b1', 
        '3': 'efficientnet_b2',
        '4': 'efficientnet_b3',
        '5': 'efficientnet_b4',
        '6': 'efficientnetv2_s.in1k',
        '7': 'efficientnetv2_m.in21k_ft_in1k',
        '8': 'efficientnet_b5',
        '9': 'convnext_tiny',
        '10': 'convnext_base'
    }
    model_name = model_mapping.get(choice, 'efficientnet_b3')
    
    # 如果選擇了V2版本或ConvNeXt，自動啟用抗過擬合模式
    auto_anti_overfitting_models = ['efficientnetv2', 'convnext']
    anti_overfitting = any(model_type in model_name for model_type in auto_anti_overfitting_models)
    
    if anti_overfitting:
        if 'efficientnetv2' in model_name:
            print(f"✅ 自動啟用抗過擬合模式 (EfficientNet V2架構)")
        elif 'convnext' in model_name:
            print(f"✅ 自動啟用抗過擬合模式 (ConvNeXt架構)")
    else:
        # 詢問是否啟用抗過擬合模式
        overfitting_choice = input("\n🛡️ 是否啟用抗過擬合模式？(適合解決Kaggle測試集準確率低的問題) (y/N): ").strip().lower()
        anti_overfitting = overfitting_choice in ['y', 'yes']
    
    # 初始化分類器
    classifier = EfficientNetCharacterClassifier(
        num_classes=50,  # 預設50類，實際會根據資料調整
        model_name=model_name,
        device=device
    )
    
    # 如果啟用抗過擬合模式，重新創建模型
    if anti_overfitting:
        classifier.anti_overfitting_mode = True
        classifier.model = classifier._create_model(anti_overfitting=True)
        print("🛡️ 抗過擬合模式已啟用")
    
    # 準備資料 - 使用和 MemoryViT 相同的資料
    train_dataset, val_dataset, test_dataset = classifier.prepare_data(data_paths)
    
    # 檢查是否要從檢查點恢復
    print("\n🔄 訓練模式選擇:")
    print("1. 從頭開始訓練 (預設)")
    print("2. 從檢查點繼續訓練")
    
    train_mode = input("請選擇 (1/2): ").strip()
    resume_from = None
    
    if train_mode == "2":
        # 尋找可用的模型檔案 (新舊格式都支援)
        model_files = []
        
        # 搜尋新格式檔案 (efficientnet_b3_epoch_XXX_acc_XX.XX.pth)
        for f in os.listdir('.'):
            if (f.startswith('efficientnet_') and '_epoch_' in f and '_acc_' in f and f.endswith('.pth')) or \
               (f.startswith('best_efficientnet_') and f.endswith('.pth')):
                model_files.append(f)
        
        # 按檔案修改時間排序 (最新的在前面)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if model_files:
            print("\n📁 找到以下模型檔案:")
            for i, file in enumerate(model_files, 1):
                # 解析檔案資訊
                file_info = ""
                try:
                    if '_epoch_' in file and '_acc_' in file:
                        # 新格式: efficientnet_b3_epoch_015_acc_98.50.pth
                        parts = file.replace('.pth', '').split('_')
                        epoch_idx = parts.index('epoch') + 1
                        acc_idx = parts.index('acc') + 1
                        epoch_num = parts[epoch_idx]
                        accuracy = parts[acc_idx]
                        file_info = f" (第{epoch_num}輪, 準確率:{accuracy}%)"
                    elif 'best_' in file and 'acc' in file:
                        # 舊格式: best_efficientnet_b3_acc98.0.pth
                        acc_part = file.split('acc')[1].replace('.pth', '')
                        file_info = f" (準確率:{acc_part}%)"
                except:
                    pass
                
                # 顯示檔案修改時間
                import time
                mod_time = os.path.getmtime(file)
                time_str = time.strftime('%m/%d %H:%M', time.localtime(mod_time))
                
                print(f"   {i}. {file}{file_info} [{time_str}]")
            
            try:
                choice = int(input(f"請選擇模型檔案 (1-{len(model_files)}): ")) - 1
                resume_from = model_files[choice]
                print(f"✅ 將從 {resume_from} 繼續訓練")
            except (ValueError, IndexError):
                print("❌ 選擇無效，將從頭開始訓練")
        else:
            print("❌ 找不到可用的模型檔案，將從頭開始訓練")
    
    # 訓練參數
    epochs = int(input("總訓練輪數 (預設 30): ") or "30")
    lr = float(input("學習率 (預設 3e-5): ") or "3e-5")
    
    # 開始訓練 (支援斷點續訓)
    history = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        lr=lr,
        auto_batch_size=True,
        patience=10,
        resume_from=resume_from
    )
    
    # 繪製結果
    classifier.plot_training_history(history)
    
    print("\n🎉 訓練完成！")

if __name__ == "__main__":
    main()