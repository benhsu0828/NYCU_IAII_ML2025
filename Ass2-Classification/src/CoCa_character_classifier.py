#!/usr/bin/env python3
"""
🔮 CoCa (Contrastive Captioners) 特徵提取 + 自定義分類頭

CoCa 是一個強大的多模態模型，結合了：
- 對比學習 (Contrastive Learning)
- 圖像描述生成 (Image Captioning)
- 優秀的視覺特徵提取能力

這個分類器使用 CoCa 作為特徵提取器，然後訓練自定義的分類頭。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from datetime import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 安裝和導入 open_clip (包含 CoCa)
try:
    import open_clip
    print("✅ open_clip 已安裝")
except ImportError:
    print("❌ 需要安裝 open_clip:")
    print("pip install open-clip-torch")
    raise ImportError("請先安裝 open_clip: pip install open-clip-torch")

class CoCaCharacterClassifier:
    """
    基於 CoCa 的辛普森角色分類器
    
    特點：
    - 使用 CoCa 作為強大的視覺特徵提取器
    - 凍結 CoCa 權重，只訓練分類頭
    - 支援多種 CoCa 模型版本
    - 高效的遷移學習
    """
    
    def __init__(self, num_classes=50, coca_model='coca_ViT-B-32', device=None):
        """
        初始化 CoCa 分類器
        
        Args:
            num_classes: 分類類別數
            coca_model: CoCa 模型版本 ('coca_ViT-B-32', 'coca_ViT-L-14')
            device: 計算設備
        """
        self.num_classes = num_classes
        self.coca_model_name = coca_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔮 CoCa 辛普森角色分類器")
        print(f"🎯 模型: {coca_model}")
        print(f"📊 類別數: {num_classes}")
        print(f"🖥️ 設備: {self.device}")
        
        # 載入 CoCa 模型
        self.coca_model, self.preprocess = self._load_coca_model()
        
        # 創建分類頭
        self.classifier_head = self._create_classification_head()
        
        # 完整模型
        self.model = CoCaClassifier(self.coca_model, self.classifier_head)
        self.model.to(self.device)
        
        # 類別映射
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        print("✅ CoCa 分類器初始化完成")
    
    def _load_coca_model(self):
        """載入預訓練的 CoCa 模型"""
        print(f"🔄 載入 CoCa 模型: {self.coca_model_name}")
        
        try:
            # 載入 CoCa 模型和預處理
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.coca_model_name, 
                pretrained='laion2b_s13b_b90k'  # 使用 LAION-2B 預訓練權重
            )
            
            print("✅ CoCa 模型載入成功")
            
            # 凍結 CoCa 參數
            for param in model.parameters():
                param.requires_grad = False
            
            print("❄️ CoCa 特徵提取器已凍結")
            
            # 獲取特徵維度
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = model.encode_image(dummy_input)
                self.feature_dim = features.shape[-1]
            
            print(f"📐 特徵維度: {self.feature_dim}")
            
            return model, preprocess
            
        except Exception as e:
            print(f"❌ 載入 CoCa 模型失敗: {e}")
            print("💡 嘗試使用較小的模型...")
            
            # 備選：使用較小的 CLIP 模型
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', 
                    pretrained='openai'
                )
                print("✅ 改用 CLIP ViT-B-32 模型")
                
                # 凍結參數
                for param in model.parameters():
                    param.requires_grad = False
                
                # 獲取特徵維度
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    features = model.encode_image(dummy_input)
                    self.feature_dim = features.shape[-1]
                
                print(f"📐 特徵維度: {self.feature_dim}")
                
                return model, preprocess
                
            except Exception as e2:
                raise RuntimeError(f"無法載入任何視覺模型: {e2}")
    
    def _create_classification_head(self):
        """創建分類頭"""
        print(f"🏗️ 創建分類頭: {self.feature_dim} → {self.num_classes}")
        
        # 多層分類頭，提高表達能力
        classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Dropout(0.2),
            
            nn.Linear(512, self.num_classes)
        )
        
        # 統計分類頭參數
        classifier_params = sum(p.numel() for p in classifier.parameters())
        print(f"🎯 分類頭參數: {classifier_params:,} ({classifier_params/1e6:.2f}M)")
        
        return classifier
    
    def get_transforms(self, is_training=True):
        """
        獲取資料預處理變換
        使用 CoCa 的標準預處理，並添加訓練時的增強
        """
        if is_training:
            # 訓練時添加一些增強，但保持與 CoCa 預處理的相容性
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # CoCa 標準輸入尺寸
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP/CoCa 標準
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            # 驗證時使用標準預處理
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        
        return transform
    
    def prepare_data(self, data_paths):
        """
        準備資料集 - 直接讀取現有資料，不進行 data aggregation
        
        Args:
            data_paths: 包含 train, val 路徑的字典
        """
        print("\n📊 準備資料...")
        
        # 確認資料路徑
        train_path = data_paths['train']
        val_path = data_paths['val']
        
        print(f"📁 載入訓練資料: {train_path}")
        print(f"📁 載入驗證資料: {val_path}")
        
        # 檢查路徑是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"訓練資料路徑不存在: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"驗證資料路徑不存在: {val_path}")
        
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
        
        # 建立統一的類別映射
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 顯示資料集統計
        print(f"\n📈 資料集統計:")
        print(f"   訓練樣本: {len(train_dataset):,}")
        print(f"   驗證樣本: {len(val_dataset):,}")
        print(f"   類別數量: {len(self.class_to_idx)}")
        
        # 顯示部分類別
        class_names = list(self.class_to_idx.keys())
        if len(class_names) <= 10:
            print(f"   類別列表: {', '.join(class_names)}")
        else:
            print(f"   類別範例: {', '.join(class_names[:5])} ... (+{len(class_names)-5} 更多)")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset=None, 
              batch_size=32, epochs=50, lr=1e-3, 
              patience=10, save_dir='models'):
        """
        訓練分類器
        
        Args:
            train_dataset: 訓練資料集
            val_dataset: 驗證資料集
            batch_size: 批次大小
            epochs: 訓練輪數
            lr: 學習率
            patience: 早停耐心值
            save_dir: 模型保存目錄
        """
        print(f"\n🚀 開始訓練 CoCa 分類器")
        print(f"📊 訓練參數:")
        print(f"   批次大小: {batch_size}")
        print(f"   學習率: {lr}")
        print(f"   最大輪數: {epochs}")
        print(f"   早停耐心: {patience}")
        
        # 創建資料載入器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # 設定優化器和損失函數
        optimizer = optim.AdamW(
            self.model.classifier_head.parameters(),  # 只訓練分類頭
            lr=lr,
            weight_decay=0.01
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 訓練記錄
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        print(f"\n📈 開始訓練...")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [訓練]")
            
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向傳播
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # 反向傳播
                loss.backward()
                optimizer.step()
                
                # 統計
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 更新進度條
                current_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
            
            # 計算訓練指標
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            train_history['train_loss'].append(avg_train_loss)
            train_history['train_acc'].append(train_acc)
            
            # 驗證階段
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                train_history['val_loss'].append(val_loss)
                train_history['val_acc'].append(val_acc)
            
            # 更新學習率
            scheduler.step()
            
            # 計算時間
            epoch_time = time.time() - epoch_start
            
            # 打印結果
            print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
            print(f"  訓練 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
            if val_loader:
                print(f"  驗證 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 早停和模型保存
            current_val_acc = val_acc if val_loader else train_acc
            
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                patience_counter = 0
                
                # 保存最佳模型
                self._save_model(save_dir, epoch, current_val_acc, train_history)
                print(f"  🎯 新的最佳模型! 驗證準確率: {best_val_acc:.2f}%")
                
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\n⏹️ 早停! {patience} 輪無改善")
                break
        
        # 訓練完成
        total_time = time.time() - start_time
        print(f"\n🎉 訓練完成!")
        print(f"⏱️ 總時間: {total_time:.1f}s")
        print(f"🎯 最佳驗證準確率: {best_val_acc:.2f}%")
        
        # 繪製訓練曲線
        self._plot_training_history(train_history, save_dir)
        
        return train_history
    
    def _validate(self, val_loader, criterion):
        """驗證模型"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        return avg_val_loss, val_acc
    
    def _save_model(self, save_dir, epoch, accuracy, history):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"coca_classifier_epoch_{epoch+1:03d}_acc_{accuracy:.2f}_{timestamp}.pth"
        model_path = os.path.join(save_dir, model_name)
        
        # 保存完整的模型狀態
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': None,  # 可選
            'accuracy': accuracy,
            'num_classes': self.num_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'coca_model_name': self.coca_model_name,
            'feature_dim': self.feature_dim,
            'history': history
        }, model_path)
        
        print(f"💾 模型已保存: {model_path}")
        
        # 也保存一個 "latest" 版本
        latest_path = os.path.join(save_dir, "coca_classifier_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': None,
            'accuracy': accuracy,
            'num_classes': self.num_classes,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'coca_model_name': self.coca_model_name,
            'feature_dim': self.feature_dim,
            'history': history
        }, latest_path)
    
    def _plot_training_history(self, history, save_dir):
        """繪製訓練歷史"""
        plt.figure(figsize=(12, 4))
        
        # Loss 曲線
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss', marker='o')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy 曲線
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy', marker='o')
        if history['val_acc']:
            plt.plot(history['val_acc'], label='Validation Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存圖片
        plot_path = os.path.join(save_dir, 'coca_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 訓練曲線已保存: {plot_path}")

class CoCaClassifier(nn.Module):
    """
    CoCa 分類器模型
    組合 CoCa 特徵提取器和自定義分類頭
    """
    
    def __init__(self, coca_model, classifier_head):
        super(CoCaClassifier, self).__init__()
        self.coca_model = coca_model
        self.classifier_head = classifier_head
        
    def forward(self, x):
        # 使用 CoCa 提取特徵
        with torch.no_grad():  # 凍結特徵提取器
            features = self.coca_model.encode_image(x)
            
        # 通過分類頭
        output = self.classifier_head(features)
        return output

def main():
    """主函數"""
    print("🔮 CoCa 辛普森角色分類器")
    print("=" * 50)
    
    # 設定參數
    NUM_CLASSES = 50  # 辛普森角色數量
    BATCH_SIZE = 16   # CoCa 模型較大，使用較小的批次
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    
    # 資料路徑 (根據您的實際路徑調整)
    data_paths = {
        'train': '/Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification/Dataset/train',
        'val': '/Users/nimab/Desktop/陽交大/NYCU_IAII_ML2025/Ass2-Classification/Dataset/val'
    }
    
    try:
        # 初始化分類器
        print("🚀 初始化 CoCa 分類器...")
        classifier = CoCaCharacterClassifier(
            num_classes=NUM_CLASSES,
            coca_model='coca_ViT-B-32'  # 或嘗試 'coca_ViT-L-14'
        )
        
        # 準備資料
        print("📊 準備資料...")
        train_dataset, val_dataset = classifier.prepare_data(data_paths)
        
        # 開始訓練
        print("🎯 開始訓練...")
        history = classifier.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            patience=10,
            save_dir='models'
        )
        
        print("\n🎉 CoCa 分類器訓練完成!")
        
    except Exception as e:
        print(f"❌ 訓練過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
