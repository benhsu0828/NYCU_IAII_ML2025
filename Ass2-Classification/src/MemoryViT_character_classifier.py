import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import platform
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from vit_pytorch.learnable_memory_vit import ViT, Adapter

def get_best_data_path():
    """
    使用固定的資料路徑配置：
    - 訓練資料：augmented/train/
    - 驗證資料：preprocessed/val/
    """
    # 檢測環境
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

class CharacterDataset(Dataset):
    """50類角色分類資料集"""
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

class MemoryViTCharacterClassifier:
    def __init__(self, num_classes=50, image_size=224, device='cuda'):
        self.device = device
        self.num_classes = num_classes
        self.image_size = image_size
        
        # 資料增強策略
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 創建基礎 ViT 模型
        self.base_vit = ViT(
            image_size=image_size,
            patch_size=16,
            num_classes=1000,  # 預訓練類別數（後續會被 Adapter 覆蓋）
            dim=768,           # 標準 ViT-Base 維度
            depth=12,          # 12 層 Transformer
            heads=12,          # 12 個注意力頭
            mlp_dim=3072,      # MLP 維度
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)
        
        # 創建角色分類 Adapter
        self.character_adapter = None  # 將在準備資料後創建
        
    def prepare_data(self, data_paths, test_size=0.15, val_size=0.15):
        """準備 50 類角色分類資料"""
        print("📂 準備角色分類資料...")
        
        train_path = data_paths['train']
        val_path = data_paths['val']
        use_existing_split = data_paths['use_existing_split']
        
        if use_existing_split and val_path:
            print("✅ 使用已有的 train/val 分割")
            return self._prepare_data_with_split(train_path, val_path, test_size)
        else:
            print("🔄 從 train 資料夾重新分割")
            return self._prepare_data_from_single_folder(train_path, test_size, val_size)
    
    def _prepare_data_with_split(self, train_path, val_path, test_size=0.15):
        """使用已有的 train/val 分割"""
        # 收集訓練資料
        print(f"📁 載入訓練資料: {train_path}")
        train_image_paths = self._collect_images(train_path)
        
        # 收集驗證資料
        print(f"📁 載入驗證資料: {val_path}")
        val_image_paths = self._collect_images(val_path)
        
        # 建立統一的類別映射
        all_image_paths = train_image_paths + val_image_paths
        all_classes = sorted(list(set([
            os.path.basename(os.path.dirname(path)) 
            for path in all_image_paths
        ])))
        
        if len(all_classes) != self.num_classes:
            print(f"⚠️ 警告：發現 {len(all_classes)} 個類別，但期望 {self.num_classes} 個")
            self.num_classes = len(all_classes)
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 創建標籤
        train_labels = [self.class_to_idx[os.path.basename(os.path.dirname(path))] 
                       for path in train_image_paths]
        val_labels = [self.class_to_idx[os.path.basename(os.path.dirname(path))] 
                     for path in val_image_paths]
        
        # 從驗證集中分出測試集
        if test_size > 0:
            val_paths_split, test_paths, val_labels_split, test_labels = train_test_split(
                val_image_paths, val_labels,
                test_size=test_size,
                stratify=val_labels,
                random_state=42
            )
        else:
            val_paths_split = val_image_paths
            val_labels_split = val_labels
            test_paths = []
            test_labels = []
        
        print(f"✅ 發現 {len(all_classes)} 個角色類別")
        print(f"✅ 訓練資料: {len(train_image_paths)} 張")
        print(f"✅ 驗證資料: {len(val_paths_split)} 張")
        print(f"✅ 測試資料: {len(test_paths)} 張")
        
        # 檢查每類別的圖片數量
        self._print_class_distribution(train_labels + val_labels_split, "Train + Val")
        
        # 創建資料集
        self.train_dataset = CharacterDataset(
            train_image_paths, train_labels, self.class_to_idx, self.train_transform
        )
        self.val_dataset = CharacterDataset(
            val_paths_split, val_labels_split, self.class_to_idx, self.val_transform
        )
        
        if test_paths:
            self.test_dataset = CharacterDataset(
                test_paths, test_labels, self.class_to_idx, self.val_transform
            )
        else:
            # 如果沒有測試集，使用驗證集的一部分作為測試集
            self.test_dataset = self.val_dataset
        
        self._create_adapter_and_save_mapping()
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _prepare_data_from_single_folder(self, data_path, test_size, val_size):
        """從單一資料夾重新分割資料"""
        # 收集所有圖片
        all_image_paths = self._collect_images(data_path)
        
        # 建立類別映射
        all_classes = sorted(list(set([
            os.path.basename(os.path.dirname(path)) 
            for path in all_image_paths
        ])))
        
        if len(all_classes) != self.num_classes:
            print(f"⚠️ 警告：發現 {len(all_classes)} 個類別，但期望 {self.num_classes} 個")
            self.num_classes = len(all_classes)
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 創建標籤
        labels = [self.class_to_idx[os.path.basename(os.path.dirname(path))] 
                 for path in all_image_paths]
        
        print(f"✅ 發現 {len(all_classes)} 個角色類別")
        print(f"✅ 總共 {len(all_image_paths)} 張圖片")
        
        # 檢查每類別的圖片數量
        self._print_class_distribution(labels, "All Data")
        
        # 分割資料集
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_image_paths, labels, 
            test_size=(test_size + val_size), 
            stratify=labels, 
            random_state=42
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(test_size / (test_size + val_size)),
            stratify=temp_labels,
            random_state=42
        )
        
        # 創建資料集
        self.train_dataset = CharacterDataset(
            train_paths, train_labels, self.class_to_idx, self.train_transform
        )
        self.val_dataset = CharacterDataset(
            val_paths, val_labels, self.class_to_idx, self.val_transform
        )
        self.test_dataset = CharacterDataset(
            test_paths, test_labels, self.class_to_idx, self.val_transform
        )
        
        print(f"📊 資料分割:")
        print(f"  訓練集: {len(self.train_dataset)} 張")
        print(f"  驗證集: {len(self.val_dataset)} 張")
        print(f"  測試集: {len(self.test_dataset)} 張")
        
        self._create_adapter_and_save_mapping()
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _collect_images(self, data_path):
        """收集指定路徑下的所有圖片"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        all_image_paths = []
        
        for ext in image_extensions:
            all_image_paths.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
        
        return all_image_paths
    
    def _print_class_distribution(self, labels, data_name):
        """打印類別分布"""
        class_counts = {}
        for label in labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"📊 {data_name} - 每類別圖片數量:")
        for cls_name, count in sorted(class_counts.items()):
            print(f"  {cls_name}: {count} 張")
    
    def _create_adapter_and_save_mapping(self):
        """創建 Adapter 並保存類別映射"""
        # 創建角色分類 Adapter
        self.character_adapter = Adapter(
            vit=self.base_vit,
            num_classes=self.num_classes,
            num_memories_per_layer=20  # 針對 50 類增加記憶數量
        ).to(self.device)
        
        print(f"✅ 角色分類 Adapter 創建完成 ({self.num_classes} 類)")
        
        # 保存類別映射
        class_mapping = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': self.num_classes
        }
        
        with open('character_class_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)
    
    def find_optimal_batch_size(self, max_batch_size=128, start_batch_size=16):
        """
        智慧找到最佳 batch size
        測試不同的 batch size 直到記憶體用盡，找到最佳配置
        """
        print("🔍 正在尋找最佳 batch size...")
        print(f"   起始大小: {start_batch_size}, 最大測試: {max_batch_size}")
        
        if not torch.cuda.is_available():
            print("   ⚠️ 未檢測到 GPU，建議使用 batch_size=4")
            return 4
        
        # 顯示 GPU 資訊
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   🖥️ GPU: {gpu_name}")
        print(f"   📊 總記憶體: {gpu_memory:.1f} GB")
        
        optimal_batch_size = start_batch_size
        best_throughput = 0
        
        # 確保模型已經創建
        if self.character_adapter is None:
            print("   ❌ 請先準備資料以創建模型")
            return start_batch_size
        
        # 測試不同的 batch size
        test_sizes = [16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
        test_sizes = [size for size in test_sizes if size <= max_batch_size]
        
        for batch_size in test_sizes:
            try:
                print(f"   📝 測試 batch size: {batch_size}")
                
                # 清空 GPU 快取
                torch.cuda.empty_cache()
                
                # 創建測試批次
                test_batch = torch.randn(
                    batch_size, 3, self.image_size, self.image_size, 
                    device=self.device, dtype=torch.float32
                )
                
                # 測試前向傳播
                self.character_adapter.eval()
                with torch.no_grad():
                    # 預熱
                    for _ in range(3):
                        _ = self.character_adapter(test_batch[:min(4, batch_size)])
                    
                    # 正式測試
                    torch.cuda.synchronize()
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    outputs = self.character_adapter(test_batch)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)  # 毫秒
                
                # 檢查記憶體使用
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                memory_percent = (memory_used / gpu_memory) * 100
                
                # 計算效能指標
                throughput = (batch_size / elapsed_time) * 1000  # images/second
                time_per_image = elapsed_time / batch_size  # ms per image
                
                print(f"      ✅ 成功 - 記憶體: {memory_percent:.1f}% ({memory_used:.1f}GB)")
                print(f"      ⏱️ 處理時間: {elapsed_time:.1f}ms")
                print(f"      🚀 吞吐量: {throughput:.1f} images/sec")
                print(f"      📊 每張圖片: {time_per_image:.2f}ms")
                
                # 計算效率分數 (綜合考慮吞吐量和記憶體使用)
                memory_efficiency = min(memory_percent / 80.0, 1.0)  # 80%以下效率較高
                throughput_efficiency = throughput / (batch_size * 50)  # 歸一化吞吐量
                
                # 綜合效率分數 (記憶體使用率越低越好，吞吐量越高越好)
                efficiency_score = throughput_efficiency * (2.0 - memory_efficiency)
                
                print(f"      📈 效率分數: {efficiency_score:.3f}")
                
                # 如果記憶體使用超過 85%，停止增加（保留一些安全邊界）
                if memory_percent > 85:
                    print(f"      ⚠️ 記憶體使用過高 ({memory_percent:.1f}%)，停止增加")
                    break
                
                # 更新最佳配置 - 改用效率分數而非單純吞吐量
                if efficiency_score > best_throughput:
                    best_throughput = efficiency_score
                    optimal_batch_size = batch_size
                elif batch_size > 32 and time_per_image > 15.0:  # 如果每張圖片超過15ms且batch>32，停止
                    print(f"      ⚠️ 單張圖片處理時間過長 ({time_per_image:.2f}ms)，效率開始下降")
                    break
                
                # 清理測試資料
                del test_batch, outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"      ❌ 記憶體不足 (OOM)")
                    break
                else:
                    print(f"      ❌ 其他錯誤: {e}")
                    break
        
        # 清空快取
        torch.cuda.empty_cache()
        
        print(f"\n🎯 檢測結果:")
        print(f"   最佳 batch size: {optimal_batch_size}")
        print(f"   最佳效率分數: {best_throughput:.3f}")
        
        # 分析為什麼選擇這個 batch size
        print(f"\n📊 分析:")
        if optimal_batch_size == 16:
            print("   🔍 選擇 batch_size=16 的可能原因:")
            print("     • ViT 注意力機制複雜度高，較小 batch 更高效")
            print("     • GPU 記憶體頻寬限制，大 batch 沒有帶來速度提升")
            print("     • 模型計算密集，受計算複雜度影響大於記憶體量")
        elif optimal_batch_size <= 32:
            print("   ⚡ 適中的 batch size，平衡了記憶體使用和計算效率")
        else:
            print("   💪 較大的 batch size，您的 GPU 性能強勁")
        
        # 提供不同場景的建議
        print(f"\n💡 不同場景建議:")
        print(f"     🚀 最大吞吐量: batch_size={optimal_batch_size}")
        print(f"     🎯 穩定訓練: batch_size={max(16, optimal_batch_size // 2)}")
        print(f"     🛡️ 保守安全: batch_size=16")
        
        # 額外建議
        if optimal_batch_size >= 64:
            print("\n   💡 您的 GPU 記憶體充足，可以考慮:")
            print("      • 使用更大的圖像尺寸 (256x256)")
            print("      • 嘗試更複雜的資料增強")
        elif optimal_batch_size >= 32:
            print("\n   👍 GPU 記憶體適中，當前設定很好")
        else:
            print("\n   ⚠️ 為了更好的效能，可以考慮:")
            print("      • 降低圖像尺寸 (192x192)")
            print("      • 使用梯度累積技術")
            print("      • 混合精度訓練 (FP16)")
        
        return optimal_batch_size
    
    def train(self, batch_size=None, epochs=50, lr=1e-4, warmup_epochs=10, auto_batch_size=True, use_mixed_precision=True):
        """訓練 MemoryViT 角色分類模型"""
        print("🚀 開始訓練 MemoryViT 角色分類模型...")
        
        # 檢查混合精度支援
        if use_mixed_precision and torch.cuda.is_available():
            print("⚡ 啟用混合精度訓練 (FP16) - 預期提升 30-50% 速度")
            scaler = torch.cuda.amp.GradScaler()
        else:
            print("📝 使用標準精度訓練 (FP32)")
            scaler = None
            use_mixed_precision = False
        
        # 自動檢測最佳 batch size
        if batch_size is None and auto_batch_size:
            print("\n🔍 啟用自動 batch size 檢測...")
            batch_size = self.find_optimal_batch_size()
            print(f"✅ 自動選擇 batch size: {batch_size}")
        elif batch_size is None:
            batch_size = 16  # 預設值
            print(f"📝 使用預設 batch size: {batch_size}")
        else:
            print(f"📝 使用指定 batch size: {batch_size}")
        
        # 資料載入器
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=6,
            pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=6,
            pin_memory=True
        )
        
        # 損失函數和優化器
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 標籤平滑
        
        # 只訓練 Adapter 參數，凍結基礎 ViT
        adapter_params = [p for p in self.character_adapter.parameters() if p.requires_grad]
        optimizer = optim.AdamW(adapter_params, lr=lr, weight_decay=0.05)
        
        # 學習率調度器
        total_steps = len(train_loader) * epochs
        warmup_steps = len(train_loader) * warmup_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # 訓練記錄
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0
        
        print(f"📊 訓練配置:")
        print(f"  總參數: {sum(p.numel() for p in self.character_adapter.parameters()):,}")
        print(f"  可訓練參數: {sum(p.numel() for p in adapter_params):,}")
        print(f"  批次大小: {batch_size}")
        print(f"  學習率: {lr}")
        print(f"  總輪數: {epochs}")
        
        for epoch in range(epochs):
            # 訓練階段
            self.character_adapter.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 混合精度前向傳播
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.character_adapter(images)
                        loss = criterion(outputs, labels)
                    
                    # 混合精度反向傳播
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 標準精度訓練
                    outputs = self.character_adapter(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                
                # 統計
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 更新進度條
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'LR': f'{current_lr:.2e}'
                })
            
            # 驗證階段
            self.character_adapter.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # 混合精度驗證
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.character_adapter(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = self.character_adapter(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 計算平均值
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # 記錄歷史
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(current_lr)
            
            # 輸出結果
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {current_lr:.2e}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # 增強的模型存檔
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.character_adapter.state_dict(),
                    'base_vit_state_dict': self.base_vit.state_dict(),  # 也保存基礎 ViT
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss,
                    'best_val_acc': best_val_acc,
                    'learning_rate': current_lr,
                    'class_mapping': {
                        'class_to_idx': self.class_to_idx,
                        'idx_to_class': self.idx_to_class,
                        'num_classes': self.num_classes
                    },
                    'training_config': {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'lr': lr,
                        'warmup_epochs': warmup_epochs,
                        'use_mixed_precision': use_mixed_precision,
                        'image_size': self.image_size
                    },
                    'history': history.copy()
                }
                
                torch.save(checkpoint, 'best_memory_vit_character_classifier.pth')
                print(f'  🎯 新的最佳驗證準確率: {best_val_acc:.2f}% (已保存模型)')
                
                # 同時保存一個輕量版本 (只有模型權重)
                torch.save({
                    'model_state_dict': self.character_adapter.state_dict(),
                    'class_mapping': {
                        'class_to_idx': self.class_to_idx,
                        'idx_to_class': self.idx_to_class,
                        'num_classes': self.num_classes
                    },
                    'val_acc': val_acc,
                    'training_config': {
                        'image_size': self.image_size,
                        'num_classes': self.num_classes
                    }
                }, f'memoryvit_model_acc{val_acc:.1f}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'  ⏰ Early stopping triggered after {patience} epochs without improvement')
                    break
            
            print('-' * 60)
        
        print(f"✅ 訓練完成！最佳驗證準確率: {best_val_acc:.2f}%")
        return history
    
    def evaluate(self, batch_size=32):
        """評估模型"""
        print("📊 評估模型性能...")
        
        # 載入最佳模型
        checkpoint = torch.load('best_memory_vit_character_classifier.pth')
        self.character_adapter.load_state_dict(checkpoint['model_state_dict'])
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        self.character_adapter.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='評估中'):
                images = images.to(self.device)
                outputs = self.character_adapter(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 計算準確率
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"🎯 測試集準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 分類報告
        class_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        print("\n📈 詳細分類報告:")
        print(classification_report(all_labels, all_predictions, target_names=class_names))
        
        # 保存結果
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return all_predictions, all_labels, accuracy
    
    def plot_training_history(self, history):
        """繪製訓練歷史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失
        axes[0, 0].plot(history['train_loss'], label='訓練損失', color='blue')
        axes[0, 0].plot(history['val_loss'], label='驗證損失', color='red')
        axes[0, 0].set_title('模型損失')
        axes[0, 0].set_ylabel('損失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 準確率
        axes[0, 1].plot(history['train_acc'], label='訓練準確率', color='blue')
        axes[0, 1].plot(history['val_acc'], label='驗證準確率', color='red')
        axes[0, 1].set_title('模型準確率')
        axes[0, 1].set_ylabel('準確率 (%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 學習率
        axes[1, 0].plot(history['learning_rate'], color='green')
        axes[1, 0].set_title('學習率變化')
        axes[1, 0].set_ylabel('學習率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 驗證準確率放大
        axes[1, 1].plot(history['val_acc'], color='red', linewidth=2)
        axes[1, 1].set_title('驗證準確率詳細')
        axes[1, 1].set_ylabel('驗證準確率 (%)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path, top_k=5):
        """預測單張圖片"""
        self.character_adapter.eval()
        
        # 載入並預處理圖片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.character_adapter(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # 轉換結果
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            class_name = self.idx_to_class[class_idx]
            prob = top_probs[0][i].item()
            results.append((class_name, prob))
        
        return results

def main():
    """主函數"""
    print("🎭 MemoryViT 50類角色分類器")
    print("=" * 50)
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用裝置: {device}")
    
    # 智慧選擇資料路徑
    data_paths, data_type = get_best_data_path()
    

    if data_paths is None:
        print("❌ 找不到訓練資料！")
        print("請確認以下路徑存在:")
        print("  - augmented/train (增強訓練資料)")
        print("  - preprocessed/val (預處理驗證資料)")
        return
    
    print(f"📂 使用資料: {data_type}")
    print(f"📍 訓練路徑: {data_paths['train']}")
    print(f"📍 驗證路徑: {data_paths['val']}")
    print("✅ 使用您已分割好的 train/val 資料")
    
    # 統計資料量
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    total_images = 0
    
    # 統計 train 資料
    for ext in image_extensions:
        total_images += len(glob.glob(os.path.join(data_paths['train'], '**', ext), recursive=True))
    
    # 如果有 val 資料，也統計進去
    if data_paths['val'] and os.path.exists(data_paths['val']):
        for ext in image_extensions:
            total_images += len(glob.glob(os.path.join(data_paths['val'], '**', ext), recursive=True))
    
    print(f"📊 總圖片數: {total_images} 張")
    
    if total_images == 0:
        print("❌ 找不到任何圖片檔案！")
        return
    
    # 初始化分類器
    classifier = MemoryViTCharacterClassifier(
        num_classes=50,  # 你的 50 個角色類別
        device=device
    )
    
    try:
        # 準備資料
        train_dataset, val_dataset, test_dataset = classifier.prepare_data(data_paths)
        
        # 詢問是否要自動檢測最佳 batch size
        print("\n⚙️ Batch Size 設定:")
        print("1. 自動檢測最佳 batch size (推薦)")
        print("2. 手動指定 batch size")
        
        choice = input("請選擇 (1/2，預設1): ").strip()
        
        if choice == "2":
            batch_size = int(input("請輸入 batch size (建議 16-64): ") or "32")
            auto_batch_size = False
            print(f"✅ 使用手動指定 batch size: {batch_size}")
        else:
            batch_size = None  # 將由自動檢測決定
            auto_batch_size = True
            print("✅ 將自動檢測最佳 batch size")
        
        # 其他訓練參數
        epochs = int(input("訓練輪數 (預設 50): ") or "50")
        lr = float(input("學習率 (預設 1e-4): ") or "1e-4")
        warmup_epochs = int(input("熱身輪數 (預設 10): ") or "10")
        
        # 混合精度選項
        if torch.cuda.is_available():
            use_mixed_precision = input("使用混合精度訓練 (FP16) 加速? (y/n，預設 y): ").strip().lower()
            use_mixed_precision = use_mixed_precision not in ['n', 'no', 'false']
            if use_mixed_precision:
                print("✅ 將使用混合精度 (FP16) - 預期提升 30-50% 速度")
            else:
                print("📝 將使用標準精度 (FP32)")
        else:
            use_mixed_precision = False
            print("⚠️ CPU 模式，無法使用混合精度")
        
        # 訓練模型
        print("\n🚀 開始訓練...")
        history = classifier.train(
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            warmup_epochs=warmup_epochs,
            auto_batch_size=auto_batch_size,
            use_mixed_precision=use_mixed_precision
        )
        
        # 繪製訓練歷史
        classifier.plot_training_history(history)
        
        # 評估模型
        print("\n📊 評估模型...")
        predictions, true_labels, accuracy = classifier.evaluate()
        
        print(f"\n🎯 最終結果:")
        print(f"  測試集準確率: {accuracy*100:.2f}%")
        print(f"  模型已保存至: best_memory_vit_character_classifier.pth")
        print(f"  類別映射已保存至: character_class_mapping.json")
        
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()