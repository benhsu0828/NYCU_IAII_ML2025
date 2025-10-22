#!/usr/bin/env python3
"""
🎯 EfficientNet 模型分析工具 - Confusion Matrix 與詳細評估

使用已訓練好的模型檔案進行：
- Confusion Matrix 繪製
- 分類報告生成
- 錯誤樣本分析
- Per-class 準確率分析
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import timm
import os
import glob
from pathlib import Path
import pandas as pd
from PIL import Image
import json

class ModelAnalyzer:
    """
    模型分析器 - 載入訓練好的模型並進行各種評估
    """
    
    def __init__(self, model_path, device=None):
        """
        初始化模型分析器
        
        Args:
            model_path: 訓練好的模型檔案路徑 (.pth)
            device: 計算設備
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.model_name = ""
        self.num_classes = 0
        
        print(f"🔍 模型分析器初始化")
        print(f"📁 模型檔案: {model_path}")
        print(f"🖥️ 使用設備: {self.device}")
        
        # 載入模型
        self._load_model()
        
    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型檔案: {self.model_path}")
        
        print(f"📂 載入模型...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 提取模型資訊
        self.model_name = checkpoint.get('model_name', 'unknown')
        self.num_classes = checkpoint.get('num_classes', 50)
        self.class_to_idx = checkpoint.get('class_to_idx', {})
        self.idx_to_class = checkpoint.get('idx_to_class', {})
        
        if not self.idx_to_class and self.class_to_idx:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 創建模型架構
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=False,  # 不載入預訓練權重
                num_classes=self.num_classes
            )
        except Exception as e:
            print(f"⚠️ 使用 timm 創建模型失敗: {e}")
            print("🔄 嘗試使用備用方案...")
            # 這裡可以添加備用的模型創建邏輯
            raise e
        
        # 載入訓練好的權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 顯示模型資訊
        val_acc = checkpoint.get('val_acc', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        
        print(f"✅ 模型載入成功！")
        print(f"   模型: {self.model_name}")
        print(f"   類別數: {self.num_classes}")
        print(f"   驗證準確率: {val_acc}")
        print(f"   訓練輪數: {epoch}")
        
    def get_transforms(self):
        """獲取測試用的資料變換 (與訓練時的驗證變換相同)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_test_data(self, data_path, batch_size=32):
        """
        載入測試資料
        
        Args:
            data_path: 測試資料路徑
            batch_size: batch size
        """
        print(f"\n📊 載入測試資料: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到資料路徑: {data_path}")
        
        # 創建資料集
        transform = self.get_transforms()
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
        # 確保類別映射一致
        if dataset.class_to_idx != self.class_to_idx:
            print("⚠️ 警告: 測試資料的類別映射與模型不一致")
            print(f"   模型類別數: {len(self.class_to_idx)}")
            print(f"   測試資料類別數: {len(dataset.class_to_idx)}")
            
            # 使用模型的類別映射
            dataset.class_to_idx = self.class_to_idx
        
        # 創建資料載入器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 不打亂順序，便於分析
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✅ 測試資料載入完成")
        print(f"   總樣本數: {len(dataset)}")
        print(f"   類別數: {len(dataset.classes)}")
        print(f"   Batch數: {len(dataloader)}")
        
        return dataloader, dataset
    
    def predict(self, dataloader):
        """
        對測試資料進行預測
        
        Args:
            dataloader: 測試資料載入器
            
        Returns:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_probs: 預測機率
        """
        print(f"\n🔮 開始預測...")
        
        self.model.eval()
        y_true = []
        y_pred = []
        y_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 預測
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = outputs.argmax(dim=1)
                
                # 收集結果
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probabilities.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   已處理: {(batch_idx + 1) * dataloader.batch_size} 樣本")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"✅ 預測完成！")
        print(f"   總樣本數: {len(y_true)}")
        print(f"   整體準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return y_true, y_pred, y_probs
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, figsize=(15, 12)):
        """
        繪製 Confusion Matrix
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            save_path: 保存路徑
            figsize: 圖片尺寸
        """
        print(f"\n📊 繪製 Confusion Matrix...")
        
        # 計算 confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 創建類別名稱列表
        class_names = [self.idx_to_class.get(i, f'Class_{i}') for i in range(self.num_classes)]
        
        # 創建圖片
        plt.figure(figsize=figsize)
        
        # 繪製熱力圖
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Sample Count'}
        )
        
        plt.title(f'Confusion Matrix\n{self.model_name} - Accuracy: {accuracy_score(y_true, y_pred):.4f}', 
                 fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存圖片
        if save_path is None:
            save_path = f"{self.model_name}_confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion Matrix 已保存: {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_normalized_confusion_matrix(self, y_true, y_pred, save_path=None, figsize=(15, 12)):
        """
        繪製標準化的 Confusion Matrix (百分比)
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            save_path: 保存路徑
            figsize: 圖片尺寸
        """
        print(f"\n📊 繪製標準化 Confusion Matrix...")
        
        # 計算標準化的 confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 創建類別名稱列表
        class_names = [self.idx_to_class.get(i, f'Class_{i}') for i in range(self.num_classes)]
        
        # 創建圖片
        plt.figure(figsize=figsize)
        
        # 繪製熱力圖
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'},
            vmin=0, vmax=1
        )
        
        plt.title(f'Normalized Confusion Matrix (%)\n{self.model_name} - Accuracy: {accuracy_score(y_true, y_pred):.4f}', 
                 fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存圖片
        if save_path is None:
            save_path = f"{self.model_name}_confusion_matrix_normalized.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 標準化 Confusion Matrix 已保存: {save_path}")
        
        plt.show()
        
        return cm_normalized
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """
        生成詳細的分類報告
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            save_path: 保存路徑
        """
        print(f"\n📋 生成分類報告...")
        
        # 創建類別名稱列表
        class_names = [self.idx_to_class.get(i, f'Class_{i}') for i in range(self.num_classes)]
        
        # 生成分類報告
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # 轉換為 DataFrame 便於顯示和保存
        df = pd.DataFrame(report).transpose()
        
        # 顯示報告
        print("📊 分類報告:")
        print("=" * 60)
        print(df.round(4))
        
        # 保存為 CSV
        if save_path is None:
            save_path = f"{self.model_name}_classification_report.csv"
        
        df.to_csv(save_path)
        print(f"💾 分類報告已保存: {save_path}")
        
        return df
    
    def analyze_per_class_accuracy(self, y_true, y_pred, save_path=None):
        """
        分析每個類別的準確率
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            save_path: 保存路徑
        """
        print(f"\n🎯 分析每類別準確率...")
        
        # 計算每個類別的準確率
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # 創建結果 DataFrame
        class_names = [self.idx_to_class.get(i, f'Class_{i}') for i in range(self.num_classes)]
        
        results = []
        for i, (class_name, accuracy) in enumerate(zip(class_names, per_class_acc)):
            total_samples = cm.sum(axis=1)[i]
            correct_predictions = cm.diagonal()[i]
            
            results.append({
                'Class': class_name,
                'Accuracy': accuracy,
                'Correct': correct_predictions,
                'Total': total_samples,
                'Error_Count': total_samples - correct_predictions
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Accuracy', ascending=False)
        
        # 顯示結果
        print("🏆 各類別準確率排名:")
        print("=" * 60)
        for _, row in df.head(10).iterrows():
            print(f"{row['Class']:20} | 準確率: {row['Accuracy']:.4f} ({row['Accuracy']*100:6.2f}%) | "
                  f"正確/總數: {row['Correct']}/{row['Total']}")
        
        print("\n❌ 準確率最低的類別:")
        print("-" * 60)
        for _, row in df.tail(5).iterrows():
            print(f"{row['Class']:20} | 準確率: {row['Accuracy']:.4f} ({row['Accuracy']*100:6.2f}%) | "
                  f"錯誤數: {row['Error_Count']}")
        
        # 繪製準確率分布圖
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(df)), df['Accuracy'], 
                      color=['green' if acc >= 0.9 else 'orange' if acc >= 0.7 else 'red' 
                            for acc in df['Accuracy']])
        
        plt.title(f'Per-Class Accuracy\n{self.model_name}', fontsize=16)
        plt.xlabel('Classes (sorted by accuracy)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(range(len(df)), df['Class'], rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加準確率文字
        for i, (bar, acc) in enumerate(zip(bars, df['Accuracy'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存圖片和數據
        if save_path is None:
            save_path = f"{self.model_name}_per_class_accuracy"
        
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        df.to_csv(f"{save_path}.csv", index=False)
        
        print(f"💾 每類別準確率已保存: {save_path}.png, {save_path}.csv")
        
        plt.show()
        
        return df
    
    def find_misclassified_samples(self, y_true, y_pred, y_probs, dataset, num_samples=5):
        """
        找出分類錯誤的樣本
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_probs: 預測機率
            dataset: 資料集
            num_samples: 每個類別顯示的錯誤樣本數
        """
        print(f"\n🔍 分析分類錯誤的樣本...")
        
        # 找出錯誤分類的樣本
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        print(f"📊 錯誤分類統計:")
        print(f"   總樣本數: {len(y_true)}")
        print(f"   錯誤樣本數: {len(misclassified_indices)}")
        print(f"   錯誤率: {len(misclassified_indices)/len(y_true)*100:.2f}%")
        
        # 分析每個類別的錯誤
        class_names = [self.idx_to_class.get(i, f'Class_{i}') for i in range(self.num_classes)]
        
        for true_class_idx in range(self.num_classes):
            class_name = class_names[true_class_idx]
            
            # 找出這個類別的錯誤樣本
            class_misclassified = misclassified_indices[
                y_true[misclassified_indices] == true_class_idx
            ]
            
            if len(class_misclassified) > 0:
                print(f"\n❌ {class_name} 的錯誤分類:")
                
                # 按照預測信心度排序 (信心度越高的錯誤越值得關注)
                confidences = np.max(y_probs[class_misclassified], axis=1)
                sorted_indices = class_misclassified[np.argsort(confidences)[::-1]]
                
                for i, sample_idx in enumerate(sorted_indices[:num_samples]):
                    true_label = y_true[sample_idx]
                    pred_label = y_pred[sample_idx]
                    confidence = np.max(y_probs[sample_idx])
                    
                    true_class_name = class_names[true_label]
                    pred_class_name = class_names[pred_label]
                    
                    print(f"   樣本 {sample_idx}: {true_class_name} → {pred_class_name} "
                          f"(信心度: {confidence:.3f})")
        
        return misclassified_indices

def get_available_models():
    """尋找可用的模型檔案"""
    print("🔍 尋找可用的模型檔案...")
    
    # 搜尋模式
    patterns = [
        "*.pth",
        "efficientnet*.pth", 
        "convnext*.pth",
        "*_epoch_*.pth",
        "best_*.pth"
    ]
    
    model_files = []
    for pattern in patterns:
        model_files.extend(glob.glob(pattern))
    
    # 去除重複並排序
    model_files = list(set(model_files))
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return model_files

def get_available_data_paths():
    """獲取可用的測試資料路徑"""
    print("📁 尋找可用的測試資料...")
    
    # 可能的測試資料路徑
    possible_paths = [
        "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/val",
        "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/val",
        "Dataset/preprocessed/val",
        "../Dataset/preprocessed/val",
        "E:/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/test",
        "/mnt/e/NYCU/NYCU_IAII_ML2025/Ass2-Classification/Dataset/preprocessed/test"
    ]
    
    available_paths = []
    for path in possible_paths:
        if os.path.exists(path):
            available_paths.append(path)
    
    return available_paths

def main():
    """主函數"""
    print("🎯 EfficientNet 模型分析工具")
    print("=" * 60)
    print("📊 功能: Confusion Matrix, 分類報告, 錯誤分析")
    print("=" * 60)
    
    # 1. 選擇模型檔案
    # model_files = get_available_models()
    model_files = ["convnext_tiny_epoch_013_acc_99.91.pth"]

    if not model_files:
        print("❌ 找不到任何模型檔案！")
        print("💡 請確認當前目錄下有 .pth 檔案")
        return
    
    print(f"\n📂 找到 {len(model_files)} 個模型檔案:")
    for i, file in enumerate(model_files, 1):
        # 獲取檔案資訊
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        mod_time = os.path.getmtime(file)
        time_str = pd.Timestamp.fromtimestamp(mod_time).strftime('%m/%d %H:%M')
        
        print(f"  {i}. {file} ({file_size:.1f}MB, {time_str})")
    
    try:
        choice = int(input(f"\n請選擇模型檔案 (1-{len(model_files)}): ")) - 1
        model_path = model_files[choice]
        print(f"✅ 選擇模型: {model_path}")
    except (ValueError, IndexError):
        print("❌ 選擇無效，使用最新的模型檔案")
        model_path = model_files[0]
    
    # 2. 選擇測試資料
    data_paths = get_available_data_paths()
    
    if not data_paths:
        print("❌ 找不到測試資料路徑！")
        print("💡 請確認以下路徑存在:")
        print("   - Dataset/preprocessed/val")
        print("   - Dataset/preprocessed/test")
        return
    
    print(f"\n📁 找到 {len(data_paths)} 個資料路徑:")
    for i, path in enumerate(data_paths, 1):
        print(f"  {i}. {path}")
    
    try:
        choice = int(input(f"\n請選擇測試資料路徑 (1-{len(data_paths)}): ")) - 1
        data_path = data_paths[choice]
        print(f"✅ 選擇資料: {data_path}")
    except (ValueError, IndexError):
        print("❌ 選擇無效，使用第一個資料路徑")
        data_path = data_paths[0]
    
    # 3. 初始化分析器
    try:
        analyzer = ModelAnalyzer(model_path)
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return
    
    # 4. 載入測試資料
    try:
        dataloader, dataset = analyzer.load_test_data(data_path)
    except Exception as e:
        print(f"❌ 資料載入失敗: {e}")
        return
    
    # 5. 進行預測
    try:
        y_true, y_pred, y_probs = analyzer.predict(dataloader)
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        return
    
    # 6. 生成分析報告
    print(f"\n🎯 開始生成分析報告...")
    
    # Confusion Matrix
    cm = analyzer.plot_confusion_matrix(y_true, y_pred)
    
    # 標準化 Confusion Matrix  
    cm_norm = analyzer.plot_normalized_confusion_matrix(y_true, y_pred)
    
    # 分類報告
    report_df = analyzer.generate_classification_report(y_true, y_pred)
    
    # 每類別準確率分析
    per_class_df = analyzer.analyze_per_class_accuracy(y_true, y_pred)
    
    # 錯誤樣本分析
    misclassified_indices = analyzer.find_misclassified_samples(
        y_true, y_pred, y_probs, dataset
    )
    
    print(f"\n🎉 分析完成！")
    print(f"📁 所有結果已保存到當前目錄")
    print(f"📊 總體準確率: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    main()