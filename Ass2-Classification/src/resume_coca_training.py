#!/usr/bin/env python3
"""
🔄 CoCa 分類器續訓腳本

這個腳本專門用於從已保存的檢查點繼續訓練 CoCa 分類器
使用方法：python resume_coca_training.py
"""

import os
import sys
from CoCa_character_classifier import CoCaCharacterClassifier

def main():
    """續訓主函數"""
    print("🔄 CoCa 分類器續訓工具")
    print("=" * 40)
    
    # 檢查模型目錄
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print(f"❌ 模型目錄不存在: {model_dir}")
        return
    
    # 尋找檢查點
    print("🔍 搜尋可用檢查點...")
    checkpoints = CoCaCharacterClassifier.find_checkpoints(model_dir)
    
    if not checkpoints:
        print("❌ 沒有找到可用的檢查點")
        print("💡 請先運行完整訓練創建檢查點")
        return
    
    # 選擇檢查點
    print(f"\n請選擇要續訓的檢查點 (1-{len(checkpoints)}): ", end="")
    try:
        choice = int(input()) - 1
        if not (0 <= choice < len(checkpoints)):
            print("❌ 無效選擇")
            return
        
        selected_checkpoint = checkpoints[choice]
        checkpoint_path = selected_checkpoint['path']
        
        print(f"✅ 選擇檢查點: {selected_checkpoint['filename']}")
        print(f"📊 當前狀態: 第{selected_checkpoint['epoch']+1}輪, 準確率 {selected_checkpoint['accuracy']:.2f}%")
        
    except ValueError:
        print("❌ 輸入無效")
        return
    
    # 續訓參數設定
    print("\n⚙️ 續訓參數設定:")
    
    # 額外訓練輪數
    try:
        additional_epochs = int(input("額外訓練輪數 (預設 15): ") or "15")
    except:
        additional_epochs = 15
    
    # 新學習率
    try:
        new_lr_input = input("新學習率 (預設 1e-5): ") or "1e-5"
        new_lr = float(new_lr_input)
    except:
        new_lr = 1e-5
    
    # 批次大小
    try:
        batch_size = int(input("批次大小 (預設 16): ") or "16")
    except:
        batch_size = 16
    
    print(f"\n📋 續訓配置:")
    print(f"   檢查點: {os.path.basename(checkpoint_path)}")
    print(f"   額外輪數: {additional_epochs}")
    print(f"   學習率: {new_lr}")
    print(f"   批次大小: {batch_size}")
    
    confirm = input("\n確認開始續訓？(y/N): ").lower()
    if confirm != 'y':
        print("❌ 取消續訓")
        return
    
    try:
        # 初始化分類器
        print("\n🚀 初始化 CoCa 分類器...")
        classifier = CoCaCharacterClassifier(
            num_classes=50,
            coca_model='coca_ViT-B-32'
        )
        
        # 準備資料
        print("📊 準備資料...")
        data_paths = {
            'train': 'Dataset/train',
            'val': 'Dataset/val'
        }
        train_dataset, val_dataset = classifier.prepare_data(data_paths)
        
        # 開始續訓
        print("🔄 開始續訓...")
        history = classifier.resume_training(
            checkpoint_path=checkpoint_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            additional_epochs=additional_epochs,
            new_lr=new_lr,
            batch_size=batch_size,
            patience=8,  # 更早的早停
            save_dir=model_dir
        )
        
        print("\n🎉 續訓完成!")
        
        # 顯示最終結果
        if history and 'val_acc' in history:
            final_acc = max(history['val_acc'])
            print(f"🎯 最終準確率: {final_acc:.2f}%")
        
    except Exception as e:
        print(f"❌ 續訓過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()