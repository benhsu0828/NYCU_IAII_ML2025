import torch
import torch
from vit_pytorch.learnable_memory_vit import ViT, Adapter

def test_memory_vit_concept():
    """
    MemoryViT 概念測試
    
    這個範例展示如何使用一個預訓練的 ViT 模型
    配合多個 Adapter 來處理不同的分類任務
    """
    print("🧠 MemoryViT 概念演示")
    print("=" * 50)
    
    # 步驟 1: 創建基礎 ViT 模型（通常在大規模資料上預訓練）
    print("📌 步驟 1: 創建基礎 ViT 模型")
    base_vit = ViT(
        image_size=256,
        patch_size=16,
        num_classes=1000,    # 預訓練任務的類別數（如 ImageNet）
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    print(f"✅ 基礎 ViT 創建完成，參數量: {sum(p.numel() for p in base_vit.parameters()):,}")
    
    # 測試基礎模型
    img = torch.randn(4, 3, 256, 256)
    logits = base_vit(img)  # (4, 1000)
    print(f"✅ 基礎模型輸出形狀: {logits.shape}")
    
    # 在這裡進行你的預訓練...
    print("🔄 [這裡進行大規模預訓練...]")
    
    # 步驟 2: 任務 1 - 角色性別分類（2 類）
    print("\n📌 步驟 2: 創建性別分類 Adapter")
    gender_adapter = Adapter(
        vit=base_vit,               # 使用預訓練的 ViT（參數凍結）
        num_classes=2,              # 男性 vs 女性
        num_memories_per_layer=5    # 每層 5 個可學習記憶
    )
    
    # 測試性別分類
    gender_output = gender_adapter(img)  # (4, 2)
    print(f"✅ 性別分類輸出形狀: {gender_output.shape}")
    
    # 步驟 3: 任務 2 - 角色情緒分類（5 類）
    print("\n📌 步驟 3: 創建情緒分類 Adapter") 
    emotion_adapter = Adapter(
        vit=base_vit,               # 同樣的預訓練 ViT
        num_classes=5,              # 5 種情緒
        num_memories_per_layer=8    # 更複雜的任務需要更多記憶
    )
    
    # 測試情緒分類
    emotion_output = emotion_adapter(img)  # (4, 5)
    print(f"✅ 情緒分類輸出形狀: {emotion_output.shape}")
    
    # 步驟 4: 任務 3 - 50 類角色分類
    print("\n📌 步驟 4: 創建 50 類角色分類 Adapter")
    character_adapter = Adapter(
        vit=base_vit,               # 同樣的預訓練 ViT
        num_classes=50,             # 50 個角色類別
        num_memories_per_layer=20   # 更多類別需要更多記憶
    )
    
    # 測試角色分類
    character_output = character_adapter(img)  # (4, 50)
    print(f"✅ 角色分類輸出形狀: {character_output.shape}")
    
    # 參數量比較
    print("\n📊 參數量比較:")
    base_params = sum(p.numel() for p in base_vit.parameters())
    gender_params = sum(p.numel() for p in gender_adapter.parameters() if p.requires_grad)
    emotion_params = sum(p.numel() for p in emotion_adapter.parameters() if p.requires_grad)
    character_params = sum(p.numel() for p in character_adapter.parameters() if p.requires_grad)
    
    print(f"  基礎 ViT: {base_params:,} 參數")
    print(f"  性別 Adapter: {gender_params:,} 參數 ({gender_params/base_params*100:.2f}%)")
    print(f"  情緒 Adapter: {emotion_params:,} 參數 ({emotion_params/base_params*100:.2f}%)")
    print(f"  角色 Adapter: {character_params:,} 參數 ({character_params/base_params*100:.2f}%)")
    
    print("\n💡 MemoryViT 的優勢:")
    print("  ✅ 一個基礎模型服務多個任務")
    print("  ✅ 共享視覺特徵表示")
    print("  ✅ Adapter 參數量很小")
    print("  ✅ 避免災難性遺忘")
    print("  ✅ 新增任務只需訓練 Adapter")
    
    return base_vit, gender_adapter, emotion_adapter, character_adapter

if __name__ == "__main__":
    # 運行概念演示
    base_vit, gender_adapter, emotion_adapter, character_adapter = test_memory_vit_concept()
    
    print("\n🚀 現在你可以:")
    print("  1. 使用 MemoryViT_character_classifier.py 訓練 50 類角色分類")
    print("  2. 同時創建其他任務的 Adapter（性別、情緒等）")
    print("  3. 所有任務共享同一個基礎 ViT 的視覺特徵")

# do your usual training with ViT
# ...


# then, to finetune, just pass the ViT into the Adapter class
# you can do this for multiple Adapters, as shown below

adapter1 = Adapter(
    vit = v,
    num_classes = 2,               # number of output classes for this specific task
    num_memories_per_layer = 5     # number of learnable memories per layer, 10 was sufficient in paper
)

logits1 = adapter1(img) # (4, 2) - predict 2 classes off frozen ViT backbone with learnable memories and task specific head

# yet another task to finetune on, this time with 4 classes

adapter2 = Adapter(
    vit = v,
    num_classes = 4,
    num_memories_per_layer = 10
)

logits2 = adapter2(img) # (4, 4) - predict 4 classes off frozen ViT backbone with learnable memories and task specific head
