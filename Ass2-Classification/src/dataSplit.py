# ...existing code...
import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_train_val_by_folder(
    src_dir,
    dst_dir,
    val_ratio=0.2,
    seed=42,
    copy=True,
    min_val_samples=1
):
    """
    將 src_dir 裡的每個 class 子資料夾按比例切成 train/val，
    輸出到 dst_dir/train/<class>/ 和 dst_dir/val/<class>/。

    參數:
      src_dir: 原始資料目錄，結構應為 src_dir/<class>/*.jpg
      dst_dir: 輸出目錄
      val_ratio: 驗證集比例 (0..1)
      seed: 隨機種子
      copy: True => 複製檔案；False => 移動檔案
      min_val_samples: 每類別至少要放到 val 的樣本數 (若不足會放 0 或 1，並警告)
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    train_root = dst_dir / "train"
    val_root = dst_dir / "val"

    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    classes = [p for p in src_dir.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found in {src_dir}")

    for cls in sorted(classes):
        imgs = sorted([p for p in cls.rglob("*") if p.is_file()])
        n = len(imgs)
        if n == 0:
            print(f"⚠️  類別 {cls.name} 沒有檔案，跳過")
            continue

        # 若類別太少，保證至少 min_val_samples 放到 val（視情況）
        if n <= min_val_samples:
            train_files = imgs
            val_files = []
            if min_val_samples > 0 and n > 0:
                # 如果只想要 1 個 val，且 n>1 才移一個到 val
                if n > 1:
                    val_files = [imgs[0]]
                    train_files = imgs[1:]
        else:
            test_size = val_ratio
            # 針對該類別做 split（每類別維持比例）
            train_files, val_files = train_test_split(
                imgs, test_size=test_size, random_state=seed, shuffle=True
            )

        # 建目錄
        tgt_train_cls = train_root / cls.name
        tgt_val_cls = val_root / cls.name
        tgt_train_cls.mkdir(parents=True, exist_ok=True)
        tgt_val_cls.mkdir(parents=True, exist_ok=True)

        # 複製或移動
        op = shutil.copy2 if copy else shutil.move
        for p in train_files:
            dstp = tgt_train_cls / p.name
            if not dstp.exists():
                op(p, dstp)
        for p in val_files:
            dstp = tgt_val_cls / p.name
            if not dstp.exists():
                op(p, dstp)

        print(f"Class {cls.name}: total={n}, train={len(train_files)}, val={len(val_files)}")

    print("Class number:", len(classes))
    print("完成：train/val 資料已建好於", dst_dir)


if __name__ == "__main__":
    #python dataSplit.py
    parser = argparse.ArgumentParser(description="Split dataset with folder-per-class into train/val")
    parser.add_argument("--src", default="E:\\NYCU\\NYCU_IAII_ML2025\\Ass2-Classification\\Dataset\\raw\\train", 
                        help="source folder containing character folders")
    parser.add_argument("--dst", default="E:\\NYCU\\NYCU_IAII_ML2025\\Ass2-Classification\\Dataset\\preprocessed", 
                        help="destination root, will create train/ and val/ inside")
    parser.add_argument("--val", type=float, default=0.2, help="validation ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible splits")
    parser.add_argument("--copy", action="store_true", 
                        help="copy files instead of moving (default: move files)")
    parser.add_argument("--min_val", type=int, default=1, help="minimum validation samples per class")
    args = parser.parse_args()

    print(f"🗂️  資料分割設定:")
    print(f"   來源目錄: {args.src}")
    print(f"   輸出目錄: {args.dst}")
    print(f"   驗證集比例: {args.val}")
    print(f"   操作模式: {'複製' if args.copy else '移動'}")
    print(f"   隨機種子: {args.seed}")

    split_train_val_by_folder(
        src_dir=args.src,
        dst_dir=args.dst,
        val_ratio=args.val,
        seed=args.seed,
        copy=args.copy,
        min_val_samples=args.min_val
    )

    print("\n✅ 資料分割完成！ConvNeXt-V2 可直接使用資料夾結構進行訓練。")
