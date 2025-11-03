"""
台語語音辨識資料增強腳本 - 整合 RawBoost 和 Audiomentations
作者: GitHub Copilot
用途: 對訓練資料進行多種增強，提升模型泛化能力
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa
from tqdm import tqdm
import argparse
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

# ==================== RawBoost 參數設定 ====================
class RawBoostConfig:
    """RawBoost 演算法參數"""
    def __init__(self):
        # Algorithm 1: LnL (Linear and Non-linear convolutive noise)
        self.N_f = 5
        self.nBands = 5
        self.minF = 20
        self.maxF = 8000
        self.minBW = 100
        self.maxBW = 1000
        self.minCoeff = 10
        self.maxCoeff = 100
        self.minG = 0
        self.maxG = 0
        self.minBiasLinNonLin = 5
        self.maxBiasLinNonLin = 20
        
        # Algorithm 2: ISD (Impulsive Signal Dependent noise)
        self.P = 10
        self.g_sd = 2
        
        # Algorithm 3: SSI (Stationary Signal Independent noise)
        self.SNRmin = 10
        self.SNRmax = 40

# ==================== 資料增強函數 ====================

def apply_rawboost(audio, sr, args, algo_type):
    """
    應用 RawBoost 資料增強
    
    Args:
        audio: 音訊陣列 (numpy array)
        sr: 採樣率
        args: RawBoost 參數
        algo_type: 演算法類型 (1-7)
            1: LnL (線性/非線性卷積噪音)
            2: ISD (脈衝噪音)
            3: SSI (平穩加性噪音)
            4: 1+2+3 (串聯)
            5: 1+2 (串聯)
            6: 1+3 (串聯)
            7: 2+3 (串聯)
    
    Returns:
        增強後的音訊
    """
    
    # Algorithm 1: Convolutive noise
    if algo_type == 1:
        audio = LnL_convolutive_noise(
            audio, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
    
    # Algorithm 2: Impulsive noise
    elif algo_type == 2:
        audio = ISD_additive_noise(audio, args.P, args.g_sd)
    
    # Algorithm 3: Stationary additive noise
    elif algo_type == 3:
        audio = SSI_additive_noise(
            audio, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    
    # Algorithm 4: All three in series (1+2+3)
    elif algo_type == 4:
        audio = LnL_convolutive_noise(
            audio, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        audio = ISD_additive_noise(audio, args.P, args.g_sd)
        audio = SSI_additive_noise(
            audio, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    
    # Algorithm 5: 1+2 in series
    elif algo_type == 5:
        audio = LnL_convolutive_noise(
            audio, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        audio = ISD_additive_noise(audio, args.P, args.g_sd)
    
    # Algorithm 6: 1+3 in series
    elif algo_type == 6:
        audio = LnL_convolutive_noise(
            audio, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        audio = SSI_additive_noise(
            audio, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    
    # Algorithm 7: 2+3 in series
    elif algo_type == 7:
        audio = ISD_additive_noise(audio, args.P, args.g_sd)
        audio = SSI_additive_noise(
            audio, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, sr
        )
    
    return audio


def normalize_audio(audio):
    """正規化音訊到 [-1, 1]"""
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def audio_to_int16(audio):
    """將音訊轉換為 int16 格式"""
    audio = normalize_audio(audio)
    return (audio * 32767).astype(np.int16)


# ==================== 主要處理函數 ====================

def augment_dataset(
    input_dir,
    output_dir,
    csv_input,
    csv_output,
    algo_types=[1, 2, 3],
    target_sr=16000,
    num_augmentations_per_algo=1
):
    """
    對整個資料集進行 RawBoost 增強
    
    Args:
        input_dir: 原始音訊目錄
        output_dir: 輸出目錄
        csv_input: 輸入 CSV 檔案
        csv_output: 輸出 CSV 檔案
        algo_types: 要使用的演算法類型列表
        target_sr: 目標採樣率
        num_augmentations_per_algo: 每種演算法生成的增強版本數量
    """
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入 CSV
    df = pd.read_csv(csv_input)
    print(f"📊 讀取 {csv_input}，共 {len(df)} 筆資料")
    print(f"CSV 欄位: {df.columns.tolist()}\n")
    
    # 初始化 RawBoost 參數
    rawboost_args = RawBoostConfig()
    
    # 儲存新增的資料列
    new_rows = []
    
    # 取得所有音訊檔案
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"🎵 開始處理 {len(audio_files)} 個音訊檔案...")
    print(f"📌 使用演算法: {algo_types}")
    print(f"📌 每種演算法生成 {num_augmentations_per_algo} 個增強版本\n")
    
    # 進度條
    pbar = tqdm(audio_files, desc="處理音訊檔案")
    
    for file in pbar:
        try:
            # 更新進度條描述
            pbar.set_description(f"處理 {file}")
            
            # 載入音訊
            audio_path = os.path.join(input_dir, file)
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # 取得檔案 ID (不含副檔名)
            file_id = file.replace('.wav', '')

            # ✅ 檢查 file_id 是否為純數字
            if not file_id.isdigit():
                print(f"\n⚠️ 跳過非數字檔名: {file}")
                continue
            
            # 1️⃣ 儲存原始檔案 (標準化處理)
            audio_normalized = audio_to_int16(audio)
            wavfile.write(
                os.path.join(output_dir, file),
                target_sr,
                audio_normalized
            )
            
            # 2️⃣ 對每種演算法生成增強版本
            for algo_type in algo_types:
                for aug_idx in range(num_augmentations_per_algo):
                    # 應用 RawBoost 增強
                    augmented_audio = apply_rawboost(audio.copy(), sr, rawboost_args, algo_type)
                    
                    # 轉換為 int16
                    augmented_audio_int16 = audio_to_int16(augmented_audio)
                    
                    # 生成新檔名
                    if num_augmentations_per_algo > 1:
                        aug_filename = f"{file_id}_rawboost_algo{algo_type}_v{aug_idx+1}.wav"
                    else:
                        aug_filename = f"{file_id}_rawboost_algo{algo_type}.wav"
                    
                    # 儲存增強音訊
                    wavfile.write(
                        os.path.join(output_dir, aug_filename),
                        target_sr,
                        augmented_audio_int16
                    )
                    
                    # 更新 CSV (找到原始列並複製)
                    original_row = df.loc[df['id'] == int(file_id)].iloc[0].copy()
                    original_row['id'] = aug_filename.replace('.wav', '')
                    new_rows.append(original_row)
        
        except Exception as e:
            print(f"\n❌ 處理失敗 {file}: {e}")
            continue
    
    print(f"\n✅ 音訊處理完成！")
    print(f"   原始檔案: {len(audio_files)}")
    print(f"   增強檔案: {len(new_rows)}")
    print(f"   總計: {len(audio_files) + len(new_rows)}\n")
    
    # 合併原始和增強資料的 CSV
    df_augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_augmented.to_csv(csv_output, index=False)
    
    print(f"✅ CSV 檔案已更新: {csv_output}")
    print(f"   總筆數: {len(df_augmented)}")


# ==================== 命令列介面 ====================

def main():
    parser = argparse.ArgumentParser(description='台語語音資料 RawBoost 增強')
    
    parser.add_argument('--input_dir', type=str, default='./train/preprocessed',
                        help='輸入音訊目錄')
    parser.add_argument('--output_dir', type=str, default='./train/rawboost_augmented',
                        help='輸出音訊目錄')
    parser.add_argument('--csv_input', type=str, default='./train/trainAgg-toneless.csv',
                        help='輸入 CSV 檔案')
    parser.add_argument('--csv_output', type=str, default='./train/trainAgg-toneless-rawboost.csv',
                        help='輸出 CSV 檔案')
    parser.add_argument('--algo_types', type=int, nargs='+', default=[3, 6],
                        help='RawBoost 演算法類型 (1-7)')
    parser.add_argument('--num_aug', type=int, default=1,
                        help='每種演算法生成的增強版本數量')
    parser.add_argument('--sr', type=int, default=16000,
                        help='目標採樣率')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 台語語音資料 RawBoost 增強腳本")
    print("=" * 70)
    print(f"📂 輸入目錄: {args.input_dir}")
    print(f"📂 輸出目錄: {args.output_dir}")
    print(f"📄 輸入 CSV: {args.csv_input}")
    print(f"📄 輸出 CSV: {args.csv_output}")
    print(f"🔧 演算法類型: {args.algo_types}")
    print(f"🔢 增強版本數: {args.num_aug}")
    print(f"🎵 採樣率: {args.sr} Hz")
    print("=" * 70)
    print()
    
    # 執行增強
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_input=args.csv_input,
        csv_output=args.csv_output,
        algo_types=args.algo_types,
        target_sr=args.sr,
        num_augmentations_per_algo=args.num_aug
    )
    
    print("\n" + "=" * 70)
    print("✅ 所有處理完成！")
    print("=" * 70)
    
    print("\n💡 演算法說明:")
    print("   1 = LnL (線性/非線性卷積噪音)")
    print("   2 = ISD (脈衝噪音)")
    print("   3 = SSI (平穩加性噪音) ✅ 推薦用於語音辨識")
    print("   4 = 1+2+3 (串聯)")
    print("   5 = 1+2 (串聯)")
    print("   6 = 1+3 (串聯) ✅ 推薦用於語音辨識")
    print("   7 = 2+3 (串聯)")


if __name__ == "__main__":
    main()
