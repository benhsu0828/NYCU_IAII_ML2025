import glob, os
import numpy as np
from scipy.io import wavfile
from audiomentations import Compose, SomeOf, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift, Shift, AddBackgroundNoise, AddShortNoises, PolarityInversion, ApplyImpulseResponse
from audiomentations.core.audio_loading_utils import load_sound_file
import nlpaug.augmenter.audio as naa
import nlpaug.flow as naf
import pandas as pd

TrainInPath = "train/train/train"
TrainOutPath = "train/preprocessed"
Test_InPath = "test-random/test-random"
Test_OutPath = "test-random/preprocessed"

train_csv_in = "train/train/train-toneless.csv"
train_csv_out = "train/train/trainAgg-toneless.csv"
new_rows = []


# 讀取 CSV 檔案
df = pd.read_csv(train_csv_in)
# print(f"讀取 {train_csv_in}，共 {len(df)} 筆資料")
# print(f"CSV 欄位: {df.columns.tolist()}")
# print(f"ID 型態: {df['id'].dtype}")
# print("\n前 5 筆資料:")
# print(df.head())

# 建立輸出資料夾
os.makedirs(TrainOutPath, exist_ok=True)
os.makedirs(Test_OutPath, exist_ok=True)

sr = 22050

augment1 = naf.Sometimes([
    naa.VtlpAug(sampling_rate=sr, zone=(0.0, 1.0), coverage=1.0, factor=(0.9, 1.1)),
], aug_p=0.4)

augment2 = Compose([
    AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.2),  # 修正：移除 _in
    TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False, p=0.4),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    AddBackgroundNoise(
        sounds_path="background_noises",
        min_snr_db=10,  # 修正：統一改成 min_snr_db
        max_snr_db=30.0,
        p=0.4),
    AddShortNoises(
        sounds_path="short_noises",
        min_snr_db=10,  # 修正：統一改成 min_snr_db
        max_snr_db=30.0,
        noise_rms="relative_to_whole_input",
        min_time_between_sounds=2.0,
        max_time_between_sounds=8.0,
        p=0.3),
    ApplyImpulseResponse(
        ir_path="rir", p=0.4
    )
])

# 處理訓練資料
print("🎵 處理訓練資料...")
train_files = [f for f in os.listdir(TrainInPath) if f.endswith(".wav")]

for i, file in enumerate(train_files, 1):
    try:
        samples, sample_rate = load_sound_file(
            os.path.join(TrainInPath, file), sample_rate=None
        )
        print(f"[{i}/{len(train_files)}] {file} - {sample_rate}Hz, {len(samples)} samples")
        
        # Augment/transform/perturb the audio data
        augmented_samples1 = augment1.augment(samples)
        augmented_samples2 = augment2(samples=augmented_samples1[0], sample_rate=sample_rate)
        
        # 轉成 16 kHz sampling, signed-integer, 16 bits
        # 步驟 1: 正規化到 [-1, 1]
        if augmented_samples2.max() > 0:
            augmented_samples2 = augmented_samples2 / np.max(np.abs(augmented_samples2))
        if samples.max() > 0:
            samples = samples / np.max(np.abs(samples))
        # 步驟 2: 轉換為 int16
        augmented_samples2_int16 = (augmented_samples2 * 32767).astype(np.int16)
        samples_int16 = (samples * 32767).astype(np.int16)

        wavfile.write(
            os.path.join(TrainOutPath, file), rate=16000, data=samples_int16
        )
        wavfile.write(
            os.path.join(TrainOutPath, file.split(".")[0] + "_augmented.wav"), rate=16000, data=augmented_samples2_int16
        )
        # 修改train-toneless.csv
        # 找到原始列
        original_row = df.loc[df['id'] == int(file.split(".")[0])].iloc[0].copy()
        # 修改檔名
        original_row['id'] = file.split(".")[0] + "_augmented"
        # 加入新列
        new_rows.append(original_row)

    except Exception as e:
        print(f"❌ 處理失敗 {file}: {e}")

print("\n✅ 訓練資料處理完成！\n")

# 處理測試資料
print("🎵 處理測試資料...")
test_files = [f for f in os.listdir(Test_InPath) if f.endswith(".wav")]
for i, file in enumerate(test_files, 1):
    try:
        samples, sample_rate = load_sound_file(
            os.path.join(Test_InPath, file), sample_rate=None
        )
        # print(f"[{i}/{len(test_files)}] {file}")
        
        if samples.max() > 0:
            samples = samples / np.max(np.abs(samples))

        # 步驟 2: 轉換為 int16
        samples_int16 = (samples * 32767).astype(np.int16)

        wavfile.write(
            os.path.join(Test_OutPath, file), rate=16000, data=samples_int16
        )
    except Exception as e:
        print(f"❌ 處理失敗 {file}: {e}")

print("\n✅ 測試資料處理完成！")
print(f"📁 訓練資料: {TrainOutPath}")
print(f"📁 測試資料: {Test_OutPath}")

# 儲存處理後的 CSV 檔案
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=False)
df.to_csv(train_csv_out, index=False)
