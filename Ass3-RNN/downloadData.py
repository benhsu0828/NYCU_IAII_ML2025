# 自動下載並設定噪音資料集
import os
import subprocess
import shutil
from pathlib import Path
import urllib.request
import zipfile
import librosa
import soundfile as sf
from tqdm import tqdm

def setup_noise_datasets():
    """
    自動下載並設定資料增強所需的噪音資料集
    """
    print("📦 開始下載噪音資料集...")
    
    # 1. 下載 MS-SNSD (微軟噪音資料集)
    print("\n1️⃣ 下載 MS-SNSD 背景噪音...")
    if not os.path.exists("MS-SNSD"):
        try:
            subprocess.run(["git", "clone", "https://github.com/microsoft/MS-SNSD.git"], check=True)
            print("✅ MS-SNSD 下載完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 下載失敗: {e}")
            print("   請確認已安裝 git 或手動下載：https://github.com/microsoft/MS-SNSD")
            return
    else:
        print("⏭️ MS-SNSD 已存在，跳過")
    
    # 2. 建立 background_noises 資料夾
    print("\n2️⃣ 設定背景噪音資料夾...")
    os.makedirs("background_noises", exist_ok=True)
    
    # 複製噪音檔案
    noise_source = "MS-SNSD/noise_train"
    if os.path.exists(noise_source):
        for file in os.listdir(noise_source):
            if file.endswith('.wav'):
                src = os.path.join(noise_source, file)
                dst = os.path.join("background_noises", file)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
        print(f"✅ 複製了 {len(os.listdir('background_noises'))} 個背景噪音檔案")
    
    # 3. 下載 ESC-50 用於短暫噪音
    print("\n3️⃣ 下載 ESC-50 短暫噪音...")
    if not os.path.exists("ESC-50-master"):
        try:
            print("   正在下載 ESC-50...")
            urllib.request.urlretrieve(
                "https://github.com/karolpiczak/ESC-50/archive/master.zip",
                "master.zip"
            )
            print("   正在解壓縮...")
            with zipfile.ZipFile("master.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("master.zip")
            print("✅ ESC-50 下載完成")
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            print("   請手動下載：https://github.com/karolpiczak/ESC-50/archive/master.zip")
    else:
        print("⏭️ ESC-50 已存在，跳過")
    
    # 4. 建立 short_noises 資料夾
    print("\n4️⃣ 設定短暫噪音資料夾...")
    os.makedirs("short_noises", exist_ok=True)
    
    esc50_audio = "ESC-50-master/audio"
    if os.path.exists(esc50_audio):
        count = 0
        for file in os.listdir(esc50_audio):
            if file.endswith('.wav'):
                src = os.path.join(esc50_audio, file)
                dst = os.path.join("short_noises", file)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    count += 1
                if count == 100: #取前100個檔案
                    break
        print(f"✅ 複製了 {count} 個短暫噪音檔案")

    
    # 5. 下載 RIR (房間脈衝響應)
    print("\n5️⃣ 下載 RIR 資料...")
    os.makedirs("rir", exist_ok=True)
    
    # 使用 OpenSLR 的 RIR 資料集 (較小且品質好)
    if len(os.listdir("rir")) == 0:
        try:
            print("   正在下載 RIR 資料集...")
            urllib.request.urlretrieve(
                "http://www.openslr.org/resources/28/rirs_noises.zip",
                "rirs.zip"
            )
            print("   正在解壓縮...")
            with zipfile.ZipFile("rirs.zip", 'r') as zip_ref:
                zip_ref.extractall("rir_temp")
            
            # 複製 RIR 檔案
            rir_source = "rir_temp/RIRS_NOISES/simulated_rirs"
            if os.path.exists(rir_source):
                count = 0
                for root, dirs, files in os.walk(rir_source):
                    for file in files:
                        if file.endswith('.wav'):
                            src = os.path.join(root, file)
                            dst = os.path.join("rir", file)
                            shutil.copy(src, dst)
                            count += 1
                            if count >= 50:  # 只取 50 個
                                break
                    if count >= 50:
                        break
            
            # 清理暫存檔
            shutil.rmtree("rir_temp", ignore_errors=True)
            os.remove("rirs.zip")
            print(f"✅ 設定了 {len(os.listdir('rir'))} 個 RIR 檔案")
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            print("   請手動下載：http://www.openslr.org/resources/28/rirs_noises.zip")
    else:
        print(f"⏭️ RIR 已存在 ({len(os.listdir('rir'))} 個檔案)")
    
    # 6. 顯示摘要
    print("\n" + "="*60)
    print("✅ 噪音資料集設定完成！")
    print("="*60)
    print(f"📁 background_noises: {len(os.listdir('background_noises'))} 個檔案")
    print(f"📁 short_noises: {len(os.listdir('short_noises'))} 個檔案")
    print(f"📁 rir: {len(os.listdir('rir'))} 個檔案")
    print("="*60)

TARGET_SR = 22050  # 目標採樣率

def resample_folder(folder_path):
    """將資料夾中的所有音訊重新採樣到目標採樣率"""
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    print(f"📁 處理 {folder_path}...")
    for file in tqdm(files):
        file_path = os.path.join(folder_path, file)
        
        # 載入音訊
        y, sr = librosa.load(file_path, sr=None)
        
        # 如果採樣率不同，重新採樣
        if sr != TARGET_SR:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            
            # 覆蓋原檔案
            sf.write(file_path, y_resampled, TARGET_SR, subtype='PCM_16')
            print(f"   ✅ {file}: {sr} Hz → {TARGET_SR} Hz")

# 執行設定
setup_noise_datasets()

# 處理所有噪音資料夾
resample_folder("background_noises")
resample_folder("short_noises")
resample_folder("rir")

print("✅ 所有噪音檔案已統一採樣率！")