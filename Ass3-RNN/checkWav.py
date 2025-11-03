# 檢查 WAV 檔案格式 (支援所有格式)
import os
import struct
import numpy as np
from scipy.io import wavfile
import soundfile as sf

def check_wav_format(file_path):
    """
    檢查 WAV 檔案的詳細格式資訊 (支援所有格式)
    """
    if not os.path.exists(file_path):
        print(f"❌ 檔案不存在: {file_path}")
        return
    
    print("="*60)
    print(f"📄 檔案: {file_path}")
    print("="*60)
    
    try:
        # 使用 soundfile 讀取 (支援更多格式)
        info = sf.info(file_path)
        data, sample_rate = sf.read(file_path)
        
        # 基本資訊
        n_channels = info.channels
        framerate = info.samplerate
        n_frames = info.frames
        duration = info.duration
        subtype = info.subtype  # 資料格式
        format_name = info.format  # 檔案格式
        
        # 判斷位元深度
        bit_depth_map = {
            'PCM_16': 16,
            'PCM_24': 24,
            'PCM_32': 32,
            'FLOAT': 32,
            'DOUBLE': 64,
            'PCM_U8': 8,
        }
        bit_depth = bit_depth_map.get(subtype, 'unknown')
        
        # 判斷資料類型
        if subtype.startswith('PCM'):
            if subtype == 'PCM_U8':
                data_type = "unsigned 8-bit integer"
            else:
                data_type = f"signed {bit_depth}-bit integer"
        elif subtype == 'FLOAT':
            data_type = "32-bit float"
        elif subtype == 'DOUBLE':
            data_type = "64-bit float"
        else:
            data_type = subtype
        
        print(f"\n📊 音訊格式:")
        print(f"   檔案格式 (Format): {format_name}")
        print(f"   子類型 (Subtype): {subtype}")
        print(f"   聲道數 (Channels): {n_channels} ({'Mono' if n_channels == 1 else 'Stereo' if n_channels == 2 else f'{n_channels} channels'})")
        print(f"   採樣率 (Sample Rate): {framerate} Hz ({framerate/1000:.1f} kHz)")
        print(f"   位元深度 (Bit Depth): {bit_depth} bits")
        print(f"   資料類型 (Data Type): {data_type}")
        print(f"   總幀數 (Frames): {n_frames:,}")
        print(f"   時長 (Duration): {duration:.2f} 秒")
        
        # 資料分析
        print(f"\n📈 資料分析:")
        print(f"   NumPy dtype: {data.dtype}")
        print(f"   資料形狀 (Shape): {data.shape}")
        
        if len(data.shape) == 1:
            print(f"   聲道配置: Mono")
        else:
            print(f"   聲道配置: {data.shape[1]} channels")
        
        # 統計資訊
        print(f"\n📉 音訊統計:")
        print(f"   最小值: {np.min(data):.6f}")
        print(f"   最大值: {np.max(data):.6f}")
        print(f"   平均值: {np.mean(data):.6f}")
        print(f"   標準差: {np.std(data):.6f}")
        
        # 檔案大小
        file_size = os.path.getsize(file_path)
        print(f"\n💾 檔案資訊:")
        print(f"   檔案大小: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        # 檢查是否符合目標格式
        print(f"\n✅ 格式檢查:")
        target_checks = {
            "16 kHz 採樣率": framerate == 16000,
            "16-bit 深度": bit_depth == 16,
            "Signed Integer": subtype == 'PCM_16',
            "單聲道": n_channels == 1
        }
        
        for check_name, passed in target_checks.items():
            status = "✓" if passed else "✗"
            print(f"   {status} {check_name}")
        
        # 如果不符合目標格式，給出建議
        if not all(target_checks.values()):
            print(f"\n💡 轉換建議:")
            if framerate != 16000:
                print(f"   • 需要重新採樣到 16 kHz (目前: {framerate} Hz)")
            if bit_depth != 16 or subtype != 'PCM_16':
                print(f"   • 需要轉換為 16-bit signed integer (目前: {data_type})")
            if n_channels != 1:
                print(f"   • 需要轉換為單聲道 (目前: {n_channels} channels)")
            
            print(f"\n   可使用 SoX 指令:")
            print(f"   sox input.wav -r 16000 -b 16 -c 1 output.wav")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
        print(f"\n嘗試使用備用方法...")
        
        # 備用方法：直接讀取 WAV 標頭
        try:
            with open(file_path, 'rb') as f:
                # 讀取 RIFF 標頭
                riff = f.read(4)
                if riff != b'RIFF':
                    print(f"   ❌ 不是有效的 WAV 檔案")
                    return
                
                file_size = struct.unpack('<I', f.read(4))[0]
                wave_tag = f.read(4)
                
                # 讀取 fmt chunk
                fmt_tag = f.read(4)
                fmt_size = struct.unpack('<I', f.read(4))[0]
                audio_format = struct.unpack('<H', f.read(2))[0]
                num_channels = struct.unpack('<H', f.read(2))[0]
                sample_rate = struct.unpack('<I', f.read(4))[0]
                byte_rate = struct.unpack('<I', f.read(4))[0]
                block_align = struct.unpack('<H', f.read(2))[0]
                bits_per_sample = struct.unpack('<H', f.read(2))[0]
                
                format_names = {
                    1: 'PCM',
                    3: 'IEEE Float',
                    6: 'A-law',
                    7: 'μ-law'
                }
                
                print(f"\n📊 WAV 標頭資訊:")
                print(f"   音訊格式: {format_names.get(audio_format, f'Unknown ({audio_format})')}")
                print(f"   聲道數: {num_channels}")
                print(f"   採樣率: {sample_rate} Hz")
                print(f"   位元深度: {bits_per_sample} bits")
                print(f"   Byte Rate: {byte_rate}")
                print(f"   Block Align: {block_align}")
                
        except Exception as e2:
            print(f"   ❌ 備用方法也失敗: {e2}")

# 使用方式
if __name__ == "__main__":
    # 先檢查是否安裝 soundfile
    try:
        import soundfile as sf
    except ImportError:
        print("❌ 需要安裝 soundfile 套件")
        print("請執行: pip install soundfile")
        exit(1)
    
    # 方式 1: 直接指定檔案路徑
    file_path = input("請輸入 WAV 檔案路徑: ").strip().strip('"').strip("'")
    check_wav_format(file_path)
    
    # # 方式 2: 批次檢查資料夾中的檔案
    # check_multiple = input("\n是否要檢查整個資料夾? (y/n): ").lower()
    # if check_multiple == 'y':
    #     folder_path = input("請輸入資料夾路徑: ").strip().strip('"').strip("'")
    #     if os.path.exists(folder_path):
    #         wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    #         print(f"\n找到 {len(wav_files)} 個 WAV 檔案\n")
    #         for file in wav_files:
    #             check_wav_format(os.path.join(folder_path, file))
    #             print("\n")
    #     else:
    #         print(f"❌ 資料夾不存在: {folder_path}")

# import os
# import librosa
# import soundfile as sf
# from tqdm import tqdm

# TARGET_SR = 22050  # 目標採樣率

# def resample_folder(folder_path):
#     """將資料夾中的所有音訊重新採樣到目標採樣率"""
#     files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
#     print(f"📁 處理 {folder_path}...")
#     for file in tqdm(files):
#         file_path = os.path.join(folder_path, file)
        
#         # 載入音訊
#         y, sr = librosa.load(file_path, sr=None)
        
#         # 如果採樣率不同，重新採樣
#         if sr != TARGET_SR:
#             y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            
#             # 覆蓋原檔案
#             sf.write(file_path, y_resampled, TARGET_SR, subtype='PCM_16')
#             print(f"   ✅ {file}: {sr} Hz → {TARGET_SR} Hz")

# # 處理所有噪音資料夾
# resample_folder("background_noises")
# resample_folder("short_noises")
# resample_folder("rir")

# print("✅ 所有噪音檔案已統一採樣率！")