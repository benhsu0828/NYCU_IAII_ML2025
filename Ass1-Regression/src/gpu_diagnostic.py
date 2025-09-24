#!/usr/bin/env python3
"""
GPU 和 CUDA 診斷工具
幫助診斷為什麼 TensorFlow 使用 CPU 而不是 GPU
"""

def check_nvidia_gpu():
    """檢查 NVIDIA GPU 和驅動"""
    print("🔍 檢查 NVIDIA GPU 和驅動...")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 和驅動正常")
            print("GPU 資訊:")
            # 只顯示前幾行重要資訊
            lines = result.stdout.split('\n')[:15]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("❌ nvidia-smi 執行失敗")
            return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi 執行超時")
        return False
    except FileNotFoundError:
        print("❌ 找不到 nvidia-smi 指令")
        print("💡 可能原因:")
        print("   1. 未安裝 NVIDIA 驅動")
        print("   2. 系統環境變數未設定")
        print("   3. 沒有 NVIDIA GPU")
        return False
    except Exception as e:
        print(f"❌ 檢查 GPU 時發生錯誤: {e}")
        return False

def check_cuda_installation():
    """檢查 CUDA 安裝"""
    print("\n🔍 檢查 CUDA 安裝...")
    
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA 編譯器 (nvcc) 可用")
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if version_line:
                print(f"   版本: {version_line[0].strip()}")
            return True
        else:
            print("❌ CUDA 編譯器 (nvcc) 不可用")
            return False
    except FileNotFoundError:
        print("❌ 找不到 nvcc 指令")
        print("💡 可能原因:")
        print("   1. 未安裝 CUDA Toolkit")
        print("   2. CUDA 未加入 PATH 環境變數")
        return False
    except Exception as e:
        print(f"❌ 檢查 CUDA 時發生錯誤: {e}")
        return False

def check_tensorflow():
    """檢查 TensorFlow 安裝和 GPU 支援"""
    print("\n🔍 檢查 TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 已安裝，版本: {tf.__version__}")
        
        # 檢查編譯時是否包含 CUDA 支援
        is_cuda_built = tf.test.is_built_with_cuda()
        print(f"   編譯時是否包含 CUDA: {is_cuda_built}")
        
        if not is_cuda_built:
            print("❌ TensorFlow 未包含 CUDA 支援")
            print("💡 解決方法:")
            print("   重新安裝 GPU 版本:")
            print("   pip uninstall tensorflow")
            print("   pip install tensorflow[and-cuda]")
            return False
        
        # 檢查建置資訊
        build_info = tf.sysconfig.get_build_info()
        print(f"   編譯時 CUDA 版本: {build_info.get('cuda_version', 'N/A')}")
        print(f"   編譯時 cuDNN 版本: {build_info.get('cudnn_version', 'N/A')}")
        
        # 檢查可用設備
        physical_devices = tf.config.list_physical_devices()
        print("   可用設備:")
        for device in physical_devices:
            print(f"     {device}")
        
        # 專門檢查 GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   🚀 檢測到 {len(gpus)} 個 GPU")
            
            # 測試 GPU 計算
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                    c = tf.matmul(a, b)
                print("   ✅ GPU 計算測試成功")
                print(f"   測試結果: {c.numpy()}")
                return True
                
            except Exception as e:
                print(f"   ❌ GPU 計算測試失敗: {e}")
                return False
        else:
            print("   ❌ 未檢測到 GPU")
            return False
            
    except ImportError:
        print("❌ TensorFlow 未安裝")
        print("💡 安裝指令:")
        print("   CPU 版本: pip install tensorflow")
        print("   GPU 版本: pip install tensorflow[and-cuda]")
        return False
    except Exception as e:
        print(f"❌ 檢查 TensorFlow 時發生錯誤: {e}")
        return False

def check_environment_variables():
    """檢查相關環境變數"""
    print("\n🔍 檢查環境變數...")
    
    import os
    
    important_vars = [
        'CUDA_PATH',
        'CUDA_HOME',
        'LD_LIBRARY_PATH',
        'PATH'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            if var == 'PATH':
                # PATH 太長，只顯示包含 cuda 的部分
                cuda_paths = [p for p in value.split(os.pathsep) if 'cuda' in p.lower()]
                if cuda_paths:
                    print(f"   {var} (CUDA 相關): {cuda_paths}")
                else:
                    print(f"   {var}: 未包含 CUDA 路徑")
            else:
                print(f"   {var}: {value}")
        else:
            print(f"   {var}: 未設定")

def provide_solutions():
    """提供解決方案"""
    print("\n🔧 常見問題解決方案:")
    
    print("\n1. 如果顯示 'TensorFlow 未包含 CUDA 支援':")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow[and-cuda]")
    
    print("\n2. 如果 nvidia-smi 不可用:")
    print("   - 安裝最新的 NVIDIA 驅動程式")
    print("   - 重新啟動電腦")
    
    print("\n3. 如果 CUDA 版本不匹配:")
    print("   - 檢查 TensorFlow 文件的 CUDA 版本需求")
    print("   - 安裝對應版本的 CUDA Toolkit")
    
    print("\n4. 如果環境變數問題:")
    print("   - 確認 CUDA 安裝路徑加入 PATH")
    print("   - Windows: 通常在 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin")
    print("   - Linux: 通常在 /usr/local/cuda/bin")

def main():
    """主診斷流程"""
    print("🏠 GPU 和 CUDA 診斷工具")
    print("=" * 50)
    
    # 逐步檢查
    gpu_ok = check_nvidia_gpu()
    cuda_ok = check_cuda_installation()
    tf_ok = check_tensorflow()
    
    check_environment_variables()
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 診斷結果總結:")
    print(f"   NVIDIA GPU: {'✅' if gpu_ok else '❌'}")
    print(f"   CUDA 安裝: {'✅' if cuda_ok else '❌'}")
    print(f"   TensorFlow GPU: {'✅' if tf_ok else '❌'}")
    
    if gpu_ok and cuda_ok and tf_ok:
        print("\n🎉 所有檢查通過！TensorFlow 應該可以使用 GPU")
    else:
        print("\n⚠️ 發現問題，請參考解決方案")
        provide_solutions()

if __name__ == "__main__":
    main()
