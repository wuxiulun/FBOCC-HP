import torch
import gc

def print_device_info():
    """
    打印可用的计算设备（CPU 或 GPU）的详细信息。
    """
    print(f"Torch 版本: {torch.__version__}")
    print(f"Torch 构建配置：\n{torch.__config__.show()}")  # 更易读的配置信息

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA 可用！GPU 数量：{num_gpus}")
        print(f"CUDA 版本：{torch.version.cuda}")
        print(f"cuDNN 版本：{torch.backends.cudnn.version()}")

        for i in range(num_gpus):
            print(f"\n--- GPU {i} ---")
            print(f"  设备名称：{torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  计算能力：{props.major}.{props.minor}")
            print(f"  总显存：{props.total_memory / (1024**3):.2f} GB")  # 转换为 GB
            print(f"  多处理器数量：{props.multi_processor_count}")
            # 如果需要，可以在此处添加更多属性，例如时钟频率、显存时钟频率等。

        # 如果有多个 GPU，显示当前设备（如果不是 0）
        current_device = torch.cuda.current_device()
        if num_gpus > 1:
            print(f"\n当前设备（索引）：{current_device}")


    else:
        print("CUDA 不可用。正在使用 CPU。")
        # 如果需要，您可以在此处添加有关 CPU 的信息。
        # （例如，使用 platform、multiprocessing 或 psutil 库，但它不如查询 CUDA 设备标准化。）

    # 清除 PyTorch 的 CUDA 内存缓存（仅当 CUDA 可用时才相关）
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        print("\nPyTorch CUDA 内存缓存已清除。")
    else:
        print("\n（没有 CUDA 设备，因此没有 CUDA 缓存可清除。）")

if __name__ == '__main__':
    print_device_info()