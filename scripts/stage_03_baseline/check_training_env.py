"""
阶段三：训练环境检查脚本
文件名：check_training_env.py

作用：
1. 检查当前 Python 版本。
2. 检查是否已经安装 PyTorch。
3. 检查 PyTorch 是否能使用 NVIDIA CUDA GPU。
4. 输出 GPU 名称，帮助后续决定训练参数。

运行方法：
在 PowerShell 中进入项目目录后执行：

    python check_training_env.py

这个脚本不会修改任何文件，也不会开始训练模型。
"""

import platform
import sys


def main() -> None:
    """打印当前训练环境的关键信息。"""

    print("=" * 60)
    print("阶段三：训练环境检查")
    print("=" * 60)

    # sys.version 会输出完整的 Python 版本和编译信息。
    print(f"Python 版本：{sys.version}")
    print(f"操作系统：{platform.platform()}")

    print("\n检查 PyTorch...")

    try:
        # 把 import 放进 try 中：
        # 如果电脑还没有安装 torch，程序不会直接崩溃，
        # 而是给出更容易理解的提示。
        import torch
    except ImportError:
        print("结果：当前 Python 环境中没有安装 PyTorch。")
        print("下一步：把这段输出发给我，我会根据你的电脑情况指导安装。")
        return

    print(f"PyTorch 版本：{torch.__version__}")

    # torch.cuda.is_available() 返回 True / False。
    # True 表示当前 PyTorch 可以调用 NVIDIA CUDA GPU。
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用：{cuda_available}")

    if cuda_available:
        # 当前使用的 GPU 编号通常是 0。
        device_index = torch.cuda.current_device()

        # 获取显卡名称，例如 NVIDIA GeForce RTX 4060 Laptop GPU。
        gpu_name = torch.cuda.get_device_name(device_index)

        print(f"当前 CUDA 设备编号：{device_index}")
        print(f"GPU 名称：{gpu_name}")

        # 显存总量以字节为单位，这里换算成 GB，方便阅读。
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        print(f"GPU 总显存：{total_memory_gb:.2f} GB")

        print("\n环境结论：可以优先使用 GPU 训练。")
    else:
        print("\n环境结论：当前 PyTorch 无法使用 CUDA GPU。")
        print("这不一定代表电脑没有独立显卡，也可能是 PyTorch 安装版本不对。")
        print("把输出发给我，我们再判断。")


# 只有直接执行本文件时，才调用 main()。
# 如果以后这个文件被其他 Python 文件 import，则不会自动执行检查。
if __name__ == "__main__":
    main()
