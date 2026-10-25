import sys
import time
import torch

def test_gpu_environment():
    print("=" * 60)
    print("           MASK R-CNN GPU ENVIRONMENT TEST")
    print("=" * 60)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[!] WARNING: CUDA is NOT available to PyTorch!")
        print("    PyTorch was likely installed without CUDA support.")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"GPU Count         : {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"GPU [{i}]           : {props.name} ({vram_gb:.2f} GB VRAM, Compute {props.major}.{props.minor})")

    # Quick Tensor operation test on GPU
    print("\n[*] Testing CUDA Tensor operations & Warm-up...")
    x = torch.randn((1000, 1000), device="cuda")
    start = time.perf_counter()
    for _ in range(100):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"[+] CUDA Matrix Multiplication (100 passes): {elapsed:.2f} ms")
    print("[+] GPU Acceleration is working properly!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_gpu_environment()
