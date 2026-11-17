"""Print the GPU/software stack this project runs on.

Run inside the project venv:
    python scripts/check_env.py

Its output is copied into the README's "Environment" section, so every
version number there comes from this machine.
"""
import platform
import sys


def main() -> None:
    print(f"python       {sys.version.split()[0]} ({platform.platform()})")

    import torch

    print(f"torch        {torch.__version__}")
    print(f"torch CUDA   {torch.version.cuda}")  # None = CPU-only build
    print(f"cuDNN        {torch.backends.cudnn.version()}")
    print(f"CUDA avail   {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch - fix this before training.")

    props = torch.cuda.get_device_properties(0)
    print(
        f"GPU          {props.name}, {props.total_memory / 2**30:.1f} GiB, "
        f"compute capability {props.major}.{props.minor}"
    )

    # is_available() only says a driver was found. Running a real kernel
    # proves the wheel was built for this GPU architecture.
    x = torch.randn(4096, 4096, device="cuda")
    y = (x @ x).abs().mean().item()
    torch.cuda.synchronize()
    print(f"matmul       ok (mean |x@x| = {y:.2f})")

    for name in ("ultralytics", "onnx", "onnxruntime", "tensorrt"):
        try:
            mod = __import__(name)
        except ImportError:
            print(f"{name:<12} not installed")
            continue
        line = f"{name:<12} {getattr(mod, '__version__', '?')}"
        if name == "onnxruntime":
            line += f"  providers={mod.get_available_providers()}"
        print(line)


if __name__ == "__main__":
    main()
