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
        print(f"{name:<12} {getattr(mod, '__version__', '?')}")
        if name == "onnxruntime":
            print(f"  ort session  {ort_cuda_session_providers(mod)}")
        if name == "tensorrt":
            mod.Builder(mod.Logger(mod.Logger.ERROR))  # loads libnvinfer and initialises CUDA
            print("  trt builder  ok")


def ort_cuda_session_providers(ort) -> list[str]:
    """Providers a real session ends up with. get_available_providers() only lists what was
    compiled in; if the CUDA libraries fail to load, a session silently falls back to CPU."""
    from onnx import TensorProto, helper

    # CUDA 13 pip wheels put their .so files in site-packages/nvidia/cu13/lib, which is not on the
    # loader path. `import torch` loads the libraries torch needs, but not libcurand, which the ORT
    # CUDA provider also needs. Without this call the session silently runs on CPU.
    # Ultralytics makes the same call before creating its ONNX session (nn/backends/onnx.py).
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    io = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in ("x", "y")]
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "t", io[:1], io[1:])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return session.get_providers()  # want CUDAExecutionProvider first


if __name__ == "__main__":
    main()
