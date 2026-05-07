import os
import torch
from torch.utils.cpp_extension import CUDA_HOME

PKG = os.environ.get("PKG", "").strip()
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
cuda_home = os.environ.get("CUDA_HOME") or CUDA_HOME or "/usr/local/cuda-12.8"
cuda_lib = os.path.join(cuda_home, "lib64")

def build_droid_backends():
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ROOT = os.path.dirname(os.path.abspath(__file__))
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    nvcc_arch_flags = []
    if archs:
        for a in archs.replace(" ", "").split(";"):
            sm = a.replace(".", "")
            nvcc_arch_flags += [f"-gencode=arch=compute_{sm},code=sm_{sm}"]
    else:
        nvcc_arch_flags = [
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
        ]

    setup(
        name="droid_backends",
        ext_modules=[
            CUDAExtension(
                "droid_backends",
                include_dirs=[os.path.join(ROOT, "thirdparty/eigen")],
                sources=[
                    "src/lib/droid.cpp",
                    "src/lib/droid_kernels.cu",
                    "src/lib/correlation_kernels.cu",
                    "src/lib/altcorr_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3"] + nvcc_arch_flags,
                },
                extra_link_args=[
                    f"-Wl,-rpath,{cuda_lib}",
                    f"-Wl,-rpath,{torch_lib}",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )

def build_lietorch():
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ROOT = os.path.dirname(os.path.abspath(__file__))
    archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    nvcc_arch_flags = []
    if archs:
        for a in archs.replace(" ", "").split(";"):
            sm = a.replace(".", "")
            nvcc_arch_flags += [f"-gencode=arch=compute_{sm},code=sm_{sm}"]
    else:
        nvcc_arch_flags = [
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
        ]

    setup(
        name="lietorch",
        version="0.2",
        description="Lie Groups for PyTorch",
        packages=["lietorch"],
        package_dir={"": "thirdparty/lietorch"},
        ext_modules=[
            CUDAExtension(
                "lietorch_backends",
                include_dirs=[
                    os.path.join(ROOT, "thirdparty/lietorch/lietorch/include"),
                    os.path.join(ROOT, "thirdparty/eigen"),
                ],
                sources=[
                    "thirdparty/lietorch/lietorch/src/lietorch.cpp",
                    "thirdparty/lietorch/lietorch/src/lietorch_gpu.cu",
                    "thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp",
                ],
                extra_link_args=[
                    f"-Wl,-rpath,{cuda_lib}",
                    f"-Wl,-rpath,{torch_lib}",
                ],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )

def build_simple_knn():
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    cxx_compiler_flags = []
    if os.name == "nt":
        cxx_compiler_flags.append("/wd4624")
    setup(
        name="simple_knn",
        ext_modules=[
            CUDAExtension(
                name="simple_knn._C",
                sources=[
                    "thirdparty/simple-knn/spatial.cu",
                    "thirdparty/simple-knn/simple_knn.cu",
                    "thirdparty/simple-knn/ext.cpp",
                ],
                extra_compile_args={
                    "nvcc": ["-include", "cfloat"],
                    "cxx":  cxx_compiler_flags + ["-include", "cfloat"],
                },
                extra_link_args=[
                    f"-Wl,-rpath,{cuda_lib}",
                    f"-Wl,-rpath,{torch_lib}",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )

def build_diff_gauss():
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ROOT = os.path.dirname(os.path.abspath(__file__))
    setup(
        name="diff_gaussian_rasterization",
        packages=["diff_gaussian_rasterization"],
        package_dir={"": "thirdparty/diff-gaussian-rasterization-w-pose"},
        ext_modules=[
            CUDAExtension(
                name="diff_gaussian_rasterization._C",
                sources=[
                    "thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.cu",
                    "thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/forward.cu",
                    "thirdparty/diff-gaussian-rasterization-w-pose/cuda_rasterizer/backward.cu",
                    "thirdparty/diff-gaussian-rasterization-w-pose/rasterize_points.cu",
                    "thirdparty/diff-gaussian-rasterization-w-pose/ext.cpp",
                ],
                extra_compile_args={
                    "nvcc": [
                        "-I"
                        + os.path.join(
                            ROOT,
                            "thirdparty/diff-gaussian-rasterization-w-pose/third_party/glm/",
                        )
                    ]
                },
                extra_link_args=[
                    f"-Wl,-rpath,{cuda_lib}",
                    f"-Wl,-rpath,{torch_lib}",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )

if   PKG == "droid_backends":            build_droid_backends()
elif PKG == "lietorch":                  build_lietorch()
elif PKG == "simple_knn":                build_simple_knn()
elif PKG == "diff_gaussian_rasterization": build_diff_gauss()
else:
    raise SystemExit("Please set PKG to one of: droid_backends | lietorch | simple_knn | diff_gaussian_rasterization")
