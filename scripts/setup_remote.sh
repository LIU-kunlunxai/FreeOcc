#!/bin/bash
# ===================================================================
# FreeOcc 远程服务器一键安装脚本
# 自动适配 CUDA 版本 + GPU 架构
#
# 用法: bash setup_remote.sh [FREEOC_DIR]
# ===================================================================
set -euo pipefail

FREEOC_DIR="${1:-$HOME/FreeOcc}"

echo "=================================="
echo "FreeOcc 远程环境安装"
echo "=================================="
echo "FreeOcc 目录: $FREEOC_DIR"

# ──── 自动检测 CUDA ────
if [ -n "${CUDA_HOME:-}" ]; then
    CUDA_HOME="$CUDA_HOME"
elif [ -d /usr/local/cuda-12.8 ]; then
    CUDA_HOME=/usr/local/cuda-12.8
elif [ -d /usr/local/cuda-12.6 ]; then
    CUDA_HOME=/usr/local/cuda-12.6
elif [ -d /usr/local/cuda-12.4 ]; then
    CUDA_HOME=/usr/local/cuda-12.4
elif [ -d /usr/local/cuda-12.1 ]; then
    CUDA_HOME=/usr/local/cuda-12.1
elif [ -d /usr/local/cuda-11.8 ]; then
    CUDA_HOME=/usr/local/cuda-11.8
elif [ -d /usr/local/cuda ]; then
    CUDA_HOME=/usr/local/cuda
else
    CUDA_HOME=""
fi

if [ -z "$CUDA_HOME" ]; then
    echo "错误: 未找到 CUDA, 请设置 CUDA_HOME"
    echo "  export CUDA_HOME=/path/to/cuda"
    exit 1
fi

CUDA_VER=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1 | tr -d '.')
echo "CUDA_HOME: $CUDA_HOME (version: ${CUDA_VER:0:2}.${CUDA_VER:2})"

# ──── 自动检测 GPU 架构 ────
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "$GPU_ARCH" ]; then
    echo "警告: 无法检测 GPU, 默认使用 sm_80 (A100)"
    GPU_ARCH="80"
fi
echo "GPU 架构: sm_$GPU_ARCH"

# ──── 根据 CUDA 版本选 PyTorch ────
# CUDA 12.x → PyTorch 2.9 cu128, CUDA 11.8 → PyTorch 2.5 cu118
CUDA_MAJOR="${CUDA_VER:0:2}"
if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_VERSION="2.9.0"
    TORCH_CUDA_TAG="cu128"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    PYTORCH3D_CUDA=0  # 不编译 CUDA 扩展
else
    TORCH_VERSION="2.5.0"
    TORCH_CUDA_TAG="cu118"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    PYTORCH3D_CUDA=0
fi

echo "PyTorch: $TORCH_VERSION ($TORCH_CUDA_TAG)"
echo ""

# ──── Step 1: 创建 conda 环境 ────
echo "[1/7] 创建 conda 环境 freeocc ..."
if conda env list 2>/dev/null | grep -q "^freeocc "; then
    echo "  环境 freeocc 已存在，跳过创建"
else
    conda env create -f "$FREEOC_DIR/environment.yaml" -y
fi

eval "$(conda shell.bash hook)"
conda activate freeocc
PYTHON="$(which python)"
echo "  Python: $PYTHON ($($PYTHON --version))"

# ──── Step 2: 安装 PyTorch ────
echo "[2/7] 安装 PyTorch $TORCH_VERSION + $TORCH_CUDA_TAG ..."
$PYTHON -c "import torch; print('torch', torch.__version__)" 2>/dev/null && echo "  PyTorch 已安装" || {
    pip install --index-url "$TORCH_INDEX" \
        "torch==$TORCH_VERSION" "torchvision==0.$(echo $TORCH_VERSION | cut -d'.' -f2).0" "torchaudio==$TORCH_VERSION"
}
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用!'; print(f'  torch {torch.__version__} CUDA {torch.version.cuda} OK')"

# ──── Step 3: 安装 PyTorch3D + torch-scatter ────
echo "[3/7] 安装 PyTorch3D + torch-scatter ..."
$PYTHON -c "import pytorch3d" 2>/dev/null && echo "  PyTorch3D 已安装" || {
    pip install fvcore iopath
    PYTORCH3D_NO_EXTENSION=1 \
    pip install --no-build-isolation --no-deps \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable"
}

SCATTER_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"
$PYTHON -c "import torch_scatter" 2>/dev/null && echo "  torch-scatter 已安装" || {
    pip install torch-scatter -f "$SCATTER_URL"
}

# ──── Step 4: 安装 Python 运行时依赖 ────
echo "[4/7] 安装 Python 运行时依赖 ..."
pip install \
    hydra-core omegaconf tqdm termcolor ipdb \
    kornia faiss-cpu einops plyfile pyliblzfse \
    open3d opencv-python==4.9.0 opencv-python-headless==4.9.0 \
    glfw imgviz PyGLM PyOpenGL PyOpenGL-accelerate \
    plotly kaleido evo torchmetrics \
    ftfy==6.2.0 regex==2023.8.8 fsspec "transformers>=4.37.2,<4.38" \
    openpyxl==3.1.2 huggingface_hub==0.23.0 safetensors==0.4.3 \
    timm==0.6.7 pycocotools easydict torchtyping

# ──── Step 5: 安装 OpenMMLab (Trident 需要) ────
echo "[5/7] 安装 OpenMMLab ..."
$PYTHON -c "import mmseg" 2>/dev/null && echo "  mmseg 已安装" || {
    pip install -U openmim
    pip install -U "mmengine>=0.10.7"
    mim install "mmcv==2.1.0" 2>/dev/null || {
        echo "  mim 安装 mmcv 失败，尝试从源码编译..."
        pip install "mmcv==2.1.0" --no-build-isolation
    }
    pip install "mmsegmentation>=1.2.2"
}

# ──── Step 6: 编译 CUDA 扩展 ────
echo "[6/7] 编译 CUDA 扩展 (arch=sm_${GPU_ARCH}) ..."
cd "$FREEOC_DIR"
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export CUDA_HOME="$CUDA_HOME"

for ext in droid_backends lietorch simple_knn; do
    $PYTHON -c "import ${ext}" 2>/dev/null && echo "  $ext 已安装" || {
        echo "  编译 $ext ..."
        PKG=$ext python setup.py install
    }
done

$PYTHON -c "import diff_gaussian_rasterization" 2>/dev/null && echo "  diff_gaussian_rasterization 已安装" || {
    echo "  编译 diff_gaussian_rasterization ..."
    PKG=diff_gaussian_rasterization python setup.py install
}

$PYTHON -c "from src.gs2occ.localagg_prob.local_aggregate_prob import LocalAggregator" 2>/dev/null && echo "  LocalAggregator 已安装" || {
    echo "  编译 LocalAggregator ..."
    pushd src/gs2occ/localagg_prob
    python setup.py build_ext --inplace
    popd
}

# ──── Step 7: 验证 ────
echo "[7/7] 验证安装 ..."
$PYTHON << PYEOF
import torch
print(f"  torch {torch.__version__}  CUDA {torch.version.cuda}  available={torch.cuda.is_available()}")
import droid_backends;               print("  droid_backends OK")
import lietorch;                     print("  lietorch OK")
from simple_knn import _C;           print("  simple_knn OK")
import diff_gaussian_rasterization;  print("  diff_gaussian_rasterization OK")
from src.gs2occ.localagg_prob.local_aggregate_prob import LocalAggregator
print("  LocalAggregator OK")
import pytorch3d;                    print("  pytorch3d OK")
import open3d;                       print("  open3d OK")
print("  全部验证通过!")
PYEOF

echo ""
echo "=================================="
echo "安装完成!"
echo "=================================="
echo ""
echo "运行 FreeOcc:"
echo ""
echo "  conda activate freeocc"
echo "  cd $FREEOC_DIR"
echo ""
echo "  python run.py \\"
echo "    mode=rgbd \\"
echo "    use_gt_poses=True \\"
echo "    data.input_folder=/path/to/freeocc_data \\"
echo "    data.cam.H=480 data.cam.W=640 \\"
echo "    data.cam.H_out=480 data.cam.W_out=640 \\"
echo "    data.cam.fx=386.502 data.cam.fy=385.938 \\"
echo "    data.cam.cx=321.497 data.cam.cy=241.840 \\"
echo "    data.png_depth_scale=1000.0 \\"
echo "    mapping.online_opt.filter.bin_th=0.03 \\"
echo "    mapping.loss.supervise_with_prior=False"
