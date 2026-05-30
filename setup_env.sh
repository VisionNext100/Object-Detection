#!/bin/bash
# ============================================================
# AutoDL 环境一键配置脚本 (RTX 4090 / PyTorch 2.3.0 / py3.12 / cu12.1)
#
# 用法 (在 baseline 目录下):
#   cd /root/autodl-tmp/baseline
#   bash setup_env.sh
#
# 关键点:
#   - 用 `pip install -e .` 以"可编辑模式"安装本地 ultralytics 源码,
#     这样后续 Phase 2.5/3 修改 ultralytics/ 里的 loss.py、build.py 等
#     才能立即生效, 无需重装。
#   - torch/torchvision 已在镜像里, 不重装 (用 --no-build-isolation 思路:
#     直接让 pip 判定已满足)。
#   - numpy 锁 1.26.4, 避免被升级到 2.x 引发 opencv / torch 不兼容。
# ============================================================
set -e

cd "$(dirname "$0")"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "=========================================="

# 1. 换 AutoDL 学术加速 (可选, 若 pip 慢可取消注释)
# source /etc/network_turbo

# 2. 以可编辑模式安装本地 ultralytics (拉取其依赖, 但不动已装好的 torch)
echo "[1/3] 安装本地 ultralytics (可编辑模式)..."
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 锁定 numpy 版本 + 补装后续阶段要用的小依赖
echo "[2/3] 锁定 numpy 并补装依赖..."
pip install "numpy==1.26.4" tqdm nvitop ensemble-boxes -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 验证环境
echo "[3/3] 验证 torch + CUDA + ultralytics ..."
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("cuda 可用    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
    print("显存(GB)     :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
import numpy, cv2
print("numpy        :", numpy.__version__)
print("opencv       :", cv2.__version__)
from ultralytics import YOLO
import ultralytics
print("ultralytics  :", ultralytics.__version__, "(本地源码)")
print("\n✅ 环境就绪")
PY
