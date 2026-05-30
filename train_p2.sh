#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_p2_${TIMESTAMP}.log"

echo "=========================================="
echo "Phase 3: YOLOv8s + P2 (小目标检测头)"
echo "  - 架构: yolov8s-p2.yaml"
echo "  - 权重: yolov8s.pt (部分加载)"
echo "  - imgsz: 1280, batch: 8 (P2 显存高)"
echo "  - 日志: ${LOG_FILE}"
echo "=========================================="

python -u train_p2.py 2>&1 | tee "${LOG_FILE}"

echo "Phase 3 训练完成! 日志: ${LOG_FILE}"
