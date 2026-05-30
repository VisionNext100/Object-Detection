#!/bin/bash
set -e

cd "$(dirname "$0")"

# 创建日志目录
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "=========================================="
echo "Phase 1: YOLOv8s + imgsz=1280 + 300 epochs"
echo "Status log: ${LOG_FILE}"
echo "=========================================="

{
  echo "Training started at $(date)"
  echo "Command: python -u train.py"
} > "${LOG_FILE}"

# 训练输出只打印到终端, 不保存完整日志
set +e
python -u train.py
STATUS=$?
set -e

{
  echo "Training finished at $(date)"
  echo "Status: ${STATUS}"
} >> "${LOG_FILE}"

if [ "${STATUS}" -eq 0 ]; then
  echo "Training finished! Status log saved to ${LOG_FILE}"
else
  echo "Training failed with status ${STATUS}. Status log saved to ${LOG_FILE}"
fi

exit "${STATUS}"
