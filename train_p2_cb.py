"""
Phase 4 训练脚本: P2 + imgsz=1536 + 类别均衡加权采样 (长尾优化)

在 Phase 3 (v8s_p2_1536_phase3, OJ=8883) 基础上叠加:
  - 类别均衡加权采样 (Class-Balanced Sampler)
    含稀有类目标的图像被过采样, 直接缓解 45 类的长尾不均衡
    (作业明确点名的 "多类别不均衡" 问题)。
  - 通过环境变量 YOLO_CB_SAMPLER=1 开启 (在 ultralytics/data/build.py 中实现);
    YOLO_CB_POWER 控制力度: 0=不加权, 0.5=温和(默认), 1=完全反频率(激进)。

⚠️ 推理时仍用 imgsz=1536:
   python infer.py --model runs/detect/v8s_p2_1536_cb_phase4/weights/best.pt \
       --imgsz 1536 --save_json infer_res/predictions_phase4.json
"""
import os
# 必须在 import / 训练前设置, build_dataloader 运行时读取
os.environ["YOLO_CB_SAMPLER"] = "1"     # 开启类别均衡加权采样
os.environ["YOLO_CB_POWER"] = "0.5"     # 温和力度; 想更激进可调 0.7~1.0

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8s-p2.yaml').load('./pretrain_models/yolov8s.pt')

    model.train(
        # === 数据与基础 (与 Phase 3 一致) ===
        data='data.yaml',
        imgsz=1536,
        epochs=300,
        batch=6,                 # 若 OOM 降到 4
        workers=8,
        device=0,
        cache='ram',
        amp=True,

        # === 优化器 (与 Phase 1/2/3 一致) ===
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,
        cos_lr=True,

        # === 训练策略 ===
        patience=50,
        close_mosaic=20,

        # === 数据增强 (与 Phase 3 一致) ===
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.5,

        # === 日志与保存 ===
        name='v8s_p2_1536_cb_phase4',
        exist_ok=True,
        save_period=-1,
        verbose=True,
        seed=42,
        plots=True,
    )
