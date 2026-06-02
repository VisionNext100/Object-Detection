"""
Phase 8 训练脚本: P2 + imgsz=1792 + 长尾采样 (为融合再造一个强力差异化模型)

动机:
  - 分辨率是本数据集最强杠杆 (1280->1536 单步 +386 OJ)。继续提到 1792,
    目标中位边长在训练分辨率下进一步增大, 极小目标可检性更好。
  - 同时它与已有 1280/1536 模型"分辨率不同 -> 犯不同的错", 是 WBF 融合的
    理想新成员 (融合最吃"又强又不同"的模型)。

配置 = Phase 4 (当前最佳单模型 9085) 的策略, 仅把分辨率升到 1792:
  - 架构 yolov8s-p2, 仍只加载 yolov8s.pt
  - 长尾加权采样 (YOLO_CB_SAMPLER=1) 保留
  - 损失保持默认 CIoU+BCE (Phase 5 证明 WIoU+VFL 会掉分, 不再使用)

⚠️ 显存: 1792 比 1536 多约 36% 显存, batch 从 6 降到 4; 若仍 OOM 降到 2 或 3。
⚠️ 推理务必用 imgsz=1792:
   python infer.py --model runs/detect/v8s_p2_1792_cb_phase8/weights/best.pt \
       --imgsz 1792 --save_json infer_res/predictions_phase8.json
"""
import os
os.environ["YOLO_CB_SAMPLER"] = "1"     # 沿用 Phase 4 长尾加权采样
os.environ["YOLO_CB_POWER"] = "0.5"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8s-p2.yaml').load('./pretrain_models/yolov8s.pt')

    model.train(
        # === 数据与基础 ===
        data='data.yaml',
        imgsz=1792,              # 核心改动: 1536 -> 1792
        epochs=300,
        batch=4,                 # 1792 显存高; 若 OOM 降到 2~3
        workers=8,
        device=0,
        cache='ram',
        amp=True,

        # === 优化器 (与 Phase 1-4 一致) ===
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

        # === 数据增强 (与 Phase 4 一致) ===
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
        name='v8s_p2_1792_cb_phase8',
        exist_ok=True,
        save_period=-1,
        verbose=True,
        seed=42,
        plots=True,
    )
