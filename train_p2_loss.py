"""
Phase 5 训练脚本: P2 + 1536 + 长尾采样 + 损失函数升级 (WIoU + Varifocal)

在 Phase 4 (v8s_p2_1536_cb_phase4, OJ=9085) 基础上叠加:
  - 回归损失 CIoU -> WIoU v3   (作业鼓励的 "新型损失函数"; 对小目标的非单调
    动态聚焦, 减少低质量框的有害梯度)
  - 分类损失 BCE  -> Varifocal (IoU-aware 软标签, 与长尾/密集预测强协同)

实现方式 (改在 ultralytics 源码, 由环境变量开关, 默认关闭不影响既有 Phase):
  - YOLO_IOU_TYPE : ciou(默认) / eiou / wiou   -> ultralytics/utils/loss.py BboxLoss
  - YOLO_CLS_LOSS : bce(默认)  / vfl  / focal  -> ultralytics/utils/loss.py v8DetectionLoss
  - YOLO_CB_*     : 沿用 Phase 4 的类别均衡采样

⚠️ 若本组合反而掉分, 按以下顺序回退试验 (每次只改一个环境变量):
   1) 只升分类: YOLO_IOU_TYPE=ciou, YOLO_CLS_LOSS=vfl
   2) 只升回归: YOLO_IOU_TYPE=eiou, YOLO_CLS_LOSS=bce
   3) 回归换 wiou->eiou (eiou 更温和)

⚠️ 推理仍用 imgsz=1536:
   python infer.py --model runs/detect/v8s_p2_1536_loss_phase5/weights/best.pt \
       --imgsz 1536 --save_json infer_res/predictions_phase5.json
"""
import os
# 必须在训练前设置: 损失/采样器在 model.train() 内部构造时读取这些环境变量
os.environ["YOLO_CB_SAMPLER"] = "1"     # 沿用 Phase 4 长尾加权采样
os.environ["YOLO_CB_POWER"] = "0.5"
os.environ["YOLO_IOU_TYPE"] = "wiou"    # 回归损失: WIoU v3
os.environ["YOLO_CLS_LOSS"] = "vfl"     # 分类损失: Varifocal

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8s-p2.yaml').load('./pretrain_models/yolov8s.pt')

    model.train(
        # === 数据与基础 (与 Phase 4 一致) ===
        data='data.yaml',
        imgsz=1536,
        epochs=300,
        batch=6,                 # 若 OOM 降到 4
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
        name='v8s_p2_1536_loss_phase5',
        exist_ok=True,
        save_period=-1,
        verbose=True,
        seed=42,
        plots=True,
    )
