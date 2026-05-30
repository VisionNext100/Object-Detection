"""
Phase 3 训练脚本: YOLOv8s-P2 + imgsz=1536 (提分辨率, 主攻极小目标)

为什么提分辨率:
  数据实测: 目标中位边长 ~32px@2048, 缩到 1280 只剩 ~20px, 是当前最大瓶颈。
  imgsz 1280 -> 1536 后, 同一目标变成 ~24px, P2 头(stride=4)在 1536 下
  最小可检 ~6px, 对这种"全是极小目标"的数据集预期收益最大。

相比 Phase 2 (v8s_p2_1280_phase2) 的唯一改动:
  - imgsz 1280 -> 1536
  - batch 8 -> 6 (分辨率升高, 显存占用约 1.44x, 防 OOM; 若仍 OOM 降到 4)
  其余超参完全一致, 保证可比性。

⚠️ 推理时务必用 imgsz=1536 (与训练一致):
   python infer.py --model runs/detect/v8s_p2_1536_phase3/weights/best.pt \
       --imgsz 1536 --save_json infer_res/predictions_phase3.json
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8s-p2.yaml').load('./pretrain_models/yolov8s.pt')

    model.train(
        # === 数据与基础 ===
        data='data.yaml',
        imgsz=1536,              # Phase2 的 1280 -> 1536, 核心改动
        epochs=300,
        batch=6,                 # 1536 显存更高, 从 8 降到 6; 若首个 epoch OOM 降到 4
        workers=8,
        device=0,
        cache='ram',
        amp=True,

        # === 优化器 (与 Phase 1/2 一致) ===
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,
        cos_lr=True,

        # === 训练策略 (与 Phase 2 一致) ===
        patience=50,
        close_mosaic=20,

        # === 数据增强 (与 Phase 2 一致) ===
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,              # ⚠️ 交通标志不可上下翻
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.5,

        # === 日志与保存 ===
        name='v8s_p2_1536_phase3',
        exist_ok=True,
        save_period=-1,
        verbose=True,
        seed=42,
        plots=True,
    )
