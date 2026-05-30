"""
Phase 3 训练脚本: YOLOv8s + P2 小目标检测头

为什么 P2:
  数据分析显示 99.6% 的目标占图像面积 <1% (原图 2048, bbox 中位 ~32px)
  默认 YOLOv8 从 P3 (stride=8) 开始检测, 最小能检 ~16px, 已在分辨率极限.
  加 P2 (stride=4) 后, 最小能检 ~8px, 对极小目标有显著提升.

权重加载逻辑:
  - 架构: yolov8s-p2.yaml (官方内置)
  - 预训练: yolov8s.pt (仍然遵守作业约束)
  - .load() 会智能匹配: backbone 全加载, 新增的 P2 head 层随机初始化

显存说明:
  - P2 特征图 320x320x128 显存占用是 P3+P4+P5 总和的 1.5 倍
  - v8s + P2 @ imgsz=1280 在 3090 24G 上 batch=8 安全, OOM 降到 4
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # YOLO 会自动从 'yolov8s-p2.yaml' 解析出 (架构=yolov8-p2.yaml, scale='s')
    # .load() 加载 yolov8s.pt 中所有形状匹配的权重 (backbone 完全匹配, P2 head 随机初始化)
    model = YOLO('yolov8s-p2.yaml').load('./pretrain_models/yolov8s.pt')

    model.train(
        # === 数据与基础 ===
        data='data.yaml',
        imgsz=1280,
        epochs=300,
        batch=8,                 # ⚠️ P2 比普通 v8s 多吃 ~40% 显存, 4090 24G 用 8; OOM 再降到 4
        workers=8,
        device=0,                # AutoDL 单卡, GPU 编号 0
        cache='ram',
        amp=True,

        # === 优化器 (与 Phase 1 一致, 保持可比性) ===
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
        close_mosaic=15,

        # === 数据增强 (与 Phase 1 一致) ===
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
        copy_paste=0.3,          # 长尾救星, 23 个稀有类靠它

        # === 日志与保存 ===
        # ⚠️ 不要写 project='runs/detect': 会导致 runs/detect/runs/detect 双层路径
        name='v8s_p2_1280_phase3',
        exist_ok=True,
        save_period=-1,
        verbose=True,
        seed=42,                 # 与 Phase 1 同种子, 便于消融对比
        plots=True,
    )
