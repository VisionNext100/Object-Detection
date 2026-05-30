import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # 仍使用 yolov8s 架构 + yolov8s.pt 预训练权重 (作业约束)
    model = YOLO('./pretrain_models/yolov8s.pt')

    model.train(
        # === 数据与基础 ===
        data='data.yaml',
        imgsz=1280,              # 关键: 小目标 (交通标志) 必须用大分辨率
        epochs=300,              # 长训练 + patience 自动早停
        batch=16,                # v8s @ 1280 在 4090 24G AMP 下可跑 16; 若 OOM 降到 12 或 8
        workers=8,
        device=0,                # AutoDL 单卡, GPU 编号 0
        cache='ram',             # 把图像缓存到内存,4090 实例 120G 内存足够,显著加速
        amp=True,

        # === 优化器: AdamW 在小数据集上通常优于 SGD ===
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,
        cos_lr=True,             # 余弦退火

        # === 训练策略 ===
        patience=50,             # 50 轮 mAP 不提升则早停
        close_mosaic=15,         # 最后 15 轮关闭 mosaic, 让模型适应真实分布

        # === 数据增强 (针对交通标志特化) ===
        # ⚠️ 交通标志不可上下翻转 (会改变 p11/i4 等带数字标志的语义)
        # ⚠️ 旋转角度也要小, 避免把"禁止"标志转成奇怪的方向
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
        mixup=0.15,              # 长尾类的救星
        copy_paste=0.3,          # 同上, 把稀有类粘贴到其他图

        # === 日志与保存 ===
        # ⚠️ 不要写 project='runs/detect': ultralytics 会再前缀 RUNS_DIR/detect,
        #    导致 runs/detect/runs/detect 双层路径。留空即保存到 runs/detect/<name>
        name='v8s_1280_phase1',
        exist_ok=True,
        save_period=-1,
        verbose=True,
        seed=42,
        plots=True,
    )
