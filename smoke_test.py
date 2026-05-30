"""
冒烟测试: 正式长训练前的快速验证 (约 2-4 分钟)。

目的:
  - 用真实的 imgsz=1280 / batch=16 配置, 但只取 5% 数据 (fraction=0.05),
    快速验证: ① 数据管线正常 ② GPU 可用 ③ 这个 batch/分辨率不会 OOM。
  - 通过后再放心执行 `bash train.sh` 启动 300 epoch 正式训练。

用法 (在 baseline 目录下):
  python smoke_test.py
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./pretrain_models/yolov8s.pt')

    model.train(
        data='data.yaml',
        imgsz=1280,              # 与正式训练一致, 用来真实探测显存
        epochs=3,
        batch=16,                # 与正式训练一致; 若这里 OOM, 把 train.py 的 batch 也一起降
        fraction=0.05,           # 只用 5% 训练数据, 几分钟跑完
        workers=8,
        device=0,
        cache=False,             # 冒烟测试不缓存, 省启动时间
        amp=True,
        optimizer='AdamW',
        lr0=1e-3,
        name='smoke_test',
        exist_ok=True,
        plots=False,
        verbose=True,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("✅ 冒烟测试通过: 数据管线 + GPU + 显存(batch=16@1280) 均正常")
    print("   现在可以启动正式训练:  bash train.sh   (或  python train.py)")
    print("=" * 60)
