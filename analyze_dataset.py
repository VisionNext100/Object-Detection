"""
数据集快速诊断脚本。
输出:
  1. 各类别样本数 (找出长尾)
  2. bbox 相对图像的尺寸分布 (确认小目标程度)
  3. 图像分辨率分布
  4. 每图平均目标数

用法: python analyze_dataset.py
"""
import os
from collections import Counter
from pathlib import Path

import cv2
import yaml

# 让脚本无论从哪个目录运行都能找到数据集 (基于脚本所在目录)
BASE_DIR = Path(__file__).resolve().parent


def load_classes(yaml_path="data.yaml"):
    with open(BASE_DIR / yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]


def analyze_split(split_dir, class_names):
    image_dir = BASE_DIR / split_dir / "images"
    label_dir = BASE_DIR / split_dir / "labels"

    class_counter = Counter()
    bbox_sizes = []          # (rel_w, rel_h)
    img_sizes = []           # (W, H)
    objs_per_img = []

    label_files = sorted(label_dir.glob("*.txt"))
    print(f"\n[{split_dir}] 标签文件数: {len(label_files)}")

    for lbl_path in label_files:
        img_stem = lbl_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = image_dir / f"{img_stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        img_sizes.append((W, H))

        with open(lbl_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        objs_per_img.append(len(lines))

        for ln in lines:
            parts = ln.split()
            cls_id = int(parts[0])
            _, _, w, h = map(float, parts[1:5])
            class_counter[cls_id] += 1
            bbox_sizes.append((w, h))

    # --- 报告 ---
    print(f"图像数 (有标签): {len(img_sizes)}")
    if img_sizes:
        ws, hs = zip(*img_sizes)
        print(f"图像分辨率: W in [{min(ws)}, {max(ws)}], H in [{min(hs)}, {max(hs)}]")

    print(f"总目标数: {sum(class_counter.values())}")
    if objs_per_img:
        print(f"每图目标数: 平均 {sum(objs_per_img)/len(objs_per_img):.2f}, "
              f"最大 {max(objs_per_img)}, 最小 {min(objs_per_img)}")

    if bbox_sizes:
        ws_rel = [b[0] for b in bbox_sizes]
        hs_rel = [b[1] for b in bbox_sizes]
        # 估算等效边长 sqrt(w*h)
        areas = [w * h for w, h in bbox_sizes]
        tiny = sum(1 for a in areas if a < 0.01)    # 占图像面积 <1%
        small = sum(1 for a in areas if 0.01 <= a < 0.05)
        medium = sum(1 for a in areas if 0.05 <= a < 0.2)
        large = sum(1 for a in areas if a >= 0.2)
        total = len(areas)
        print(f"BBox 相对尺寸 (相对图像 W,H 归一化):")
        print(f"  w_rel: 中位 {sorted(ws_rel)[len(ws_rel)//2]:.4f}, "
              f"最大 {max(ws_rel):.4f}")
        print(f"  h_rel: 中位 {sorted(hs_rel)[len(hs_rel)//2]:.4f}, "
              f"最大 {max(hs_rel):.4f}")
        print(f"  极小目标 (面积<1%): {tiny} ({100*tiny/total:.1f}%)")
        print(f"  小目标 (1%~5%):   {small} ({100*small/total:.1f}%)")
        print(f"  中目标 (5%~20%):  {medium} ({100*medium/total:.1f}%)")
        print(f"  大目标 (>20%):    {large} ({100*large/total:.1f}%)")

    # 类别分布
    print(f"\n类别分布 (按样本数降序, 共 {len(class_counter)} 类有样本):")
    sorted_classes = class_counter.most_common()
    for cls_id, cnt in sorted_classes:
        name = class_names[cls_id] if cls_id < len(class_names) else f"<{cls_id}>"
        bar = "#" * int(50 * cnt / sorted_classes[0][1])
        print(f"  [{cls_id:2d}] {name:<8s} {cnt:5d}  {bar}")

    # 缺失类
    missing = [i for i in range(len(class_names)) if i not in class_counter]
    if missing:
        print(f"\n⚠️  未出现的类 ({len(missing)} 个): "
              f"{[class_names[i] for i in missing]}")

    return class_counter, bbox_sizes


if __name__ == "__main__":
    names = load_classes("data.yaml")
    print(f"配置类别数: {len(names)}")

    print("\n" + "=" * 60)
    print("训练集分析")
    print("=" * 60)
    train_cnt, _ = analyze_split("object_det_dataset/train", names)

    print("\n" + "=" * 60)
    print("验证集分析")
    print("=" * 60)
    val_cnt, _ = analyze_split("object_det_dataset/val", names)

    # 长尾建议
    print("\n" + "=" * 60)
    print("长尾分析建议")
    print("=" * 60)
    if train_cnt:
        max_cnt = train_cnt.most_common(1)[0][1]
        rare = [(cid, c) for cid, c in train_cnt.items() if c < max_cnt * 0.1]
        if rare:
            print(f"稀有类 (样本数 < 头部类的 10%): {len(rare)} 个")
            print("建议: ")
            print(" - copy_paste 增强 (已在 train.py 中设为 0.3)")
            print(" - 必要时对稀有类做过采样")
            print(" - 用 TT100K 完整版补充稀有类样本 (Phase 2)")
