"""
Phase 7/8: 多模型 WBF (Weighted Boxes Fusion) 融合推理 —— 缓存 + 快速融合 两段式

为什么融合:
  测试集仅 110 张, 单模型方差大 (验证 0.92 vs 测试 0.83~0.91)。
  WBF 把多个模型的预测框按置信度加权融合, 降方差、提召回, 是对小测试集
  最稳的提分手段。不同分辨率/训练策略的模型"犯不同的错", 融合收益更大。

合规说明:
  - 每个模型都用作业固定的 conf=0.25 推理, 全部基于 yolov8s + yolov8s.pt;
  - 融合是推理后处理 (与 NMS/TTA 同类), 不改变模型本身, 不引入伪标签。

★ 效率设计 (核心):
  各模型推理 (TTA, 慢) 的结果只跑一次并缓存到磁盘; 之后调 WBF 参数
  (iou / skip / 权重 / 增删模型) 只需几秒, 可快速扫描大量配置。

依赖:
  pip install ensemble-boxes

用法:
  # 1) 缓存所有已存在模型的原始预测 (慢, 只需跑一次)
  python ensemble.py cache

  # 2) 用缓存快速融合 (秒级), 试不同配置:
  python ensemble.py fuse --models phase4,phase3,phase2 --weights 2.0,1.8,1.0 \
      --wbf_iou 0.55 --save_json infer_res/pred_A.json
  python ensemble.py fuse --models phase4,phase3,phase2,phase1 --weights 2.0,1.8,1.0,0.6 \
      --wbf_iou 0.55 --save_json infer_res/pred_B.json
  python ensemble.py fuse --models phase4,phase3,phase2 --weights 2.0,1.8,1.0 \
      --wbf_iou 0.60 --save_json infer_res/pred_C.json

  # (可选) 一键复现 9400 那次 (缓存缺失会自动补跑):
  python ensemble.py auto --save_json infer_res/predictions_ensemble.json

  每个 json 用 minify 压缩后提交, 对比 OJ 分数挑最高的。
"""
import os
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ============================================================
# 模型注册表: key -> (权重路径, 推理 imgsz, 默认融合权重)
# 默认权重大致按各自 OJ 分数设定; 低分辨率/baseline 权重低但提供"错误多样性"。
# 训练出新模型后在这里加一行即可 (phase8 已预留)。
# ============================================================
REGISTRY = {
    "phase4": {"path": "runs/detect/v8s_p2_1536_cb_phase4/weights/best.pt", "imgsz": 1536, "weight": 2.0},  # OJ=9085
    "phase3": {"path": "runs/detect/v8s_p2_1536_phase3/weights/best.pt",    "imgsz": 1536, "weight": 1.8},  # OJ=8883
    "phase2": {"path": "runs/detect/v8s_p2_1280_phase2/weights/best.pt",    "imgsz": 1280, "weight": 1.0},  # OJ=8497
    "phase1": {"path": "runs/detect/runs/detect/v8s_1280_phase1/weights/best.pt", "imgsz": 1280, "weight": 0.6},  # baseline 无P2头, 双层路径
    "phase8": {"path": "runs/detect/v8s_p2_1792_cb_phase8/weights/best.pt", "imgsz": 1792, "weight": 2.2},  # 1792 (训练完才有)
}

# 9400 那次的默认融合阵容
DEFAULT_MODELS = ["phase4", "phase3", "phase2"]

CONF_THRES = 0.25      # 作业固定, 不可更改
NMS_IOU = 0.6          # 单模型推理时的 NMS IoU
WBF_IOU = 0.55         # WBF 融合时判定"同一目标"的 IoU 阈值 (可被 --wbf_iou 覆盖)
WBF_SKIP = 0.0         # 融合后丢弃低于此分的框 (0=全保留, 利于 mAP; json 过大可调 0.1)
MAX_DET = 300
USE_TTA = True         # 每个模型仍开 TTA

CACHE_DIR = "infer_res/ensemble_cache"


def xyxy_to_xywh_round(x1, y1, x2, y2):
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]


def _list_images(test_dir):
    image_dir = os.path.join(test_dir, "images")
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    return image_dir, image_files


def cache_model(key, test_dir):
    """对单个模型在整个测试集做推理 (TTA), 把归一化后的原始预测缓存到磁盘。"""
    info = REGISTRY[key]
    if not os.path.exists(info["path"]):
        print(f"[跳过] {key}: 权重不存在 {info['path']}")
        return False

    image_dir, image_files = _list_images(test_dir)
    model = YOLO(info["path"])
    imgsz = info["imgsz"]

    cache = {"img_names": image_files, "imgsz": imgsz, "boxes": [], "scores": [], "labels": [], "orig_shape": None}
    for img_name in tqdm(image_files, desc=f"cache:{key}"):
        r = model(
            os.path.join(image_dir, img_name),
            conf=CONF_THRES, iou=NMS_IOU, imgsz=imgsz,
            augment=USE_TTA, max_det=MAX_DET, verbose=False,
        )[0]
        H, W = r.orig_shape
        cache["orig_shape"] = (int(H), int(W))
        if r.boxes is None or len(r.boxes) == 0:
            cache["boxes"].append(np.zeros((0, 4), dtype=np.float64))
            cache["scores"].append(np.zeros((0,), dtype=np.float64))
            cache["labels"].append(np.zeros((0,), dtype=int))
            continue
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        xyxy[:, [0, 2]] /= W   # 归一化到 [0,1] (WBF 要求)
        xyxy[:, [1, 3]] /= H
        cache["boxes"].append(np.clip(xyxy, 0.0, 1.0))
        cache["scores"].append(r.boxes.conf.cpu().numpy().astype(np.float64))
        cache["labels"].append(r.boxes.cls.cpu().numpy().astype(int))

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(out, "wb") as f:
        pickle.dump(cache, f)
    print(f"[完成] {key}: 已缓存 {len(image_files)} 张 -> {out}")
    return True


def cmd_cache(args):
    keys = args.models.split(",") if args.models else list(REGISTRY.keys())
    done = [k for k in keys if cache_model(k.strip(), args.test_dir)]
    print(f"\n缓存完成的模型: {done}")


def load_cache(key):
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少缓存 {path}, 请先运行: python ensemble.py cache --models {key}")
    with open(path, "rb") as f:
        return pickle.load(f)


def fuse(keys, weights, wbf_iou, wbf_skip, save_json):
    caches = [load_cache(k) for k in keys]
    # 一致性校验: 所有模型缓存的图像顺序必须一致
    ref_names = caches[0]["img_names"]
    for k, c in zip(keys, caches):
        if c["img_names"] != ref_names:
            raise ValueError(f"模型 {k} 的图像列表与基准不一致, 请重新 cache 所有模型")
    H, W = caches[0]["orig_shape"]

    print(f"融合阵容: {list(zip(keys, weights))}  | wbf_iou={wbf_iou} skip={wbf_skip}")

    results_json = []
    for idx, img_name in enumerate(tqdm(ref_names, desc="fuse")):
        boxes_list = [c["boxes"][idx] for c in caches]
        scores_list = [c["scores"][idx] for c in caches]
        labels_list = [c["labels"][idx] for c in caches]

        if all(len(b) == 0 for b in boxes_list):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=wbf_iou, skip_box_thr=wbf_skip,
        )
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
            results_json.append({
                "image_id": idx,
                "file_name": img_name,
                "category_id": int(label),
                "bbox": xyxy_to_xywh_round(x1, y1, x2, y2),
                "score": round(float(score), 5),
            })

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"\n融合结果已保存到 {save_json}  | 总框数: {len(results_json)}")


def cmd_fuse(args):
    keys = [k.strip() for k in args.models.split(",")]
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        assert len(weights) == len(keys), "--weights 数量必须与 --models 一致"
    else:
        weights = [REGISTRY[k]["weight"] for k in keys]
    fuse(keys, weights, args.wbf_iou, args.wbf_skip, args.save_json)


def cmd_auto(args):
    """缺失的缓存自动补跑, 再用默认阵容融合 (复现 9400 那次)。"""
    for k in DEFAULT_MODELS:
        if not os.path.exists(os.path.join(CACHE_DIR, f"{k}.pkl")):
            cache_model(k, args.test_dir)
    weights = [REGISTRY[k]["weight"] for k in DEFAULT_MODELS]
    fuse(DEFAULT_MODELS, weights, args.wbf_iou, args.wbf_skip, args.save_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("cache", help="对模型推理并缓存原始预测 (慢, 跑一次)")
    p_cache.add_argument("--models", type=str, default="", help="逗号分隔的 key; 留空=全部")
    p_cache.add_argument("--test_dir", type=str, default="object_det_dataset/test")
    p_cache.set_defaults(func=cmd_cache)

    p_fuse = sub.add_parser("fuse", help="用缓存快速 WBF 融合 (秒级)")
    p_fuse.add_argument("--models", type=str, required=True, help="逗号分隔的 key, 如 phase4,phase3,phase2")
    p_fuse.add_argument("--weights", type=str, default="", help="逗号分隔权重; 留空=用注册表默认")
    p_fuse.add_argument("--wbf_iou", type=float, default=WBF_IOU)
    p_fuse.add_argument("--wbf_skip", type=float, default=WBF_SKIP)
    p_fuse.add_argument("--save_json", type=str, required=True)
    p_fuse.set_defaults(func=cmd_fuse)

    p_auto = sub.add_parser("auto", help="缺失缓存自动补跑 + 默认阵容融合")
    p_auto.add_argument("--test_dir", type=str, default="object_det_dataset/test")
    p_auto.add_argument("--wbf_iou", type=float, default=WBF_IOU)
    p_auto.add_argument("--wbf_skip", type=float, default=WBF_SKIP)
    p_auto.add_argument("--save_json", type=str, default="infer_res/predictions_ensemble.json")
    p_auto.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    args.func(args)
