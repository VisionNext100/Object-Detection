"""
Phase 7: 多模型 WBF (Weighted Boxes Fusion) 融合推理

为什么融合:
  测试集仅 110 张, 单模型方差大 (验证 0.92 vs 测试 0.83~0.91)。
  WBF 把多个模型的预测框按置信度加权融合, 降方差、提召回, 是对小测试集
  最稳的提分手段。不同分辨率/训练策略的模型"犯不同的错", 融合收益更大。

合规说明:
  - 每个模型都用作业固定的 conf=0.25 推理, 全部基于 yolov8s + yolov8s.pt;
  - 融合是推理后处理 (与 NMS/TTA 同类), 不改变模型本身, 不引入伪标签。

依赖:
  pip install ensemble-boxes

用法:
  python ensemble.py --save_json infer_res/predictions_ensemble.json
  bash minify_json.sh infer_res/predictions_ensemble.json
  # 提交 predictions_ensemble_min.json
"""
import os
import json
import argparse

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ============================================================
# 参与融合的模型 (path, 推理 imgsz, 融合权重)
# 权重大致按各自 OJ 分数设定; 低分辨率模型权重低但提供"错误多样性"。
# 跑完某个新模型后, 把它按需加进来即可。
# ============================================================
MODELS = [
    {"path": "runs/detect/v8s_p2_1536_cb_phase4/weights/best.pt", "imgsz": 1536, "weight": 2.0},  # Phase4 OJ=9085
    {"path": "runs/detect/v8s_p2_1536_phase3/weights/best.pt",    "imgsz": 1536, "weight": 1.8},  # Phase3 OJ=8883
    {"path": "runs/detect/v8s_p2_1280_phase2/weights/best.pt",    "imgsz": 1280, "weight": 1.0},  # Phase2 OJ=8497
]

CONF_THRES = 0.25      # 作业固定, 不可更改
NMS_IOU = 0.6          # 单模型推理时的 NMS IoU
WBF_IOU = 0.55         # WBF 融合时判定"同一目标"的 IoU 阈值
WBF_SKIP = 0.0         # 融合后丢弃低于此分的框 (0=全保留, 利于 mAP; 若 json 过大可调到 0.1)
MAX_DET = 300
USE_TTA = True         # 每个模型仍开 TTA


def xyxy_to_xywh_round(x1, y1, x2, y2):
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]


def run_ensemble(test_dir, save_json):
    models = [(YOLO(m["path"]), m["imgsz"], m["weight"]) for m in MODELS]
    weights = [m["weight"] for m in MODELS]
    print(f"已加载 {len(models)} 个模型用于 WBF 融合:")
    for m in MODELS:
        print(f"  - {m['path']}  (imgsz={m['imgsz']}, weight={m['weight']})")

    image_dir = os.path.join(test_dir, "images")
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )

    results_json = []
    for idx, img_name in enumerate(tqdm(image_files, desc="Ensemble")):
        img_path = os.path.join(image_dir, img_name)

        boxes_list, scores_list, labels_list = [], [], []
        H = W = None
        for model, imgsz, _ in models:
            r = model(
                img_path, conf=CONF_THRES, iou=NMS_IOU, imgsz=imgsz,
                augment=USE_TTA, max_det=MAX_DET, verbose=False,
            )[0]
            H, W = r.orig_shape  # (H, W), 本数据集均为 2048x2048
            if r.boxes is None or len(r.boxes) == 0:
                boxes_list.append(np.zeros((0, 4)))
                scores_list.append(np.zeros((0,)))
                labels_list.append(np.zeros((0,)))
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            # WBF 要求坐标归一化到 [0,1]
            xyxy[:, [0, 2]] /= W
            xyxy[:, [1, 3]] /= H
            boxes_list.append(np.clip(xyxy, 0.0, 1.0))
            scores_list.append(r.boxes.conf.cpu().numpy().astype(np.float64))
            labels_list.append(r.boxes.cls.cpu().numpy().astype(int))

        # 若所有模型都无框, 跳过
        if all(len(b) == 0 for b in boxes_list):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=WBF_IOU, skip_box_thr=WBF_SKIP,
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

    print(f"\n融合结果已保存到 {save_json}")
    print(f"处理图像: {len(image_files)} 张  | 融合后总框数: {len(results_json)} 个")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="object_det_dataset/test")
    parser.add_argument("--save_json", type=str, default="infer_res/predictions_ensemble.json")
    args = parser.parse_args()
    run_ensemble(args.test_dir, args.save_json)
