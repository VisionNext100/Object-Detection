import os
import json
import argparse
from tqdm import tqdm
from ultralytics import YOLO

CLASS_NAMES = {
    0: 'pl80',  1: 'p6',   2: 'p5',    3: 'pm55', 4: 'pl60',
    5: 'ip',    6: 'p11',  7: 'i2r',   8: 'p23',  9: 'pg',
    10: 'il80', 11: 'ph4', 12: 'i4',   13: 'pl70', 14: 'pne',
    15: 'ph4.5', 16: 'p12', 17: 'p3',  18: 'pl5',  19: 'w13',
    20: 'i4l',  21: 'pl30', 22: 'p10', 23: 'pn',  24: 'w55',
    25: 'p26',  26: 'p13', 27: 'pr40', 28: 'pl20', 29: 'pm30',
    30: 'pl40', 31: 'i2',  32: 'pl120', 33: 'w32', 34: 'ph5',
    35: 'il60', 36: 'w57', 37: 'pl100', 38: 'w59', 39: 'il100',
    40: 'p19',  41: 'pm20', 42: 'i5',  43: 'p27', 44: 'pl50',
}


def xyxy_to_xywh(box):
    # 保留 2 位小数: mAP@0.5 下 0.01px 精度完全足够, 但能显著缩小 json 体积
    x1, y1, x2, y2 = box
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]


def run_inference(model_path, test_dir, save_json, conf_thres, imgsz, iou, augment):
    model = YOLO(model_path)

    image_dir = os.path.join(test_dir, "images")
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )

    results_json = []

    for idx, img_name in enumerate(tqdm(image_files, desc="Inferring")):
        img_path = os.path.join(image_dir, img_name)

        results = model(
            img_path,
            conf=conf_thres,
            iou=iou,
            imgsz=imgsz,
            augment=augment,       # TTA: 多尺度+翻转,几乎免费的+1~3 mAP
            max_det=300,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes, scores, class_ids):
            results_json.append({
                "image_id": idx,
                "file_name": img_name,
                "category_id": int(cls_id),
                "bbox": xyxy_to_xywh(box),
                "score": round(float(score), 5),
            })

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(results_json, f, indent=4)

    print(f"\n预测结果已保存到 {save_json}")
    print(f"处理图像: {len(image_files)} 张  | 总检测框: {len(results_json)} 个")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="./runs/detect/v8s_1280_phase1/weights/best.pt",
                        help="训练好的模型路径")
    parser.add_argument("--test_dir", type=str,
                        default="object_det_dataset/test",
                        help="测试集目录")
    parser.add_argument("--save_json", type=str,
                        default="./infer_res/predictions_phase1.json",
                        help="预测结果保存路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (作业固定 0.25, 不可更改)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="推理图像尺寸,需与训练保持一致")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU 阈值")
    parser.add_argument("--no-tta", action="store_true",
                        help="关闭 TTA (默认开启)")

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        test_dir=args.test_dir,
        save_json=args.save_json,
        conf_thres=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
        augment=not args.no_tta,
    )
