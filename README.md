# Object-Detection
## Yehan WANG, ECNU

<div align="center">
    <img src="https://cdn.jsdelivr.net/gh/VisionNext100/VisionNext100.github.io@main/public/images/projects/project-objdet.png" width="800" alt="Object-Detection">
    <br>
</div>

### Ⅰ Overview

A **YOLOv8s**-based traffic-sign detection system for **45 categories** in complex autonomous-driving scenes, optimized for **mAP@0.5**.

Built on Ultralytics with modifications tailored to tiny objects and long-tailed class distributions:

- **P2 detection head** (stride=4) for extremely small signs (~99.6% of boxes occupy &lt;1% of image area)
- **Higher input resolutions**: 1280 / 1536 / 1792
- **Long-tail–friendly augmentation & sampling**: `copy_paste`, class-balanced sampler
- **Custom losses**: WIoU / EIoU, Varifocal Loss (toggled via environment variables)
- **Multi-model WBF ensemble**: cache-then-fuse to reduce variance on a small test set

Dataset config: `data.yaml` (`nc: 45`, root `./object_det_dataset`). Pretrain constraint: **yolov8s.pt** only.

### Ⅱ Method

| Phase | Script | Highlights |
| :---: | --- | --- |
| 1 | `train.py` | YOLOv8s @ 1280, AdamW + cosine LR; no vertical flip; mixup / copy_paste |
| 2 | `train_p2.py` | `yolov8s-p2.yaml` + load `yolov8s.pt`; stronger copy_paste; smaller batch |
| 3–4 | `train_p2_1536.py` / `train_p2_cb.py` | imgsz 1536; class-balanced sampling |
| 5 | `train_p2_loss.py` | P2 + 1536 + CB + WIoU / Varifocal |
| 8 | `train_p2_1792.py` | Higher-resolution ablation |
| Infer | `infer.py` | conf=0.25 (fixed); TTA on by default; export JSON |
| Ensemble | `ensemble.py` | Cached WBF; `cache` / `fuse` / `auto` subcommands |

Dataset diagnostics: `python analyze_dataset.py` (class imbalance, bbox scale, image resolution).

### Ⅲ Setup

Recommended on Linux + CUDA (e.g. AutoDL RTX 4090):

```bash
cd Object-Detection
bash setup_env.sh
# or install the local ultralytics package in editable mode and pin numpy
pip install -e .
pip install "numpy==1.26.4" tqdm nvitop ensemble-boxes
```

See `requirements.txt` for a dependency summary. Place `yolov8s.pt` under `./pretrain_models/` and prepare `object_det_dataset/{train,val,test}` as specified in `data.yaml`.

### Ⅳ Training

```bash
# Phase 1 baseline
python train.py
# or
bash train.sh

# Phase 2: P2 head
python train_p2.py
# or
bash train_p2.sh

# Higher resolution / long-tail sampling / loss upgrades
python train_p2_1536.py
python train_p2_cb.py
python train_p2_loss.py
python train_p2_1792.py
```

Checkpoints are saved to `runs/detect/<exp_name>/weights/best.pt` by default.

### Ⅴ Inference & Ensemble

```bash
# Single-model inference (imgsz must match training)
python infer.py \
  --model runs/detect/v8s_p2_1536_cb_phase4/weights/best.pt \
  --imgsz 1536 \
  --save_json infer_res/predictions_phase4.json

# Multi-model WBF: cache once (slow), then fuse repeatedly (fast)
python ensemble.py cache
python ensemble.py fuse --models phase4,phase3,phase2 --weights 2.0,1.8,1.0 \
  --wbf_iou 0.55 --save_json infer_res/pred_ensemble.json
# or one-shot default lineup
python ensemble.py auto --save_json infer_res/predictions_ensemble.json
```

Use `minify_json.sh` to compress prediction JSON before submission.

### Ⅵ Repository Layout

```text
Object-Detection/
├── data.yaml              # 45-class dataset config
├── analyze_dataset.py     # dataset diagnostics
├── train*.py / train*.sh  # staged training
├── infer.py               # single-model inference → JSON
├── ensemble.py            # WBF ensemble inference
├── setup_env.sh           # one-shot environment setup
├── ultralytics/           # local editable Ultralytics (loss / sampler patches)
└── requirements.txt
```

### Ⅶ License

See [LICENSE](./LICENSE).
