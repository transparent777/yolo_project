# Nut Binary Classifier — Implementation Plan

> **Execution:** Use Inline Execution with task-by-task tracking. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train YOLOv8n-cls OK/NG classifier on NUT.v2, validate >92% test accuracy, export ONNX for STM32N6570.

**Architecture:** Convert YOLO detection labels to ImageNet folder structure → train yolov8n-cls with binary head → evaluate on test set → export ONNX int8.

**Tech Stack:** Python 3.9, PyTorch 2.8.0+cu128, Ultralytics 8.4.67, onnx, onnx-simplifier

---

### Task 1: Data Conversion Script

**Files:**
- Create: `scripts/convert_to_cls.py`
- Create: `datasets/nut_classification/` (auto-generated)

**Purpose:** Convert NUT.v2 YOLO detection labels to ImageNet-style folder structure for classification training.

- [ ] **Step 1: Write the conversion script**

```python
"""
scripts/convert_to_cls.py
将 NUT.v2 YOLO 检测标注转换为分类数据集文件夹格式。
OK: class 0,1  NG: class 2,3,4,5,6,7
"""
import os
import shutil
from pathlib import Path

# 配置
DATASET_ROOT = Path(__file__).parent.parent / "Nuts.v2(DST1506)"
OUTPUT_ROOT = Path(__file__).parent.parent / "datasets" / "nut_classification"

OK_CLASSES = {0, 1}  # Excellent, Side_Excellent
NG_CLASSES = {2, 3, 4, 5, 6, 7}  # Rust, Side_Rust, Fracture, Side-Fracture, Scratches, Side_Scratches

def get_label_class(label_path: Path) -> int:
    """读取 YOLO .txt 标注，返回类别 ID。
    每行格式: class_id cx cy w h
    每张图只有一个目标，取第一行的 class_id。"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return -1  # 无标注，跳过
    return int(lines[0].strip().split()[0])

def convert_split(split: str):
    """转换 train/val/test 中的一个 split"""
    src_images = DATASET_ROOT / "images" / split
    src_labels = DATASET_ROOT / "labels" / split

    if not src_images.exists():
        print(f"  [SKIP] {src_images} 不存在")
        return

    ok_count = 0
    ng_count = 0
    skip_count = 0

    for img_file in src_images.iterdir():
        if not img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue

        # 找到对应的 label 文件
        label_file = src_labels / f"{img_file.stem}.txt"
        if not label_file.exists():
            skip_count += 1
            continue

        cls_id = get_label_class(label_file)
        if cls_id in OK_CLASSES:
            dst_dir = OUTPUT_ROOT / split / "ok"
            ok_count += 1
        elif cls_id in NG_CLASSES:
            dst_dir = OUTPUT_ROOT / split / "ng"
            ng_count += 1
        else:
            skip_count += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, dst_dir / img_file.name)

    print(f"  {split}: OK={ok_count}, NG={ng_count}, skip={skip_count}")


if __name__ == "__main__":
    print("Converting NUT.v2 → nut_classification ...")
    for s in ["train", "valid", "test"]:
        convert_split(s)

    # 统计
    print("\n=== 数据集统计 ===")
    for s in ["train", "valid", "test"]:
        ok_dir = OUTPUT_ROOT / s / "ok"
        ng_dir = OUTPUT_ROOT / s / "ng"
        ok_n = len(list(ok_dir.glob("*"))) if ok_dir.exists() else 0
        ng_n = len(list(ng_dir.glob("*"))) if ng_dir.exists() else 0
        print(f"  {s}: OK={ok_n}, NG={ng_n}, total={ok_n + ng_n}")
    print("Done!")
```

- [ ] **Step 2: Run conversion script**

```powershell
conda activate yolo_env; python scripts/convert_to_cls.py
```

Expected output:
```
Converting NUT.v2 → nut_classification ...
  train: OK=840, NG=2520, skip=0
  valid: OK=315, NG=945, skip=0
  test: OK=315, NG=945, skip=0

=== 数据集统计 ===
  train: OK=840, NG=2520, total=3360
  valid: OK=315, NG=945, total=1260
  test: OK=315, NG=945, total=1260
```

- [ ] **Step 3: Verify a few samples can be read**

```powershell
conda activate yolo_env; python -c "
from PIL import Image
from pathlib import Path
import random

root = Path('datasets/nut_classification')
for split in ['train', 'val']:
    for cls in ['ok', 'ng']:
        imgs = list((root / split / cls).glob('*.jpg'))
        sample = random.choice(imgs)
        img = Image.open(sample)
        print(f'{split}/{cls}: {sample.name} size={img.size} mode={img.mode}')
print('All samples OK!')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/convert_to_cls.py
git commit -m "feat: add data conversion script (YOLO detect → cls folder)"
```

---

### Task 2: Training Script

**Files:**
- Create: `scripts/train_classifier.py`
- Create: `runs/classify/nut-cls/` (output dir)

- [ ] **Step 1: Write training script**

```python
"""
scripts/train_classifier.py
训练 YOLOv8n-cls 螺母 OK/NG 二分类模型
"""
from ultralytics import YOLO
from pathlib import Path

# 路径配置
DATA_DIR = Path(__file__).parent.parent / "datasets" / "nut_classification"

if __name__ == "__main__":
    # 加载预训练分类模型
    model = YOLO('yolov8n-cls.pt')

    results = model.train(
        data=str(DATA_DIR),       # ImageNet 文件夹格式
        epochs=50,                 # 训练轮数
        patience=10,               # EarlyStopping
        imgsz=224,                 # 图片尺寸
        batch=32,                  # batch size
        optimizer='AdamW',
        lr0=1e-3,                  # 初始学习率
        cos_lr=True,               # Cosine decay
        device=0,                  # GPU
        name='nut-cls',            # 输出目录名
        exist_ok=True,             # 覆盖已有目录

        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5,

        # 验证
        val=True,
        save=True,
        save_period=5,             # 每 5 epoch 保存一次
    )

    print(f"Training done! Best weights at: {model.trainer.best}")
    print(f"Best metrics: {model.trainer.metrics}")
```

- [ ] **Step 2: Run training**

```powershell
conda activate yolo_env; python scripts/train_classifier.py
```

Expected: 训练约 2-5 分钟（3360 张图，224×224，batch=32，RTX 4060）

- [ ] **Step 3: Verify training output**

```powershell
ls runs/classify/nut-cls/weights/
```

Expected: `best.pt` 和 `last.pt` 存在。

- [ ] **Step 4: Quick check training curves**

```powershell
ls runs/classify/nut-cls/results.png
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_classifier.py
git commit -m "feat: add YOLOv8n-cls training script"
```

---

### Task 3: Evaluation Script

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Write evaluation script**

```python
"""
scripts/evaluate.py
在 test 集上评估训练好的分类模型
"""
from ultralytics import YOLO
from pathlib import Path
import json

MODEL_PATH = Path(__file__).parent.parent / "runs" / "classify" / "nut-cls" / "weights" / "best.pt"
TEST_DIR = Path(__file__).parent.parent / "datasets" / "nut_classification" / "test"

if __name__ == "__main__":
    model = YOLO(str(MODEL_PATH))

    # 在 test 集上评估
    results = model.val(
        data=str(TEST_DIR.parent),  # 会自动找 test/ 子目录
        split='test',
        imgsz=224,
        device=0,
    )

    # 打印关键指标
    acc = results.top1 if hasattr(results, 'top1') else results.results_dict.get('metrics/accuracy_top1', 'N/A')
    acc5 = results.top5 if hasattr(results, 'top5') else results.results_dict.get('metrics/accuracy_top5', 'N/A')

    print(f"\n=== 评估结果 ===")
    print(f"Top-1 Accuracy: {acc}")
    print(f"Top-5 Accuracy: {acc5}")
    print(f"\n完整指标: {results.results_dict}")

    # 保存指标到 JSON
    with open("runs/classify/nut-cls/eval_metrics.json", "w") as f:
        json.dump(results.results_dict, f, indent=2)
```

- [ ] **Step 2: Run evaluation**

```powershell
conda activate yolo_env; python scripts/evaluate.py
```

Expected: Top-1 Accuracy > 92%

- [ ] **Step 3: Check confusion matrix**

The confusion matrix is automatically generated by ultralytics during validation. Check `runs/classify/nut-cls/confusion_matrix.png`.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add evaluation script for classifier"
```

---

### Task 4: ONNX Export Script

**Files:**
- Create: `scripts/export_for_stm32.py`
- Create: `models/` directory

- [ ] **Step 1: Write export script**

```python
"""
scripts/export_for_stm32.py
将训练好的 PyTorch 模型导出为 ONNX (int8 量化准备)
"""
from ultralytics import YOLO
from pathlib import Path
import subprocess

MODEL_PATH = Path(__file__).parent.parent / "runs" / "classify" / "nut-cls" / "weights" / "best.pt"
MODELS_DIR = Path(__file__).parent.parent / "models"

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    # Step 1: 导出 ONNX (FP32)
    print("Exporting to ONNX (FP32)...")
    model.export(
        format='onnx',
        imgsz=224,
        opset=12,
        simplify=True,
        half=False,  # FP32，为后续 int8 量化校准做准备
    )

    # 移动 ONNX 文件到 models/
    src = Path("runs/classify/nut-cls/weights/best.onnx")
    if src.exists():
        import shutil
        dst = MODELS_DIR / "nut_cls_fp32.onnx"
        shutil.move(str(src), str(dst))
        print(f"FP32 ONNX saved to: {dst}")
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")

    # Step 2: 验证 ONNX 可以加载
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(str(MODELS_DIR / "nut_cls_fp32.onnx"))
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

        # 打印输入输出信息
        print(f"  Input:  {onnx_model.graph.input[0].name} shape={onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"  Output: {onnx_model.graph.output[0].name} shape={onnx_model.graph.output[0].type.tensor_type.shape}")
        print(f"  Ops: {len(set(n.op_type for n in onnx_model.graph.node))} unique ops")

    except ImportError:
        print("  onnx not installed, skipping validation. Install: pip install onnx onnx-simplifier")

    print("\nExport complete! Next step for STM32:")
    print("  1. Quantize with X-CUBE-AI Developer Cloud (onnx → int8)")
    print("  2. Generate C code via STM32CubeMX + X-CUBE-AI")
    print("  3. Deploy to STM32N6570")
```

- [ ] **Step 2: Install ONNX tools**

```powershell
conda activate yolo_env; pip install onnx onnx-simplifier
```

- [ ] **Step 3: Run export**

```powershell
conda activate yolo_env; python scripts/export_for_stm32.py
```

Expected: 生成 `models/nut_cls_fp32.onnx`，大小约 5-10 MB（FP32）。

- [ ] **Step 4: Commit**

```bash
git add scripts/export_for_stm32.py
git commit -m "feat: add ONNX export script for STM32 deployment"
```

---

### Summary

| Task | File | Purpose |
|------|------|---------|
| 1 | `scripts/convert_to_cls.py` | YOLO labels → cls folders |
| 2 | `scripts/train_classifier.py` | Train YOLOv8n-cls |
| 3 | `scripts/evaluate.py` | Test set evaluation |
| 4 | `scripts/export_for_stm32.py` | ONNX export for MCU |

**Total estimated time:** ~15-30 min (training ~3-5 min for full 50 epochs)
