# Nut Inspection - YOLOv8 目标检测项目

## 项目概述
基于 YOLOv8 的螺母缺陷检测项目，使用 Anaconda 管理环境，GPU 加速训练。

## 环境配置

| 组件 | 版本/详情 |
|------|-----------|
| Conda 环境名 | `yolo_env` |
| 环境路径 | `D:\Anaconda3\envs\yolo_env` |
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 (CUDA 版) |
| TorchVision | 0.23.0+cu128 |
| Ultralytics | 8.4.67 |
| OpenCV | 4.13.0 |
| NumPy | 2.0.2 |
| Pillow | 11.3.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) |
| CUDA Driver | 13.0 |
| CUDA Toolkit | 12.8 |

### 激活环境
```powershell
conda activate yolo_env
```

### 关键安装记录
- 初始安装的是 PyTorch 2.8.0 CPU 版本（无法使用 GPU）
- 2026-06-15 替换为 CUDA 版本：
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
- 安装包约 3.5GB，pip 默认源下载较慢（~1.2 MB/s），建议后续使用镜像源

## 项目结构

```
Nut_inspection/
├── yolov8n.pt                    # YOLOv8 Nano 预训练模型 (6.5MB)
├── datasets/
│   ├── coco8.yaml                # COCO8 数据集配置（80类）
│   └── coco8/                    # 8张图片：4训练 / 4验证
│       ├── images/train/         # 训练图片
│       ├── images/val/           # 验证图片
│       └── labels/{train,val}/   # YOLO 格式标注
└── runs/detect/
    ├── train-6/                  # 最新训练结果 (2026-06-15)
    │   ├── weights/best.pt       # 验证集最佳模型 (6.5MB)
    │   ├── weights/last.pt       # 最后一轮模型
    │   ├── results.png           # 训练曲线图
    │   ├── results.csv           # 每轮指标数据
    │   ├── confusion_matrix.png  # 混淆矩阵
    │   ├── args.yaml             # 训练参数记录
    │   └── *.jpg                 # 批次可视化图片
    └── train ~ train-5/          # 之前的训练记录
```

## 训练命令

### 基本训练
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='datasets/coco8.yaml',  # 数据集配置
    epochs=50,                    # 训练轮数
    imgsz=640,                    # 图片尺寸
    device=0,                     # GPU 设备 (0=第一块GPU, 'cpu'=CPU)
    name='train-6',               # 训练名称
    exist_ok=True                 # 覆盖同名目录
)
```

### CLI 训练（备选）
```bash
yolo detect train data=datasets/coco8.yaml model=yolov8n.pt epochs=50 device=0
```

## 最新训练结果 (train-6, 2026-06-15)

| 指标 | 最佳值 | 说明 |
|------|--------|------|
| 训练时间 | 0.010 小时 (37秒) | 4张训练图，50 epochs |
| mAP50 | **0.889** | IoU=0.5 时最佳精度 |
| mAP50-95 | **0.633** | 更严格的综合指标 |
| Precision (best) | 0.791 | 精确率 |
| Recall (best) | 0.833 | 召回率 |
| GPU 显存占用 | ~0.6-0.9 GB | 远低于 8GB 上限 |
| 推理速度 | 6.8ms/张 (GPU) | 不含前后处理 |

### 各类别表现 (best.pt 验证)
| 类别 | mAP50 | mAP50-95 |
|------|-------|----------|
| dog | 0.995 | 0.697 |
| horse | 0.995 | 0.746 |
| umbrella | 0.995 | 0.895 |
| potted plant | 0.995 | 0.895 |
| elephant | 0.828 | 0.290 |
| person | 0.523 | 0.273 |

> ⚠️ COCO8 仅有 8 张图片，指标波动大是正常的。实际项目需要更多数据。

## 如何解读训练结果

### 训练中关注的指标
- **box_loss / cls_loss / dfl_loss**: 越低越好，持续下降说明模型在学习
- **mAP50**: 目标检测核心指标，IoU阈值=0.5
- **mAP50-95**: 更严格，IoU从0.5到0.95取平均

### 结果文件
- `results.png` — 最重要！看 loss 下降曲线和 mAP 上升曲线
- `confusion_matrix.png` — 看哪些类别容易混淆
- `val_batch*_pred.jpg` — 直观查看模型预测效果

### 判断训练好坏
- ✅ loss 持续下降并趋于稳定
- ✅ mAP 持续上升并趋于稳定
- ⚠️ mAP 在验证集开始下降 → 过拟合，减少 epochs
- ⚠️ loss 不下降 → 学习率不合适或数据有问题

## 注意事项
1. 使用 `conda activate yolo_env` 激活环境后再运行脚本
2. 训练默认使用 GPU (`device=0`)，如果不可用会自动回退 CPU
3. 数据集路径在 `coco8.yaml` 中使用绝对路径，换机器需要修改
4. 每次训练会在 `runs/detect/` 下创建递增编号的新目录
5. 模型权重 `.pt` 文件包含完整模型结构和参数，可直接用于推理
