# Nut Inspection - YOLOv8 目标检测项目

## 项目概述
基于 YOLOv8 的螺母缺陷检测项目，使用 Anaconda 管理环境，GPU 加速训练。
当前阶段：YOLOv8n-cls OK/NG 二分类 → STM32N6570 部署。

## 数据来源
**唯一数据源：** `Nuts.v2(DST1506)/`（只读，不可修改）
训练用数据由脚本转换生成至 `datasets/nut_classification/`

## 三层记录系统
| 文件 | 用途 | 更新时机 |
|------|------|---------|
| `CLAUDE.md` | 项目环境/架构/训练结果记录 | 每次训练完成后手动同步 |
| `progress.md` | 会话级进度日志 | 每个操作后实时更新 |
| `findings.md` | 数据分析、踩坑记录、错误排查 | 发现新信息时追加 |

### 训练结果记录（每次训练后更新）
| 日期 | 训练名 | 模型 | 数据集 | 最佳指标 | 备注 |
|------|--------|------|--------|---------|------|
| 2026-06-15 | train-6 | YOLOv8n | coco8 | mAP50=0.889 | 环境验证 |
| 2026-06-15 | nut-cls | YOLOv8n-cls | NUT.v2 (5880张) | Top-1 86.3%(val) / 91.7%(test) | 二分类首版基线 |

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
├── yolov8n.pt                    # YOLOv8n 检测预训练模型 (6.5MB)
├── yolov8n-cls.pt                # YOLOv8n-cls 分类预训练模型 (5.4MB)
├── Nuts.v2(DST1506)/             # 🔒 原始数据集（只读，唯一数据源）
│   ├── data.yaml                  # 8类配置
│   ├── images/{train,valid,test}/ # 5880 张 640×640
│   └── labels/{train,valid,test}/ # YOLO 检测格式标注
├── datasets/
│   ├── coco8/                     # COCO8 验证数据（可删）
│   └── nut_classification/        # 分类工作数据（convert_to_cls.py 生成）
│       ├── train/{ok,ng}/         # OK 840 + NG 2520
│       ├── val/{ok,ng}/           # OK 321 + NG 939
│       └── test/{ok,ng}/          # OK 315 + NG 945
├── models/
│   └── nut_cls_fp32.onnx          # ONNX 导出模型 (5.5MB, 1.4M params)
├── scripts/
│   ├── convert_to_cls.py          # YOLO 标注 → 分类文件夹
│   ├── train_classifier.py        # YOLOv8n-cls 训练
│   ├── evaluate.py                # 评估 + HTML 可视化报告
│   └── export_for_stm32.py        # ONNX 导出 + 分析
├── docs/
│   └── superpowers/
│       ├── specs/2026-06-15-nut-binary-classifier-design.md
│       └── plans/2026-06-15-nut-binary-classifier-plan.md
├── task_plan.md                   # planning-with-files 任务计划
├── findings.md                    # 研究发现与踩坑记录
├── progress.md                    # 会话进度日志
└── runs/
    ├── detect/train-6/            # COCO8 检测训练
    └── classify/nut-cls/          # 🔆 分类训练（本次新增）
        ├── weights/best.pt        # 最佳模型 (3.0MB)
        ├── weights/last.pt        # 最后一轮
        ├── results.png            # 训练曲线
        ├── results.csv            # 每轮指标
        ├── confusion_matrix.png   # 混淆矩阵
        ├── eval_metrics.json      # 测试集评估指标
        └── visual_report.html     # 🌐 可视化报告
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
6. **ultralytics 分类模型按字母序排列类别**：`ng`(0) → `ok`(1)，写代码时注意不要硬编码类别顺序

## 螺母分类训练 (nut-cls, 2026-06-15)

### 训练命令
```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
results = model.train(
    data='datasets/nut_classification',  # ImageNet 文件夹格式
    epochs=50,
    patience=10,
    imgsz=224,
    batch=32,
    optimizer='AdamW',
    lr0=1e-3,
    cos_lr=True,
    device=0,
    name='nut-cls',
    exist_ok=True,
)
```

### 测试集评估结果
| 指标 | 总体 | OK | NG |
|------|------|----|----|
| Accuracy | 91.67% | - | - |
| Precision | 91.34% | 91.75% | 91.34% |
| Recall | - | 97.67% | 73.65% |
| F1 | 0.8155 | 0.9462 | 0.8155 |

### 已知问题与改进方向
1. **NG 召回率偏低 (73.7%)**：约 26% 缺陷被漏检
   - 原因：OK:NG = 1:3 不均衡 + 数据增强偏强
   - 改进：OK 类别过采样、降低增强强度、调整学习率
2. **过拟合**：train loss → 0.003, val loss 震荡
   - 改进：weight_decay、Dropout
3. **类别映射**：ultralytics 按字母序 `ng`(0) / `ok`(1)，新手易出错

### ONNX 导出
```python
model = YOLO('runs/classify/nut-cls/weights/best.pt')
model.export(format='onnx', imgsz=224, opset=12, simplify=True)
```
导出到 `models/nut_cls_fp32.onnx` (5.5MB)，仅有 Softmax 一个算子 NPU 不兼容（可通过后处理替代）。
