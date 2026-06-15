# Nut Binary Classifier — Design Spec

> 基于 YOLOv8n-cls 的螺母 OK/NG 二分类器，最终部署到 STM32N6570

## 目标

训练一个图像级二分类模型，判断螺母图片为「合格(OK)」或「缺陷(NG)」，作为螺母检测项目的第一阶段验证。

## 约束条件

| 项目 | 说明 |
|------|------|
| 任务类型 | 图像级二分类（不做检测框） |
| 部署目标 | STM32N6570 (Neural-ART NPU, 600 GOPS) |
| 训练环境 | RTX 4060 8GB, yolo_env |
| 数据来源 | NUT.v2 公开数据集 + 自采补充 |
| 后续扩展 | 第二阶段加入检测头 + 精细分类 |

## 架构

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│ 数据转换   │ →  │ 模型训练   │ →  │ 模型评估   │ →  │ 导出 + 量化 │
│ YOLO→cls  │    │ YOLOv8n  │    │ test集   │    │ ONNX int8  │
└──────────┘    └──────────┘    └──────────┘    └───────────┘
```

## 数据策略

### NUT.v2 数据集（5880 张，8 类）

数据集位于 `Nuts.v2(DST1506)/`，YOLO 检测标注格式。

#### 二分类映射

| 二分类标签 | 原始类别 |
|-----------|---------|
| **OK** (0) | Excellent(0), Side_Excellent(1) |
| **NG** (1) | Rust(2), Side_Rust(3), Fracture(4), Side-Fracture(5), Scratches(6), Side_Scratches(7) |

#### 数据分布 (train)

| 类别 | 数量 |
|------|------|
| OK | 840 |
| NG | 2520 |
| 总计 | 3360 |

> NG 是 OK 的 3 倍，训练时使用 class_weights 或 oversampling 处理不均衡。

### 数据转换脚本

`scripts/convert_to_cls.py`：读取 YOLO .txt 标注 → 按 class_id 分 OK/NG → 复制图片到 `datasets/nut_classification/{train,val}/{ok,ng}/`

## 模型

### 选型：YOLOv8n-cls

- Backbone: YOLOv8n backbone（复用 ImageNet 预训练）
- Head: GlobalAvgPool + FC(2)
- 参数量: ~2.7M（int8 量化后 <1MB）

## 训练参数

| 参数 | 值 |
|------|-----|
| model | yolov8n-cls.pt |
| epochs | 50 (patience=10) |
| imgsz | 224 |
| batch | 32 |
| optimizer | AdamW |
| lr0 | 1e-3, cosine decay |
| augment | hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, fliplr=0.5 |

## 评估标准

| 指标 | 目标 |
|------|------|
| Accuracy (Top-1) | > 92% |
| Precision (NG) | > 90% |
| Recall (NG) | > 90% |
| F1-Score | > 0.90 |

## STM32N6570 部署管线

```
best.pt → ONNX export (opset=12, simplify)
        → INT8 量化校准（val 集校准）
        → STM32 X-CUBE-AI → C 代码 → STM32CubeMX 集成
```

> 注意：导出 ONNX 时避免 SiLU/GELU 等 NPU 不友好的算子，优先使用 ReLU 替换。

## 项目文件结构

```
Nut_inspection/
├── Nuts.v2(DST1506)/            # 原始数据集（不动）
├── datasets/
│   └── nut_classification/      # 转换后的分类数据
│       ├── train/{ok,ng}/
│       └── val/{ok,ng}/
├── models/
│   ├── nut_cls_best.pt
│   └── nut_cls_int8.onnx
├── scripts/
│   ├── convert_to_cls.py        # 数据格式转换
│   ├── train_classifier.py      # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── export_for_stm32.py      # ONNX 导出 + 量化
├── docs/superpowers/
│   ├── specs/2026-06-15-nut-binary-classifier-design.md
│   └── plans/
└── runs/classify/               # 训练输出
```

## 测试策略

- 单元测试：convert_to_cls.py 的数据转换正确性
- 集成测试：训练脚本跑通 3 个 epoch 确认无错
- 验收测试：test 集 accuracy > 92%
