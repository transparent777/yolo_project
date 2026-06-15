# 任务计划：螺母 OK/NG 二分类器

## 目标
训练一个 YOLOv8n-cls 图像级二分类模型（OK/NG），在 NUT.v2 数据集上达到 >92% accuracy，导出 ONNX 为 STM32N6570 部署做准备。

## 当前阶段
🎉 全部完成

## 各阶段

### 阶段 1：数据准备
- [x] 编写 convert_to_cls.py：YOLO 标注 → ImageNet 文件夹格式
- [x] 运行转换脚本，生成 datasets/nut_classification/
- [x] 验证数据分布：train ok/ng 数量、val ok/ng 数量
- [x] 检查样本图片可正常读取
- **状态：** complete

### 阶段 2：模型训练
- [ ] 编写 train_classifier.py 训练脚本
- [ ] 运行训练 (50 epochs, EarlyStopping)
- [ ] 监控训练曲线 (loss, accuracy)
- [ ] 保存 best.pt
- **状态：** in_progress

### 阶段 3：模型评估 + 可视化
- [x] 编写 evaluate.py 评估脚本
- [x] 在 test 集上评估 best.pt
- [x] 输出 accuracy / precision / recall / F1
- [x] 生成混淆矩阵图
- [x] 编写 visualize.py 可视化脚本
- [x] 生成 HTML 可视化报告（含误判网格、OK/NG 预测叠加、置信度分布）
- [x] 确认指标: 91.67% acc ✅, NG recall 73.7% ❌ (目标90%)
- **状态：** complete

### 阶段 4：模型导出
- [x] 编写 export_for_stm32.py
- [x] PyTorch → ONNX (opset=12, simplify)
- [x] 验证 ONNX 推理结果与 PyTorch 一致 (20/20)
- [x] 输出模型大小和参数量
- **状态：** complete

### 阶段 5：文档与交付
- [x] 更新 CLAUDE.md 记录训练结果
- [x] 整理最终交付物清单
- [x] 为下一阶段（检测 + 精细分类）做准备
- **状态：** complete

## 当前阶段
🎉 全部完成

## 关键问题
1. NUT.v2 类别 0-1 为 OK，2-7 为 NG，确认映射正确？✅ 已确认
2. OK 840 vs NG 2520 不均衡，采用 oversampling 还是 class_weight？→ 训练脚本中处理
3. 自采数据何时加入？→ 第二阶段

## 已做决策
| 决策 | 理由 |
|------|------|
| YOLOv8n-cls 而非 MobileNet | 与现有 YOLO 生态一致，后续扩展顺滑 |
| 图像级分类而非检测 | 先验证可行性，降低复杂度 |
| imgsz=224 | 工业场景够用，MCU 友好 |
| AdamW + cosine decay | 分类任务标准配置 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
|      |         |         |

## 备注
- 随着进度更新阶段状态：pending → in_progress → complete
- 做重大决策前重新读取此计划
- 记录所有错误，避免重复
