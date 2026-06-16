# 进度日志

## 2026-06-15

### 会话摘要
- 安装 superpowers 和 planning-with-files 两个 skills
- 完成项目头脑storming，确定技术方案
- 分析 NUT.v2 数据集（5880 张，8 类，YOLO 格式）
- 编写设计文档 `docs/superpowers/specs/2026-06-15-nut-binary-classifier-design.md`
- 创建任务计划 `task_plan.md`、`findings.md`、`progress.md`
- 配置 planning-with-files hooks（UserPromptSubmit/PostToolUse/PreCompact）
- 更新 CLAUDE.md 记录层策略 + 数据来源
- 设计可视化系统，新增到阶段 3

### 用户确认的四项要求
1. ✅ NUT.v2 为唯一数据源，datasets/ 为工作数据
2. ✅ planning-with-files hooks 已启用（settings.json UserPromptSubmit + PostToolUse + PreCompact）
3. ✅ CLAUDE.md 每次训练后手动同步更新（三层记录：CLAUDE.md / progress.md / findings.md）
4. ⬜ 可视化页面：设计完成，待实现（见下方设计）

### 当前状态
- ✅ 环境配置完成（yolo_env, PyTorch CUDA）
- ✅ 数据集就绪（NUT.v2）
- ✅ 技术方案确定（YOLOv8n-cls 二分类 → STM32）
- ✅ Hooks 配置完成
- ✅ 阶段 1 完成：数据转换 (5880 张)
- ✅ 阶段 2 完成：模型训练 (best 86.3% val acc)
- ✅ 阶段 3 完成：评估 + 可视化 (91.7% test acc, visual_report.html)
- ✅ 阶段 4 完成：ONNX 导出 (5.5MB, 推理一致)
- ✅ 阶段 5 完成：CLAUDE.md 更新, 所有文档就绪

### 最终交付物
| 文件 | 说明 |
|------|------|
| models/nut_cls_fp32.onnx | v1 STM32 部署用 ONNX 模型 |
| models/nut_cls_v2_fp32.onnx | 🔆 v2 优化版 ONNX 模型 |
| runs/classify/nut-cls/weights/best.pt | v1 PyTorch 最佳模型 |
| runs/classify/nut-cls-v2/weights/best.pt | 🔆 v2 PyTorch 最佳模型 |
| runs/classify/nut-cls/visual_report.html | v1 可视化评估报告 |
| scripts/convert_to_cls.py | 数据转换 |
| scripts/balance_dataset.py | 🔆 数据平衡脚本 |
| scripts/train_classifier.py | v1 训练脚本 |
| scripts/train_classifier_v2.py | 🔆 v2 优化训练脚本 |
| scripts/evaluate.py | 评估 + 可视化 |
| scripts/export_for_stm32.py | ONNX 导出 |

## 2026-06-16

### 会话摘要：优化训练 nut-cls-v2
- 分析 v1 问题：NG 召回率仅 73.7%，过拟合严重
- 制定优化方案：数据平衡 + 正则化 + 降增强
- 创建 `scripts/balance_dataset.py`：OK 过采样 3× (train: 2520+2520)
- 创建 `scripts/train_classifier_v2.py`：降低 lr、加 weight_decay/dropout/label_smoothing/warmup
- 运行 nut-cls-v2 训练 (53/80 epochs, EarlyStopping @ epoch 33)
- 在原始 test 集评估并与 v1 对比

### 优化训练结果对比
| 指标 | v1 (基线) | v2 (优化) | 变化 |
|------|-----------|-----------|------|
| Accuracy | 91.67% | 93.81% | +2.1% |
| NG Recall | 73.65% | 97.88% | +24.2% |
| OK Recall | 97.67% | 81.59% | -16.1% |
| NG Precision | 91.34% | 94.10% | +2.8% |
| 训练轮次 | 4 (ES) | 33 (ES) | 充分训练 |

### 核心结论
- NG 召回率大幅提升至 97.9%，缺陷漏检率从 26% 降至 2.1% ✅
- OK 召回率下降至 81.6%（模型变保守），但工业场景漏检代价远大于误报
- 过拟合改善，训练更稳定
- ONNX 已导出至 models/nut_cls_v2_fp32.onnx

### 当前状态
- ✅ 阶段 1-5 全部完成
- ✅ 优化训练 nut-cls-v2 完成
- ✅ ONNX v2 导出完成
- ⬜ 阈值调优（可选）：平衡 OK/NG 召回率
- ⬜ STM32N6570 部署验证
