# 研究发现

## NUT.v2 数据集分析

### 数据集信息
- 位置：`Nuts.v2(DST1506)/`
- 总图片：5880 张（train 3360 / val 1260 / test 1260）
- 标注格式：YOLO 检测格式（class_id cx cy w h，归一化坐标）
- 包含 VOC 格式备份

### 类别详情（8 类）
| Class ID | 名称 | 含义 |
|----------|------|------|
| 0 | Excellent | 合格（正面） |
| 1 | Side_Excellent | 合格（侧面） |
| 2 | Rust | 生锈 |
| 3 | Side_Rust | 生锈（侧面） |
| 4 | Fracture | 断裂 |
| 5 | Side-Fracture | 断裂（侧面） |
| 6 | Scratches | 划痕 |
| 7 | Side_Scratches | 划痕（侧面） |

### 训练集分布
| 类别 | 图片数 |
|------|--------|
| Excellent (0) | 420 |
| Side_Excellent (1) | 420 |
| **OK 合计** | **840** |
| Rust (2) | 421 |
| Side_Rust (3) | 420 |
| Fracture (4) | 420 |
| Side-Fracture (5) | 420 |
| Scratches (6) | 419 |
| Side_Scratches (7) | 420 |
| **NG 合计** | **2520** |

### 关键发现
1. 数据集非常均衡（每类 ~420 张），但 OK:NG = 1:3 不均衡
2. 每张图只有一个目标（一个螺母），适合转为分类任务
3. 图片命名包含日期戳，疑似产线实际采集
4. VOC 目录包含相同数据的 XML 格式备份
5. 标注框几乎覆盖全图（cx≈0.41, cy≈0.51, w≈0.82, h≈0.77），说明螺母占图主体

## 训练发现 (2026-06-15)
1. **ultralytics 类别排序**：分类模型按文件夹名字母序排列类别，`ng`(0) / `ok`(1)，与直觉相反。评估脚本必须匹配模型顺序。
2. **SSL 证书问题**：Windows 环境下 ultralytics 下载预训练模型时 SSL 验证失败，需用 `curl -k` 手动下载
3. **过拟合**：YOLOv8n-cls 在 3360 张训练图上出现过拟合（train loss → 0.003），需正则化
4. **EarlyStopping**：patience=10 在第 4 轮达到最佳后停止，准确率 86.3%
5. **NG 召回低**：模型偏向保守，26% 缺陷被漏判为 OK
6. **ONNX 分析**：YOLOv8n 使用 Sigmoid 激活函数而非 SiLU，对 STM32 NPU 友好。仅 Softmax 不兼容。

## 优化训练发现 (2026-06-16)

### 数据平衡效果
1. **OK 过采样 3× 有效但带来权衡**：NG 召回率 73.7% → 97.9%，但 OK 召回率 97.7% → 81.6%
   - 平衡训练集后模型不再偏向多数类 (NG)，对两类同等重视
   - 工业场景中漏检（NG→OK）比误报（OK→NG）代价大得多，这个权衡是值得的
2. **过采样 vs class_weight**：v1 使用 class_weight 效果不佳（NG 召回仍低），直接过采样更有效
3. **注意**：val 集也被过采样了，导致 val metrics 不可直接与 v1 对比。公平对比必须在原始 test 集上进行

### 正则化效果
4. **weight_decay + dropout 改善过拟合**：v1 train loss → 0.003, v2 保持在合理水平
5. **label_smoothing 降低过置信**：模型预测概率分布更平滑，不再出现 99%+ 的极端置信度
6. **warmup_epochs 稳定训练初期**：前 3 轮损失下降更平滑
7. **降低增强有效**：工业螺母场景特征相对固定，过度增强反而引入噪声

### 注意事项
8. **`label_smoothing` 参数已被 ultralytics 标记为 deprecated**（v8.4.67），未来版本可能移除
9. **SSL 证书问题依然存在**：AMP checks 下载 yolo26n.pt 时 SSL 失败，不影响训练但建议修复
10. **val 集一致性**：评估时务必使用原始 test 集（datasets/nut_classification/test/），不要用 balanced 版本的 test

## 环境信息
- PyTorch 2.8.0+cu128, Ultralytics 8.4.67
- GPU: RTX 4060 8GB
- 此前用 COCO8 验证过 YOLOv8n 训练管线正常

## 部署目标
- STM32N6570 (Cortex-M55 + Neural-ART NPU)
- NPU 600 GOPS，支持 int8 量化卷积
- 不支持 SiLU/GELU，导出 ONNX 时需注意
