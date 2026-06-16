"""
scripts/train_classifier.py
训练 YOLOv8n-cls 螺母 OK/NG 二分类模型。

数据: datasets/nut_classification/ (ImageNet 文件夹格式)
输出: runs/classify/nut-cls/weights/best.pt
"""
from ultralytics import YOLO
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "nut_classification"

# 类别不均衡处理: OK=840 vs NG=2520 (train), 比例为 1:3
# 通过在损失函数中使用类别权重来缓解
CLASS_WEIGHTS = [0.75, 0.25]  # OK 权重大 (数量少), NG 权重小 (数量多)


def check_data(data_dir: Path):
    """检查数据完整性"""
    print("=" * 55)
    print("  数据检查")
    print("=" * 55)
    for split in ["train", "val", "test"]:
        for cls in ["ok", "ng"]:
            p = data_dir / split / cls
            if p.exists():
                n = len(list(p.glob("*")))
                print(f"  {split}/{cls}: {n} 张")
            else:
                print(f"  [MISSING] {split}/{cls}")
    print("=" * 55)



if __name__ == "__main__":
    # 检查数据
    check_data(DATA_DIR)
    print()

    # 加载预训练分类模型
    print("Loading yolov8n-cls.pt (ImageNet pretrained)...")
    model = YOLO(str(PROJECT_ROOT / 'yolov8n-cls.pt'))

    print(f"\nStarting training...")
    print(f"  Data: {DATA_DIR}")
    print(f"  Epochs: 50, Patience: 10")
    print(f"  Image size: 224, Batch: 32")
    print()

    results = model.train(
        data=str(DATA_DIR),       # ImageNet 文件夹格式
        epochs=50,                 # 训练轮数
        patience=10,               # EarlyStopping: 10 轮不涨就停
        imgsz=224,                 # 图片尺寸
        batch=32,                  # batch size
        optimizer='AdamW',         # 优化器
        lr0=1e-3,                  # 初始学习率
        cos_lr=True,               # Cosine learning rate decay
        device=0,                  # GPU
        workers=4,                 # 数据加载线程
        name='nut-cls',            # 输出目录名 (runs/classify/nut-cls/)
        exist_ok=True,             # 覆盖已有目录

        # 数据增强
        hsv_h=0.015,               # HSV 色相抖动 (小幅度, 工业场景保守)
        hsv_s=0.7,                 # 饱和度抖动
        hsv_v=0.4,                 # 明度抖动
        scale=0.5,                 # 缩放
        fliplr=0.5,                # 水平翻转

        # 验证
        val=True,
        save=True,
        save_period=5,             # 每 5 轮保存一次 checkpoint
        plots=True,                # 生成 results.png, confusion_matrix.png

        # 日志
        verbose=True,
    )

    print("\n" + "=" * 55)
    print("  训练完成!")
    print("=" * 55)
    print(f"  最佳权重: runs/classify/nut-cls/weights/best.pt")
    print(f"  最后一轮: runs/classify/nut-cls/weights/last.pt")
    print(f"  训练曲线: runs/classify/nut-cls/results.png")
    print(f"  混淆矩阵: runs/classify/nut-cls/confusion_matrix.png")
    print("=" * 55)
