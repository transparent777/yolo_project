"""
scripts/convert_to_cls.py
将 NUT.v2 YOLO 检测标注转换为分类数据集文件夹格式。

映射规则:
  OK (合格): class 0 (Excellent), class 1 (Side_Excellent)
  NG (缺陷): class 2 (Rust), class 3 (Side_Rust), class 4 (Fracture),
             class 5 (Side-Fracture), class 6 (Scratches), class 7 (Side_Scratches)

输入: Nuts.v2(DST1506)/images/{train,valid,test}/ + labels/
输出: datasets/nut_classification/{train,val,test}/{ok,ng}/
"""
import shutil
from pathlib import Path

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "Nuts.v2(DST1506)"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "nut_classification"

# 二分类映射
OK_CLASSES = {0, 1}       # Excellent, Side_Excellent
NG_CLASSES = {2, 3, 4, 5, 6, 7}  # Rust → Side_Scratches

# split 名称映射: NUT.v2 用 "valid"，我们输出用 "val"
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def get_label_class(label_path: Path) -> int:
    """读取 YOLO .txt 标注文件，返回第一个目标的 class_id。
    格式: class_id cx cy w h (归一化坐标)
    每张图只有一个螺母目标，只取第一行。
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return -1

    if not lines:
        return -1

    line = lines[0].strip()
    if not line:
        return -1

    try:
        class_id = int(line.split()[0])
        return class_id
    except (ValueError, IndexError):
        return -1


def convert_split(nutv2_split: str):
    """转换一个 split (train / valid / test)"""
    src_images = DATASET_ROOT / "images" / nutv2_split
    src_labels = DATASET_ROOT / "labels" / nutv2_split
    out_split = SPLIT_MAP.get(nutv2_split, nutv2_split)

    if not src_images.exists():
        print(f"  [SKIP] 源目录不存在: {src_images}")
        return

    ok_count = 0
    ng_count = 0
    skip_count = 0
    error_count = 0

    for img_file in sorted(src_images.iterdir()):
        # 只处理图片文件
        if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue

        # 找到对应的标注文件
        label_file = src_labels / f"{img_file.stem}.txt"
        if not label_file.exists():
            skip_count += 1
            continue

        # 判断类别
        cls_id = get_label_class(label_file)
        if cls_id in OK_CLASSES:
            dst_dir = OUTPUT_ROOT / out_split / "ok"
            ok_count += 1
        elif cls_id in NG_CLASSES:
            dst_dir = OUTPUT_ROOT / out_split / "ng"
            ng_count += 1
        elif cls_id == -1:
            error_count += 1
            continue
        else:
            skip_count += 1
            continue

        # 复制图片
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(img_file, dst_dir / img_file.name)
        except Exception as e:
            print(f"  [ERROR] 复制失败: {img_file.name} - {e}")
            error_count += 1

    print(f"  {nutv2_split} -> {out_split}: OK={ok_count}, NG={ng_count}, skip={skip_count}, error={error_count}")


def print_summary():
    """打印数据集统计"""
    print("\n" + "=" * 55)
    print("  数据集统计: datasets/nut_classification/")
    print("=" * 55)
    grand_ok = 0
    grand_ng = 0

    for split in ["train", "val", "test"]:
        ok_dir = OUTPUT_ROOT / split / "ok"
        ng_dir = OUTPUT_ROOT / split / "ng"
        ok_n = len(list(ok_dir.glob("*.jpg"))) + len(list(ok_dir.glob("*.png"))) if ok_dir.exists() else 0
        ng_n = len(list(ng_dir.glob("*.jpg"))) + len(list(ng_dir.glob("*.png"))) if ng_dir.exists() else 0
        grand_ok += ok_n
        grand_ng += ng_n
        ratio = f"OK:NG = 1:{ng_n / ok_n:.1f}" if ok_n > 0 else "OK=0"
        print(f"  {split:6s}: OK={ok_n:5d}  NG={ng_n:5d}  total={ok_n + ng_n:5d}  ({ratio})")

    print(f"  {'总计':6s}: OK={grand_ok:5d}  NG={grand_ng:5d}  total={grand_ok + grand_ng:5d}")
    print("=" * 55)


if __name__ == "__main__":
    print("Converting Nuts.v2(DST1506) → datasets/nut_classification/")
    print(f"  数据源: {DATASET_ROOT}")
    print(f"  输出:   {OUTPUT_ROOT}")
    print()

    # 清空旧输出（如果存在）
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
        OUTPUT_ROOT.mkdir(parents=True)

    # 转换所有 split
    for nutv2_split in ["train", "valid", "test"]:
        convert_split(nutv2_split)

    # 打印统计
    print_summary()
