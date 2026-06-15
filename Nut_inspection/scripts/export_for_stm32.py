"""
scripts/export_for_stm32.py
将 YOLOv8n-cls best.pt 导出为 ONNX，为 STM32N6570 部署做准备。
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "nut-cls" / "weights" / "best.pt"
MODELS_DIR = PROJECT_ROOT / "models"


def analyze_onnx(onnx_path: Path):
    """分析 ONNX 模型，检查 NPU 兼容性"""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        # 统计算子类型
        ops = {}
        total_params = 0
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1

        # NPU 不友好的算子
        npu_unfriendly = {"SiLU", "GELU", "Softmax", "LayerNormalization", "LSTM", "GRU"}
        bad_ops = {k: v for k, v in ops.items() if k in npu_unfriendly}

        print(f"\n{'=' * 55}")
        print(f"  ONNX 模型分析: {onnx_path.name}")
        print(f"{'=' * 55}")

        # 输入输出
        inp = model.graph.input[0]
        out = model.graph.output[0]
        inp_shape = [d.dim_value if d.dim_value else '?' for d in inp.type.tensor_type.shape.dim]
        out_shape = [d.dim_value if d.dim_value else '?' for d in out.type.tensor_type.shape.dim]
        print(f"  Input : {inp.name}  shape={inp_shape}  type={inp.type.tensor_type.elem_type}")
        print(f"  Output: {out.name}  shape={out_shape}")

        # 模型大小
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        size_kb = onnx_path.stat().st_size / 1024
        print(f"  Size  : {size_mb:.2f} MB ({size_kb:.0f} KB)")

        # 算子统计
        print(f"  Ops   : {sum(ops.values())} total, {len(ops)} unique types")
        for op, count in sorted(ops.items(), key=lambda x: -x[1]):
            flag = " ⚠️ NPU incompatible!" if op in npu_unfriendly else ""
            print(f"    {op:20s}: {count:4d}{flag}")

        if bad_ops:
            print(f"\n  ⚠️  Warning: {sum(bad_ops.values())} NPU-incompatible ops detected!")
            print(f"     These may fall back to CPU on STM32N6570 Neural-ART.")
        else:
            print(f"\n  ✅ No NPU-incompatible ops detected.")

        # 参数量估算
        from onnx import numpy_helper
        for init in model.graph.initializer:
            total_params += numpy_helper.to_array(init).size
        print(f"  Params: {total_params:,} (~{total_params/1e6:.1f}M)")

        return True
    except ImportError:
        print("  [SKIP] onnx not installed, can't analyze.")
        return False


def verify_consistency(onnx_path: Path):
    """对比 PyTorch 和 ONNX 推理结果"""
    print(f"\n{'=' * 55}")
    print(f"  推理一致性验证")
    print(f"{'=' * 55}")

    import numpy as np
    from PIL import Image
    import onnxruntime as ort

    # 加载模型（统一用 CPU 避免 GPU/ONNX 冲突）
    pt_model = YOLO(str(MODEL_PATH))
    onnx_model = YOLO(str(onnx_path), task='classify')

    # 只在 ONNX 模型上跑验证
    test_dir = PROJECT_ROOT / "datasets" / "nut_classification" / "test"
    test_images = sorted(test_dir.rglob("*.jpg"))
    import random; random.seed(42)
    samples = random.sample(test_images, min(20, len(test_images)))

    mismatches = 0
    for img_path in samples:
        # PyTorch 推理
        pt_results = pt_model(str(img_path), verbose=False, device='cpu')
        pt_pred = pt_results[0].probs.top1
        pt_conf = pt_results[0].probs.top1conf.item()

        # ONNX 推理
        onnx_results = onnx_model(str(img_path), verbose=False, device='cpu')
        onnx_pred = onnx_results[0].probs.top1
        onnx_conf = onnx_results[0].probs.top1conf.item()

        if pt_pred != onnx_pred:
            mismatches += 1
            print(f"  MISMATCH: {img_path.name}: PT={pt_pred}({pt_conf:.3f}) ONNX={onnx_pred}({onnx_conf:.3f})")

    if mismatches == 0:
        print(f"  ✅ 全部一致！20/20 样本 PyTorch ≡ ONNX")
    else:
        print(f"  ⚠️ {mismatches}/20 不一致")

    return mismatches == 0


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)

    # Step 1: 加载模型
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"  Classes: {model.names}")

    # Step 2: 导出 ONNX (FP32)
    print(f"\nExporting to ONNX (opset=12, simplify=True)...")
    onnx_path = model.export(
        format='onnx',
        imgsz=224,
        opset=12,
        simplify=True,
        half=False,  # FP32，为后续 int8 量化校准保留精度
    )

    # 移动到 models/
    src = Path(onnx_path)
    dst = MODELS_DIR / "nut_cls_fp32.onnx"
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    print(f"  Saved to: {dst}")

    # Step 3: 分析 ONNX 模型
    analyze_onnx(dst)

    # Step 4: 验证推理一致性
    verify_consistency(dst)

    print(f"\n{'=' * 55}")
    print(f"  阶段 4 完成!")
    print(f"{'=' * 55}")
    print(f"  ONNX 模型: models/nut_cls_fp32.onnx")
    print(f"  下一步:    STM32 X-CUBE-AI 量化 + C 代码生成")
    print(f"  部署目标:  STM32N6570 Neural-ART NPU")
    print(f"{'=' * 55}")
