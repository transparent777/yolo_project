"""
scripts/evaluate.py
在 test 集上评估训练好的分类模型，输出指标 + 可视化报告。
"""
import json
import base64
import io
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from matplotlib import pyplot as plt

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "nut-cls" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "datasets" / "nut_classification" / "test"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "classify" / "nut-cls"
VIZ_DIR = OUTPUT_DIR / "predictions"

CLASS_NAMES = ["ng", "ok"]  # ultralytics 按字母序: ng(0), ok(1)


def load_model():
    """加载最佳模型"""
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    # 确认模型类别顺序
    print(f"  Model classes: {model.names}")
    return model


def run_inference(model):
    """在 test 集所有图片上运行推理"""
    print(f"\nRunning inference on test set: {TEST_DIR}")

    # 收集所有 test 图片
    ok_images = sorted((TEST_DIR / "ok").glob("*"))
    ng_images = sorted((TEST_DIR / "ng").glob("*"))

    all_results = []  # list of dict: {path, true_label, pred_label, confidence}

    for img_path in ok_images + ng_images:
        # 模型类别: 0=ng, 1=ok (字母序)
        folder_name = img_path.parent.name.lower()
        true_label = 0 if folder_name == "ng" else 1

        # YOLO classify predict
        results = model(str(img_path), verbose=False)
        probs = results[0].probs  # Probs object
        if probs is not None:
            pred_label = probs.top1
            confidence = probs.top1conf.item()
        else:
            pred_label = -1
            confidence = 0.0

        all_results.append({
            "path": str(img_path),
            "name": img_path.name,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
        })

    print(f"  Total: {len(all_results)} images")
    return all_results


def compute_metrics(results):
    """计算评估指标"""
    y_true = [r["true_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]
    y_conf = [r["confidence"] for r in results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # 每类 precision/recall/f1
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "OK": {
                "precision": round(float(prec_per_class[0]), 4),
                "recall": round(float(rec_per_class[0]), 4),
                "f1": round(float(f1_per_class[0]), 4),
            },
            "NG": {
                "precision": round(float(prec_per_class[1]), 4),
                "recall": round(float(rec_per_class[1]), 4),
                "f1": round(float(f1_per_class[1]), 4),
            },
        },
        "total_samples": len(results),
    }

    print("\n" + "=" * 55)
    print("  📊 测试集评估结果")
    print("=" * 55)
    print(f"  Accuracy : {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall   : {metrics['recall']:.2%}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    for cls_name, cls_metrics in metrics["per_class"].items():
        print(f"  {cls_name}: P={cls_metrics['precision']:.2%}  R={cls_metrics['recall']:.2%}  F1={cls_metrics['f1']:.4f}")

    print("=" * 55)

    # 保存 JSON
    json_path = OUTPUT_DIR / "eval_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {json_path}")

    return metrics


def generate_prediction_overlay(img_path, label, confidence, is_correct, size=(320, 320)):
    """生成带预测标签叠加的图片"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)

    draw = ImageDraw.Draw(img)

    # 标签颜色: 绿色=正确, 红色=错误
    if is_correct:
        border_color = (0, 200, 0)  # green
        bg_color = (0, 150, 0, 180)
    else:
        border_color = (220, 0, 0)  # red
        bg_color = (180, 0, 0, 180)

    # 边框
    for i in range(4):
        draw.rectangle([i, i, size[0] - 1 - i, size[1] - 1 - i], outline=border_color)

    # 底部标签背景
    draw.rectangle([0, size[1] - 55, size[0], size[1]], fill=bg_color)

    # 文字
    status = "✓" if is_correct else "✗"
    draw.text((8, size[1] - 50), f"{CLASS_NAMES[label]}  {confidence:.1%}  {status}",
              fill=(255, 255, 255))

    return img


def img_to_base64(img):
    """PIL Image → base64 data URI"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def generate_html_report(results, metrics, output_path):
    """生成自包含 HTML 可视化报告"""
    print(f"\nGenerating HTML report...")

    # 分离正确和错误预测
    correct = [r for r in results if r["true_label"] == r["pred_label"]]
    incorrect = [r for r in results if r["true_label"] != r["pred_label"]]

    # 按置信度排序
    correct.sort(key=lambda x: x["confidence"], reverse=True)
    incorrect.sort(key=lambda x: x["confidence"], reverse=True)

    # 生成预测画廊图片 (前 20 张正确 + 全部错误)
    gallery_images = []
    for r in correct[:12]:
        img = generate_prediction_overlay(r["path"], r["pred_label"], r["confidence"], is_correct=True)
        gallery_images.append({"b64": img_to_base64(img), "name": r["name"],
                                "true": CLASS_NAMES[r["true_label"]], "pred": CLASS_NAMES[r["pred_label"]],
                                "conf": r["confidence"], "correct": True})

    misclassified_images = []
    for r in incorrect:
        img = generate_prediction_overlay(r["path"], r["pred_label"], r["confidence"], is_correct=False)
        misclassified_images.append({"b64": img_to_base64(img), "name": r["name"],
                                      "true": CLASS_NAMES[r["true_label"]], "pred": CLASS_NAMES[r["pred_label"]],
                                      "conf": r["confidence"], "correct": False})

    # 置信度分布数据
    ok_correct_conf = [r["confidence"] for r in correct if r["true_label"] == 0]
    ng_correct_conf = [r["confidence"] for r in correct if r["true_label"] == 1]
    incorrect_conf = [r["confidence"] for r in incorrect]

    # 生成 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔩 螺母 OK/NG 分类 - 模型评估报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
.header {{ text-align: center; padding: 30px; background: linear-gradient(135deg, #1e3a5f, #0f172a); border-radius: 16px; margin-bottom: 24px; border: 1px solid #334155; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header p {{ color: #94a3b8; font-size: 14px; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
.metric-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; }}
.metric-card .value {{ font-size: 32px; font-weight: 700; }}
.metric-card .label {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
.metric-card.green .value {{ color: #4ade80; }}
.metric-card.blue .value {{ color: #60a5fa; }}
.metric-card.yellow .value {{ color: #facc15; }}
.metric-card.red .value {{ color: #f87171; }}
.section {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
.section h2 {{ font-size: 20px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
.gallery img {{ width: 100%; border-radius: 6px; border: 1px solid #475569; }}
.gallery .item {{ position: relative; }}
.gallery .caption {{ font-size: 11px; text-align: center; margin-top: 4px; color: #94a3b8; }}
.gallery .correct .caption {{ color: #4ade80; }}
.gallery .incorrect .caption {{ color: #f87171; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
th {{ color: #94a3b8; font-weight: 600; }}
.summary-row {{ display: flex; gap: 16px; margin-bottom: 16px; }}
.summary-item {{ background: #0f172a; border-radius: 8px; padding: 12px 18px; font-size: 14px; }}
.summary-item .num {{ font-size: 22px; font-weight: 700; color: #f87171; }}
.summary-item.good .num {{ color: #4ade80; }}
footer {{ text-align: center; padding: 16px; color: #475569; font-size: 12px; }}
</style>
</head>
<body>

<div class="header">
<h1>🔩 螺母 OK/NG 分类 - 模型评估报告</h1>
<p>模型: YOLOv8n-cls (best.pt) | 测试集: NUT.v2 test (1260 张) | 图片尺寸: 224×224</p>
</div>

<div class="metrics-grid">
<div class="metric-card green">
    <div class="value">{metrics['accuracy']:.1%}</div>
    <div class="label">Accuracy 准确率</div>
</div>
<div class="metric-card blue">
    <div class="value">{metrics['precision']:.1%}</div>
    <div class="label">Precision 精确率</div>
</div>
<div class="metric-card yellow">
    <div class="value">{metrics['recall']:.1%}</div>
    <div class="label">Recall 召回率</div>
</div>
<div class="metric-card red">
    <div class="value">{metrics['f1']:.4f}</div>
    <div class="label">F1-Score</div>
</div>
</div>

<div class="section">
<h2>📋 每类指标</h2>
<table>
<tr><th>类别</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
"""
    for cls_name, m in metrics["per_class"].items():
        html += f"<tr><td>{cls_name}</td><td>{m['precision']:.2%}</td><td>{m['recall']:.2%}</td><td>{m['f1']:.4f}</td></tr>\n"

    html += f"""
</table>
</div>

<div class="section">
<h2>🖼️ 预测画廊 (正确预测样本)</h2>
<div class="summary-row">
    <div class="summary-item good"><span class="num">{len(correct)}</span> 正确预测</div>
</div>
<div class="gallery">
"""
    for img in gallery_images:
        html += f"""<div class="item correct">
    <img src="data:image/jpeg;base64,{img['b64']}" alt="{img['name']}">
    <div class="caption">真实:{img['true']} → 预测:{img['pred']} ({img['conf']:.1%})</div>
</div>
"""

    html += f"""
</div>
</div>

<div class="section">
<h2>❌ 误判焦点 ({len(incorrect)} 个错误预测)</h2>
<div class="summary-row">
    <div class="summary-item"><span class="num">{len(incorrect)}</span> 误判样本</div>
    <div class="summary-item">假阳性 (OK→NG): {sum(1 for r in incorrect if r['true_label']==0)} 个</div>
    <div class="summary-item">假阴性 (NG→OK): {sum(1 for r in incorrect if r['true_label']==1)} 个</div>
</div>
<div class="gallery">
"""
    for img in misclassified_images:
        html += f"""<div class="item incorrect">
    <img src="data:image/jpeg;base64,{img['b64']}" alt="{img['name']}">
    <div class="caption">真实:{img['true']} → 预测:{img['pred']} ({img['conf']:.1%})</div>
</div>
"""

    html += f"""
</div>
</div>

<footer>
生成时间: 2026-06-15 | YOLOv8n-cls Nut Classifier | 1.44M params | Test samples: {metrics['total_samples']}
</footer>

</body>
</html>"""

    # 写文件
    report_path = OUTPUT_DIR / "visual_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report saved to: {report_path}")
    print(f"  Gallery images: {len(gallery_images)} correct + {len(misclassified_images)} misclassified")
    return report_path


if __name__ == "__main__":
    VIZ_DIR.mkdir(exist_ok=True)

    # Step 1: 加载模型
    model = load_model()

    # Step 2: 推理
    results = run_inference(model)

    # Step 3: 计算指标
    metrics = compute_metrics(results)

    # Step 4: 生成 HTML 报告
    report_path = generate_html_report(results, metrics, OUTPUT_DIR)

    # Step 5: 完成
    print(f"\n{'=' * 55}")
    print(f"  阶段 3 完成!")
    print(f"{'=' * 55}")
    print(f"  📊 指标: runs/classify/nut-cls/eval_metrics.json")
    print(f"  📈 曲线: runs/classify/nut-cls/results.png")
    print(f"  🔀 混淆: runs/classify/nut-cls/confusion_matrix.png")
    print(f"  🌐 报告: runs/classify/nut-cls/visual_report.html")
    print(f"{'=' * 55}")
