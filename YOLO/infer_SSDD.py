import csv
import json
from pathlib import Path

from ultralytics import YOLO

weights = "/home/changqi/王长启/Code/ASOC1/YOLO/runs/obb/new_test/SSDD/110real+110ours/ssdd-110real+110ours-2026/weights/best.pt"
data_yaml = "/home/changqi/王长启/Code/ASOC1/YOLO/ultralytics/cfg/datasets/SSDD.yaml"

model = YOLO(weights)

metrics = model.val(
    data=data_yaml,
    split="test",      
    imgsz=499,          
    batch=16,
    device=0,
    workers=4,
    conf=0.001,         
    iou=0.82,
    plots=True,
    project="/home/changqi/王长启/Code/ASOC1/YOLO/runs/obb/new_test/SSDD/110real+110ours",
    name="ssdd-110real+110ours-seed2026"
)

print("\nDetection results:")
for key, value in metrics.results_dict.items():
    print(f"{key}: {value:.6f}")

p = metrics.box.mp
r = metrics.box.mr

print(f"Precision: {p:.6f} ({p * 100:.2f}%)")
print(f"Recall:    {r:.6f} ({r * 100:.2f}%)")
print(f"mAP50:     {metrics.box.map50:.6f} ({metrics.box.map50 * 100:.2f}%)")
print(f"mAP50:95:  {metrics.box.map:.6f} ({metrics.box.map * 100:.2f}%)")
print(f"mAP75:     {metrics.box.map75:.6f} ({metrics.box.map75 * 100:.2f}%)")

f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
print(f"F1:         {f1:.6f} ({f1 * 100:.2f}%)")


metrics_save_dir = Path(metrics.save_dir)
metrics_save_dir.mkdir(parents=True, exist_ok=True)

saved_metrics = {
    key: float(value) for key, value in metrics.results_dict.items()
}
saved_metrics.update({
    "Precision": float(p),
    "Recall": float(r),
    "mAP50": float(metrics.box.map50),
    "mAP50:95": float(metrics.box.map),
    "mAP75": float(metrics.box.map75),
    "F1": float(f1),
})

json_path = metrics_save_dir / "metrics.json"
csv_path = metrics_save_dir / "metrics.csv"

with json_path.open("w", encoding="utf-8") as file:
    json.dump(saved_metrics, file, ensure_ascii=False, indent=2)

with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["metric", "value", "percentage"])
    for key, value in saved_metrics.items():
        writer.writerow([key, f"{value:.6f}", f"{value * 100:.2f}%"])

print(f"\nMetrics JSON saved to: {json_path}")
print(f"Metrics CSV saved to:  {csv_path}")
