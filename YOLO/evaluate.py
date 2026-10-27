from ultralytics import YOLO

weights = "/home/changqi/王长启/Code/ASOC1/YOLO/runs/obb/hrsid_real110_seed20252/weights/best.pt"
data_yaml = "/home/changqi/王长启/Code/ASOC1/YOLO/ultralytics/cfg/datasets/dataloader.yaml"

model = YOLO(weights)

metrics = model.val(
    data=data_yaml,
    split="test",      
    imgsz=512,          
    batch=16,
    device=0,
    workers=4,
    conf=0.001,         
    iou=0.7,
    plots=True,
    project="/home/changqi/王长启/Code/ASOC1/YOLO/runs/new_110real_2026",
    name="ssdd_ours_test"
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
