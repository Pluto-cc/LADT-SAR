from ultralytics import YOLO
 
# Load a model----加载自己训练好的权重
model = YOLO("/home/ubuntu/pycharm_file/ultralytics-main/runs/obb/train/weights/epoch90.pt")
 
# Customize validation settings
validation_results = model.val(data="ultralytics/cfg/datasets/SSDD.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
 
validation_results.box.map  # map50-95
validation_results.box.map50  # map50
validation_results.box.map75  # map75
validation_results.box.map90  # map75
validation_results.box.maps  # a list contains map50-95 of each category
 