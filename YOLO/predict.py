from ultralytics import YOLO
 
# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-obb.yaml")
 
# Train the model
train_results = model.train(
    data="ultralytics/cfg/datasets/SSDD.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    batch=4,
    workers=0,
    amp=False,
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    pretrained=False,
    verbose=True,
    optimizer="AdamW",
)
 
# Evaluate model performance on the validation set
metrics = model.val()
 