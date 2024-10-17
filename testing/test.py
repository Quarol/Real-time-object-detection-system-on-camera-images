from ultralytics import YOLO

model = YOLO('../yolo_models/yolov8n-pretrained-default.pt')
print(model.model.yaml)