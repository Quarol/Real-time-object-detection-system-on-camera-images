from ultralytics import YOLO
from dataclasses import dataclass

def get_yolo_classes():
    model = YOLO('yolo_models/yolov8n-pretrained-default.pt')
    return model.names

@dataclass
class YoloConfig:
    confidence_threshold = 0.5
    device = 'cuda'
    classes = [0] # people by default
    verbose = False


yolo_classes = get_yolo_classes()
yolo_config = YoloConfig()


def get_class_name(key: int):
    return yolo_classes[key]