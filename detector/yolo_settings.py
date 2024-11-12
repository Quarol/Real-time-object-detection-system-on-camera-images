from ultralytics import YOLO
from dataclasses import dataclass


dir = 'yolo_models'
YOLO_MODELS = [
    f'{dir}/yolov8n-pretrained-default.pt',
    f'{dir}/yolov8s-pretrained-default.pt',
    f'{dir}/yolov8s-pretrained-default.pt',
    f'{dir}/yolov8m-pretrained-default.pt',
    f'{dir}/yolov8l-pretrained-default.pt',
    f'{dir}/yolov8x-pretrained-default.pt',
    f'{dir}/yolov11n-pretrained-default.pt'
]


def get_yolo_classes():
    model = YOLO('yolo_models/yolov8n-pretrained-default.pt')
    return model.names


@dataclass
class YoloInferenceConfig:
    confidence_threshold = 0.5
    device = 'cuda'
    classes = [0, 12]
    verbose = False


yolo_classes = get_yolo_classes()
yolo_inference_config = YoloInferenceConfig()


def get_class_name(key: int):
    return yolo_classes[key]