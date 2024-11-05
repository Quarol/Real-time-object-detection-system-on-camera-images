from dataclasses import dataclass

@dataclass
class YoloConfig:
    confidence_threshold = 0.5
    device = 'cuda'
    verbose = False

yolo_config = YoloConfig()