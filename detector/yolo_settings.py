from ultralytics import YOLO

class YoloInferenceConfig:
    def __init__(self) -> None:
        self.confidence_threshold = 0.5
        self.device = 'cuda'
        self.classes = [0] # people by default
        self.verbose = False

    def add_detected_class(self, object_index: int) -> None:
        self.classes.append(object_index)

    def remove_detected_class(self, object_index: int) -> None:
        self.classes.remove(object_index)

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold 
    
YOLO_CLASSES = {}
yolo_inference_config = YoloInferenceConfig()

def get_class_name(index: int) -> str:
    return YOLO_CLASSES[index]