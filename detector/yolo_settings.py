class YoloInferenceConfig:
    def __init__(self) -> None:
        self.confidence_threshold = 0.5
        self.device = 'cuda'
        self.classes = [0] # people by default
        self.max_det = 50
        self.verbose = False

        self._all_classes = None

    def set_available_classes(self, classes: dict) -> None:
        self._all_classes = list(classes.values())

    def get_class_name(self, index: int) -> str:
        return self._all_classes[index]
    
    def get_available_classes(self) -> list[str]:
        return self._all_classes

    def add_detected_class(self, object_index: int) -> None:
        self.classes.append(object_index)

    def remove_detected_class(self, object_index: int) -> None:
        self.classes.remove(object_index)

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self.confidence_threshold = confidence_threshold

yolo_inference_config = YoloInferenceConfig()