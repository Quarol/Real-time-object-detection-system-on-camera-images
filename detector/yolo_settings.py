class YoloInferenceConfig:
    def __init__(self) -> None:
        self._confidence_threshold = 0.5
        self._device = 'cuda'
        self._classes = [0] # people by default
        self._max_det = 50
        self._verbose = False

        self._all_classes = None
    
    def get_available_classes(self) -> list[str]:
        return self._all_classes

    def get_detected_classes(self) -> list[str]:
        return self._classes
    def add_detected_class(self, class_index: int) -> None:
        self._classes.append(class_index)
    def remove_detected_class(self, class_index: int) -> None:
        self._classes.remove(class_index)

    def get_confidence_threshold(self) -> float:
        return self._confidence_threshold
    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        self._confidence_threshold = confidence_threshold