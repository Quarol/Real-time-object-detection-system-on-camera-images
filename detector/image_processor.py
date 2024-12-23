import cv2 as cv
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import Tuple

from detector.yolo_settings import YoloInferenceConfig

class ImageProcessor(YoloInferenceConfig):
    def __init__(self) -> None:
        super().__init__()
        self._detector = YOLO('yolo_models/yolov8n.pt')
        available_classes: dict = self._detector.names
        self._all_classes = list(available_classes.values())


    def detect_objects(self, frame: MatLike) -> Results:
        results = self._detector.predict(
            frame,
            conf=self._confidence_threshold,
            device=self._device,
            classes=self._classes,
            max_det=self._max_det,
            verbose=self._verbose
        ) # Returns list of output frames
        first_frame_result = results[0] # Get first (and only) frame

        return first_frame_result, len(first_frame_result.boxes) > 0 


    def visualize_objects_presence(self, frame: MatLike, detections: Results) -> Tuple[MatLike, bool]:
        for box in detections.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            object_class_id = self._get_object_class_id(box) 
            object_class_name = self._all_classes[object_class_id]

            label_position = (int(x_min), int(y_min) - 10)
            cv.putText(
                frame,                     
                object_class_name,         
                label_position,            
                cv.FONT_HERSHEY_SIMPLEX,   
                0.5,                       
                (0, 255, 0),               
                2,                         
                cv.LINE_AA                 
            )

        return frame
    

    def _get_object_class_id(self, box) -> int:
        return int(box.cls[0])


    def fit_frame_into_screen(self, frame: MatLike, max_frame_width, max_frame_height) -> MatLike:
        if frame is None:
            return None
        
        output_width, output_height = self._fitting_dimensions(frame, max_frame_width, max_frame_height)
        frame = cv.resize(frame, (output_width, output_height), interpolation=cv.INTER_LINEAR)
        return frame


    def _fitting_dimensions(self, frame: MatLike, max_width: int, max_height: int) -> Tuple[int, int]:
        height, width = frame.shape[:2]

        if width > max_width or height > max_height:
            scale_x = max_width / width
            scale_y = max_height / height

            scale = min(scale_x, scale_y)

            new_width = int(width * scale)
            new_height = int(height * scale)

            return new_width, new_height

        return width, height