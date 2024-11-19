import cv2 as cv
from cv2.typing import MatLike
from ultralytics import YOLO
import ultralytics as ul
from typing import Tuple, Optional

from detector.yolo_settings import yolo_inference_config
import detector.yolo_settings as yolo_settings

WARM_UP_IMAGE = 'demo_assets/people.jpg'

class ImageProcessor:
    def __init__(self) -> None:
        self.set_detection_model('yolov8n-pretrained-default.pt')
    

    def set_detection_model(self, name: str) -> None:
        self._detector = YOLO(f'yolo_models/{name}')
        yolo_settings.YOLO_CLASSES = self._detector.names
        self.detect_objects(WARM_UP_IMAGE) # Warmp up to initialize


    def set_confidence_threshold(confidence_threshold) -> None:
        yolo_inference_config.confidence_threshold = confidence_threshold


    def detect_objects(self, frame: MatLike):
        results = self._detector.predict(
            frame,
            conf=yolo_inference_config.confidence_threshold,
            device=yolo_inference_config.device,
            classes=yolo_inference_config.classes,
            verbose=yolo_inference_config.verbose
        ) # Returns list of output frames
        first_frame_result = results[0] # Get first (and only) frame

        return first_frame_result


    def visualize_objects_presence(self, frame: MatLike, detections) -> Tuple[MatLike, bool]:
        boxes = detections.boxes
        are_there_objects = False

        for box in boxes:
            are_there_objects = True
            x_min, y_min, x_max, y_max = box.xyxy[0]
            cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            object_class_id = self._get_object_class_id(box) 
            object_class_name = yolo_settings.get_class_name(object_class_id)

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

        return frame, are_there_objects


    def _get_object_class_id(self, box) -> int:
        return int(box.cls[0])


    def fit_frame_into_screen(self, frame: MatLike,
                              max_frame_width, max_frame_height,
                              min_frame_width, min_frame_height) -> MatLike:
        if frame is None:
            return None
        
        output_width, output_height = self._fitting_dimensions(frame,
                                                               max_frame_width, max_frame_height,
                                                               min_frame_width, min_frame_height)
        frame = cv.resize(frame, (output_width, output_height), interpolation=cv.INTER_LINEAR)
        return frame


    def _fitting_dimensions(self, frame, max_width, max_height, min_width, min_height) -> Tuple[int, int]:
        height, width = frame.shape[:2]

        # Case 1: frame fits
        if width <= max_width and width >= min_width and \
              height <= max_height and height >= min_height:
            return width, height
        
        # Case 2: either dimension exeeced the limit
        if width > min_width or height > min_height:
            scale_x = max_width / width
            scale_y = max_height / height

            scale = min(scale_x, scale_y)

            new_width = int(width * scale)
            new_height = int(height * scale)

            return new_width, new_height
        
        # Case 3: either dimension is below the limit
        scale_x = min_width / width
        scale_y = min_height / height
        
        scale = max(scale_x, scale_y)

        new_width = int(width * scale)
        new_height = int(height * scale)

        return new_width, new_height
        