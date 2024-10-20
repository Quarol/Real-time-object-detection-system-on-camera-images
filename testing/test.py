import time 
from ultralytics import YOLO


def _measure_time(self, func, *args, **kwargs):
    time1 = time.time()
    func(*args, **kwargs)
    time2 = time.time()
    duration = time2 - time1
    fps = 1 / duration if duration != 0 else 'inf'

    print(f'Duration: {duration}s', end='')
    print(f', FPS: {fps}')



def main():
    model = YOLO('../yolo_models/yolov8n-pretrained-default.pt', verbose=False)
    photo = '../demo_assets/pedestrians.jpg'
    res = model.predict(photo, verbose=False)[0]
    
    boxes = res.boxes
    print(boxes)
        


if __name__ == '__main__':
    main()