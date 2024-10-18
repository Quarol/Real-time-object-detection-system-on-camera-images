from ultralytics import YOLO


def main():
    model = YOLO('../yolo_models/yolov8n-pretrained-default.pt', verbose=False)
    photo = '../demo_assets/pedestrians.jpg'
    res = model.predict(photo, verbose=False)[0]
    
    boxes = res.boxes
    print(boxes)
        


if __name__ == '__main__':
    main()