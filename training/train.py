import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import set_repo_root

from ultralytics import YOLO
import torch


data_yaml = 'cos.yaml'
epochs = 100
image_size = (640, 640)
batch=4


def main():
    print('Type in model to be trained:')
    model_name = input()
    print('Saved name:')
    saved_as = input()

    model = YOLO(model_name)
    if not torch.cuda.is_available():
        print('Cuda not avaivable. Training would take too longs')
        exit()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device='cuda',
        resume=True,
        save=True
    )

    model.save(saved_as)


if __name__ == '__main__':
    main()