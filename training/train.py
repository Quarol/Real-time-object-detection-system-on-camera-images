from datetime import datetime
from ultralytics import YOLO
import torch

data_yaml = 'COCO_person_dataset.yaml'
epochs = 200
batch = 12
device ='cuda'
optimizer = 'auto'
momentum = 0.937
close_mosaic = 10
single_cls = False
resume = False
save = True
cache = False
verbose = True
amp = False # GTX 1650 broken

def train(model):
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        device=device,
        momentum=momentum,
        close_mosaic=close_mosaic,
        single_cls=single_cls,
        resume=resume,
        save=save,
        cache=cache,
        optimizer=optimizer,
        verbose=verbose,
        amp=amp
    )

    with open('results.txt', 'w') as file:
        file.write(str(results))

    return model


def main():
    print('Type in model to be trained:')
    model_name = input()
    print('Saved name:')
    saved_as = input()

    model = YOLO(model_name)
    if not torch.cuda.is_available():
        print('Cuda not avaivable. Training would take too longs')
        exit()

    print('STARTED TRAINING UUUUUUU')
    with open('trainingTime.txt', 'a') as file:
        file.write(f'Time BEFORE training: {datetime.now()}\n')

    model = train(model)

    print('FINISHED TRAINING UUUUUUU')
    with open('trainingTime.txt', 'a') as file:
        file.write(f'Time AFTER training: {datetime.now()}\n\n')

    model.save(f'../yolo_models/{saved_as}')


if __name__ == '__main__':
    main()