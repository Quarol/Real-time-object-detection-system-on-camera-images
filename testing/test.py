import time 
from ultralytics import YOLO
import cv2 as cv


def _measure_time(self, func, *args, **kwargs):
    time1 = time.time()
    func(*args, **kwargs)
    time2 = time.time()
    duration = time2 - time1
    fps = 1 / duration if duration != 0 else 'inf'

    print(f'Duration: {duration}s', end='')
    print(f', FPS: {fps}')


def calculate_average(file_name):
    with open(file_name, 'r') as file:
        # Read all lines and convert them to floats
        numbers = [float(line.strip()) for line in file if line.strip()]
            
    # Calculate the sum and count
    total_sum = sum(numbers)
    count = len(numbers)

    # Calculate the average
    if count > 0:
        average = total_sum / count
    else:
        average = 0

    return average, count


def check_camera_options(cap):
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
        (320, 240)
    ]

    fps_values = [15, 30, 60]

    print("Testing supported resolutions:")
    for width, height in resolutions:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        current_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        current_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        if current_width == width and current_height == height:
            print(f"Supported Resolution: {int(current_width)}x{int(current_height)}")
        else:
            print(f"Resolution {width}x{height} not supported.")


    print("\nTesting supported FPS values:")
    for fps in fps_values:
        cap.set(cv.CAP_PROP_FPS, fps)
        
        current_fps = cap.get(cv.CAP_PROP_FPS)

        if current_fps >= fps - 1 and current_fps <= fps + 1:
            print(f"Supported FPS: {current_fps:.2f}")
        else:
            print(f"FPS {fps} not supported.")


def main():
    cap = cv.VideoCapture(0)
    check_camera_options(cap)

    while True:
        ret, frame = cap.read()
        
        if ret == False:
            print('co do')


if __name__ == '__main__':
    main()